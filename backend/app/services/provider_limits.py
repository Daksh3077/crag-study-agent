"""Shared model cooldowns. Never retry before the provider's stated reset."""
import math
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from email.utils import parsedate_to_datetime

_wait_callback = ContextVar("provider_wait_callback", default=None)
_quota_callback = ContextVar('provider_quota_status', default=None)
_lock = threading.Lock()
_model_locks = {}
_cooldowns = {}
_pacing_deadlines = {}

class RateLimitDeferred(RuntimeError):
    def __init__(self, seconds: float, *, quota_kind: str = "rate_limit", quota_limit: str | None = None):
        self.quota_kind = quota_kind
        self.quota_limit = quota_limit
        self.retry_after = max(1, math.ceil(seconds))
        self.retry_at = datetime.fromtimestamp(time.time() + self.retry_after, timezone.utc).isoformat()
        super().__init__("Provider quota is not yet available.")

def quota_details(exc) -> dict:
    """Read Google's quota type, which can contradict its generic short RetryInfo."""
    body = getattr(exc, "body", None)
    if isinstance(body, list):
        body = body[0] if body else {}
    if not isinstance(body, dict):
        return {}
    body = body.get("error", body)
    if not isinstance(body, dict):
        return {}
    result = {}
    for detail in body.get("details", []) or []:
        if not isinstance(detail, dict) or not detail.get("@type", "").endswith("google.rpc.QuotaFailure"):
            continue
        for violation in detail.get("violations", []) or []:
            if not isinstance(violation, dict):
                continue
            identifier = str(violation.get("quotaId", "")).lower()
            kind = "daily" if "perday" in identifier else "minute" if "perminute" in identifier else "rate_limit"
            if kind == "daily" or not result:
                result = {"quota_kind": kind, "quota_limit": violation.get("quotaValue")}
    return result

def gemini_daily_reset_seconds(now: datetime | None = None) -> float:
    """Gemini daily quotas reset at midnight Pacific, including daylight saving time."""
    try:
        pacific = ZoneInfo("America/Los_Angeles")
    except ZoneInfoNotFoundError:
        # Existing Windows installs must install requirements.txt for exact reset times.
        return 86400
    now = (now or datetime.now(timezone.utc)).astimezone(pacific)
    tomorrow = (now + timedelta(days=1)).date()
    reset = datetime.combine(tomorrow, datetime.min.time(), tzinfo=pacific)
    return max(1, reset.timestamp() - now.timestamp())

def quota_matches(page, provider: str, model: str) -> bool:
    # Reports written before provider selection existed came from Groq Qwen.
    return (page.get("ocr_provider", "groq") == provider
            and page.get("ocr_model", "qwen/qwen3.8-27b") == model)

def saved_retry_seconds(pages, provider: str | None = None, model: str | None = None) -> float:
    """Honor persisted quota deadlines even when process-local cooldowns are lost."""
    now = datetime.now(timezone.utc)
    remaining = 0.0
    for page in pages:
        if provider is not None and not quota_matches(page, provider, model):
            continue
        if page.get("status") != "pending" and page.get("error_code") != "rate_limit":
            continue
        try:
            deadline = datetime.fromisoformat(page.get("retry_at", "").replace("Z", "+00:00"))
            if deadline.tzinfo is not None:
                remaining = max(remaining, (deadline - now).total_seconds())
        except (ValueError, TypeError, AttributeError):
            continue
    return remaining

@contextmanager
def observe_quota(callback):
    token = _quota_callback.set(callback)
    try:
        yield
    finally:
        _quota_callback.reset(token)


@contextmanager
def observe_waits(callback):
    previous = _wait_callback.get()
    def notify(seconds):
        if previous is not None:
            previous(seconds)
        callback(seconds)
    token = _wait_callback.set(notify)
    try:
        yield
    finally:
        _wait_callback.reset(token)

def duration_seconds(value) -> float:
    value = str(value or "").strip()
    try:
        return max(0, float(value))
    except ValueError:
        pass
    units = {"ms": .001, "s": 1, "m": 60, "h": 3600, "d": 86400}
    parts = re.findall(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h|d)", value, re.I)
    if parts:
        return sum(float(number) * units[unit.lower()] for number, unit in parts)
    try:
        return max(0, parsedate_to_datetime(value).timestamp() - time.time())
    except (ValueError, TypeError, OverflowError):
        return 0

def retry_delay(exc, attempt=0) -> float:
    headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
    delay = duration_seconds(headers.get("retry-after"))
    message = str(exc).lower()
    if quota_details(exc).get("quota_kind") == "daily":
        # A real 20-RPD rejection also carried RetryInfo=20s. Retrying every
        # minute cannot replenish a daily quota, even when Google says "retry".
        return max(delay, gemini_daily_reset_seconds()) + 1
    # Groq includes reset durations in some error bodies instead of Retry-After.
    match = re.search(r"(?:try again|retry) in\s+([\d.dhms\s]+)", message)
    if match:
        delay = max(delay, duration_seconds(match.group(1)))
    # Gemini returns google.rpc.RetryInfo in the JSON error's details array.
    body = getattr(exc, "body", None)
    if isinstance(body, list) and body:
        body = body[0]
    if isinstance(body, dict):
        body = body.get("error", body)
        if isinstance(body, dict):
            for detail in body.get("details", []) or []:
                if isinstance(detail, dict) and detail.get("@type", "").endswith("google.rpc.RetryInfo"):
                    delay = max(delay, duration_seconds(detail.get("retryDelay")))
    for kind in ("requests", "tokens"):
        remaining = headers.get("x-ratelimit-remaining-" + kind)
        if remaining is not None:
            try:
                exhausted = float(remaining) <= 0
            except (ValueError, TypeError):
                exhausted = False
            if exhausted:
                delay = max(delay, duration_seconds(headers.get("x-ratelimit-reset-" + kind)))
    if not delay:
        daily = any(word in message for word in ("tokens per day", "requests per day", "(tpd)", "(rpd)", "daily", "perday", "per_day"))
        delay = 86400 if daily else max(60, 5 * 2 ** attempt)
    return delay + 1  # Allow for clock/rounding differences.

def model_lock(key):
    with _lock:
        return _model_locks.setdefault(key, threading.Lock())

def cooldown(key, seconds):
    with _lock:
        _cooldowns[key] = max(_cooldowns.get(key, 0), time.monotonic() + seconds)

def pace_model(key, seconds):
    """Local request spacing is separate from a provider-imposed quota pause."""
    with _lock:
        _pacing_deadlines[key] = max(_pacing_deadlines.get(key, 0), time.monotonic() + seconds)

def wait_for_pacing(key):
    with _lock:
        until = _pacing_deadlines.get(key, 0)
    while True:
        remaining = until - time.monotonic()
        if remaining <= 0:
            return
        # Ordinary pacing should not announce quota exhaustion or write progress.
        time.sleep(min(remaining, 30))

def wait_for_model(key, budget):
    with _lock:
        remaining = max(0, _cooldowns.get(key, 0) - time.monotonic())
    if remaining > budget:
        raise RateLimitDeferred(remaining)
    if remaining > .05:
        callback = _wait_callback.get()
        if callback:
            callback(remaining)
        # Short sleeps permit orderly interruption; the cooldown is shared by all calls.
        until = time.monotonic() + remaining
        status_callback = _quota_callback.get()
        if status_callback:
            status_callback(key, remaining)
        try:
            while True:
                delay = until - time.monotonic()
                if delay <= 0:
                    break
                time.sleep(min(delay, 30))
        finally:
            if status_callback:
                status_callback(key, None)
    return remaining

def observe_headers(key, headers, token_reserve):
    for kind in ("requests", "tokens"):
        try:
            remaining = float(headers.get("x-ratelimit-remaining-" + kind, "inf"))
        except (ValueError, TypeError):
            continue
        # Do not wait for a full refill just because the previous response was larger
        # than the remaining balance. A 429 supplies the actual next-request wait.
        if remaining <= 0:
            delay = duration_seconds(headers.get("x-ratelimit-reset-" + kind))
            if delay:
                cooldown(key, delay + 1)
