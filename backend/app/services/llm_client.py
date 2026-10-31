"""Provider calls with bounded retries and explicit incomplete-output errors."""
import json
import logging
import re
import time
from functools import lru_cache
from typing import Callable, TypeVar
from app.core.config import get_settings
from app.services.provider_limits import RateLimitDeferred, cooldown, model_lock, observe_headers, retry_delay, wait_for_model, pace_model, wait_for_pacing, quota_details
T = TypeVar("T")

class IncompleteOutputError(RuntimeError):
    pass

class ProviderConfigurationError(ValueError):
    pass

def public_error(exc: Exception) -> str:
    if isinstance(exc, RateLimitDeferred):
        if exc.quota_kind == "daily":
            limit = f" (provider-reported daily limit: {exc.quota_limit})" if exc.quota_limit else ""
            return f"The model's daily quota is exhausted{limit}. It cannot resume after a short wait. Retry after {exc.retry_at}, or configure another available model."
        return f"Provider quota is temporarily exhausted. Retry this request after {exc.retry_at}."
    if isinstance(exc, ProviderConfigurationError):
        return str(exc)
    code = getattr(exc, "status_code", None)
    if code == 429:
        return "The model provider is rate limited. Wait a little and retry."
    if code in (401, 403):
        return "The model provider rejected its configured API key or model access."
    if code == 404:
        return "The configured model is unavailable. Check LLM_MODEL or VISION_MODEL against your provider's available models."
    if isinstance(exc, IncompleteOutputError):
        return "The model output was incomplete. Try a smaller request."
    return "The model request failed. Check the backend logs and provider configuration."

def with_retry(call: Callable[[], T], *, key: str = "default", retries: int | None = None,
               max_wait: float | None = None) -> T:
    settings = get_settings()
    retry_count = settings.provider_retries if retries is None else retries
    remaining_wait = settings.provider_max_wait_seconds if max_wait is None else max_wait
    for attempt in range(retry_count + 1):
        remaining_wait -= wait_for_model(key, remaining_wait)
        try:
            return call()
        except Exception as exc:
            code = getattr(exc, "status_code", None)
            transient = code in (429, 500, 502, 503, 504) or type(exc).__name__ in {"APITimeoutError", "APIConnectionError"}
            if code == 429:
                delay = retry_delay(exc, attempt)
                logging.getLogger(__name__).warning("Provider rate limit for %s; retry in %.1fs", key, delay)
                cooldown(key, delay)
                details = quota_details(exc)
                if details.get("quota_kind") == "daily":
                    raise RateLimitDeferred(delay, **details) from exc
                if attempt >= retry_count or delay > remaining_wait:
                    raise RateLimitDeferred(delay, **details) from exc
                continue
            if not transient or attempt >= retry_count:
                raise
            delay = min(5 * 2 ** attempt, 30)
            if delay > remaining_wait:
                raise
            time.sleep(delay)
            remaining_wait -= delay
    raise RuntimeError("Retry budget exhausted")

def completion_request(provider: str, request: dict, *, ocr: bool = False):
    """Serialize each model's calls and honor quota headers, including image crops."""
    key = provider + ":" + request["model"]
    settings = get_settings()
    def call():
        if ocr:
            wait_for_pacing(key)
            if provider == "gemini":
                # Space request starts, counting the time spent generating the
                # previous transcription toward the configured RPM interval.
                interval = max(settings.ocr_call_delay_seconds, 60 / settings.gemini_ocr_requests_per_minute + .1)
                pace_model(key, interval)
        started = time.perf_counter()
        raw = get_client(provider).chat.completions.with_raw_response.create(**request)
        response = raw.parse()
        reserve = max(getattr(response.usage, "total_tokens", 0) or 0, request.get("max_tokens", 0))
        observe_headers(key, raw.headers, reserve)
        if ocr and provider != "gemini":
            pace_model(key, settings.ocr_call_delay_seconds)
        logging.getLogger(__name__).info("Model %s completed in %.2fs (%s tokens)", request["model"], time.perf_counter() - started, getattr(response.usage, "total_tokens", None))
        return response
    with model_lock(key):
        return with_retry(call, key=key,
                          retries=settings.ocr_provider_retries if ocr else None,
                          max_wait=settings.ocr_max_wait_seconds if ocr else None)

@lru_cache(maxsize=4)
def get_client(provider: str):
    import openai
    settings = get_settings()
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=settings.anthropic_api_key, max_retries=0, timeout=settings.provider_timeout_seconds)
    configurations = {
        "groq": (settings.groq_api_key, "https://api.groq.com/openai/v1"),
        "gemini": (settings.gemini_api_key, "https://generativelanguage.googleapis.com/v1beta/openai/"),
        "openai": (settings.openai_api_key, None),
    }
    if provider not in configurations:
        raise ProviderConfigurationError("Unsupported model provider.")
    key, base_url = configurations[provider]
    if not key:
        raise ProviderConfigurationError(f"Set {provider.upper()}_API_KEY in backend/.env and restart the backend.")
    kwargs = {"base_url": base_url} if base_url else {}
    return openai.OpenAI(api_key=key, max_retries=0, timeout=settings.provider_timeout_seconds, **kwargs)

def chat(system: str, user: str, max_tokens: int = 2000, temperature: float = .1,
         json_mode: bool = False, reasoning_effort: str | None = "low", model_override: str | None = None) -> str:
    settings = get_settings()
    provider, model = settings.llm_provider, model_override or settings.llm_model
    client = get_client(provider)
    if json_mode:
        system += "\nReturn ONLY one valid JSON object, without markdown fences."
    if provider == "anthropic":
        response = with_retry(lambda: client.messages.create(model=model, system=system, messages=[{"role": "user", "content": user}], max_tokens=max_tokens, temperature=temperature), key=provider + ":" + model)
        if response.stop_reason == "max_tokens":
            raise IncompleteOutputError("Completion reached its token limit")
        text = "".join(block.text for block in response.content if block.type == "text")
    else:
        request = dict(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=max_tokens, temperature=temperature)
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        if provider == "groq" and "gpt-oss" in model:
            request["extra_body"] = {"reasoning_effort": reasoning_effort or "low", "include_reasoning": False}
        elif provider == "groq" and "qwen3." in model:
            request["extra_body"] = {"reasoning_effort": "none", "include_reasoning": False}
        response = completion_request(provider, request)
        if response.choices[0].finish_reason != "stop":
            raise IncompleteOutputError("Completion did not finish normally")
        text = response.choices[0].message.content or ""
    if not text.strip():
        raise IncompleteOutputError("Empty completion")
    return text.strip()

def chat_json(system: str, user: str, max_tokens: int = 2000, temperature: float = 0,
              schema: dict | None = None, schema_name: str = "response", reasoning_effort: str = "low",
              model_override: str | None = None) -> dict:
    # Every caller validates with a Pydantic schema, regardless of provider.
    if schema:
        system += "\nRequired JSON schema: " + json.dumps(schema)
    raw = chat(system, user, max_tokens, temperature, True, reasoning_effort, model_override)
    raw = re.sub(r"^\x60\x60\x60(?:json)?\s*|\s*\x60\x60\x60$", "", raw.strip())
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result
