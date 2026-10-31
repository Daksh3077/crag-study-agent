"""Bounded, private request progress for the supported single-worker deployment."""
import threading
import time

_lock = threading.Lock()
_requests = {}
TTL = 1800
MAX_ENTRIES = 1000


def begin(owner, request_id):
    now = time.time()
    with _lock:
        for key in list(_requests):
            if _requests[key]['updated'] + TTL < now:
                del _requests[key]
        key = (owner, request_id)
        if key in _requests:
            raise ValueError('This request ID has already been used.')
        if len(_requests) >= MAX_ENTRIES:
            finished = [k for k, r in _requests.items() if r['phase'] in {'complete', 'failed'}]
            if not finished:
                raise OverflowError('Too many active questions. Please retry shortly.')
            del _requests[min(finished, key=lambda k: _requests[k]['updated'])]
        _requests[key] = dict(phase='processing', updated=now)


def quota(owner, request_id, model_key, seconds):
    with _lock:
        row = _requests.get((owner, request_id))
        if row is None or row['phase'] in {'complete', 'failed'}:
            return
        row.update(updated=time.time(), phase='quota_wait' if seconds is not None else 'processing')
        row['retry_at'] = time.time() + seconds if seconds is not None else None
        row['provider'] = {'groq':'Groq', 'gemini':'Gemini', 'openai':'OpenAI', 'anthropic':'Anthropic'}.get(model_key.split(':')[0], 'Model provider')


def finish(owner, request_id, failed=False):
    with _lock:
        row = _requests.get((owner, request_id))
        if row:
            row.update(phase='failed' if failed else 'complete', updated=time.time(), retry_at=None)


def get(owner, request_id):
    with _lock:
        row = _requests.get((owner, request_id))
        if row is None or row['updated'] + TTL < time.time():
            return None
        return dict(phase=row['phase'], provider=row.get('provider'),
                    retry_after_seconds=max(0, (row.get('retry_at') or 0) - time.time()))
