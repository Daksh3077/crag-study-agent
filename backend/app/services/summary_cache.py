"""Bounded cache of complete section summaries, keyed by exact evidence and prompt."""
from collections import OrderedDict
from hashlib import sha256
import json
import threading
import time

_cache = OrderedDict()
_lock = threading.Lock()
MAX_ENTRIES = 128
TTL_SECONDS = 1800

def section_key(scope, provider, model, system, context):
    return sha256(json.dumps([sorted(scope), provider, model, system, context], ensure_ascii=False).encode()).hexdigest()

def get(key):
    with _lock:
        item = _cache.get(key)
        if item and time.monotonic() - item[0] < TTL_SECONDS:
            _cache.move_to_end(key)
            return item[1]
        _cache.pop(key, None)
    return None

def put(key, value):
    if not value.strip():
        return
    with _lock:
        _cache[key] = (time.monotonic(), value)
        _cache.move_to_end(key)
        while len(_cache) > MAX_ENTRIES:
            _cache.popitem(last=False)
