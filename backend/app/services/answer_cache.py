"""Reuse fully checked Ask answers only for identical questions and evidence."""
from collections import OrderedDict
from copy import deepcopy
from hashlib import sha256
import json
import threading
import time

_cache = OrderedDict()
_lock = threading.Lock()
TTL_SECONDS = 900
MAX_ENTRIES = 128
FIELDS = ('generated_answer', 'cited_source_ids', 'answer_blocks', 'grounded',
          'groundedness_confidence', 'validation_status')


def key(question, scope, context, settings):
    value = [question, sorted(scope), context, settings.llm_provider,
             settings.llm_model, settings.validator_model, settings.groundedness_threshold]
    return sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def clear():
    with _lock:
        _cache.clear()


def get(cache_key):
    with _lock:
        value = _cache.get(cache_key)
        if value and time.monotonic() - value[0] < TTL_SECONDS:
            _cache.move_to_end(cache_key)
            return deepcopy(value[1])
        _cache.pop(cache_key, None)
    return None


def put(cache_key, state):
    if (state.get('validation_status') != 'supported' or not state.get('grounded')
            or state.get('warnings') or state.get('unsupported_claims')
            or state.get('answer_coverage_gaps') or not state.get('cited_source_ids')):
        return
    value = {name: deepcopy(state[name]) for name in FIELDS if name in state}
    with _lock:
        _cache[cache_key] = (time.monotonic(), value)
        _cache.move_to_end(cache_key)
        while len(_cache) > MAX_ENTRIES:
            _cache.popitem(last=False)
