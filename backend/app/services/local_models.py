"""Load cached models once, serializing initialization across model types."""
import threading
from functools import lru_cache, wraps

_load_lock = threading.RLock()


def cached_model(loader):
    cached = lru_cache(maxsize=1)(loader)

    @wraps(loader)
    def get(*args, **kwargs):
        # lru_cache alone allows duplicate initialization on concurrent misses.
        with _load_lock:
            return cached(*args, **kwargs)

    get.cache_clear = cached.cache_clear
    return get


def prefer_local(factory, model_name):
    try:
        return factory(model_name, local_files_only=True)
    except OSError:
        # First installation may need to download weights. Other initialization
        # errors must remain visible rather than triggering a second load.
        return factory(model_name)
