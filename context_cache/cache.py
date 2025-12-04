# context_cache/cache.py
from typing import Any, Dict, Hashable

class SimpleKVCache:
    """
    Minimal in-memory cache: key -> value
    Value will be past_key_values (a list/tuple of tensors).
    """

    def __init__(self, max_size: int = 128):
        self.store: Dict[Hashable, Any] = {}
        self.max_size = max_size

    def _evict_if_needed(self):
        if len(self.store) > self.max_size:
            # dumb eviction: pop first key
            first_key = next(iter(self.store.keys()))
            self.store.pop(first_key, None)

    def get(self, key: Hashable):
        return self.store.get(key, None)

    def put(self, key: Hashable, value: Any):
        self.store[key] = value
        self._evict_if_needed()
