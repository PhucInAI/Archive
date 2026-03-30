import threading
from datetime import datetime
from typing import Dict, List, TypeVar, Generic, Callable

CACHE_DATA = TypeVar('CACHE_DATA')


class BaseCache(Generic[CACHE_DATA]):
    """
    Don't use this class directly. Use MultiCache or SingleCache instead.
    This class provides a base implementation for in-memory cache that can manage multiple items, each item is
    identified by a key (string) and each item has it own expiration time.
    """

    _max_age_in_seconds: int
    _cache_mutex: threading.Lock()
    _cache: Dict[str, List[CACHE_DATA]]
    _cache_time: Dict[str, datetime]

    def __init__(self, max_age: int):
        self._max_age_in_seconds = max_age
        self._cache_mutex = threading.Lock()
        self._cache = dict()
        self._cache_time = dict()

    def _get(self, key_list: List[str], fetcher: Callable[[List[str]], Dict[str, CACHE_DATA]]) -> Dict[str, CACHE_DATA]:
        if len(key_list) == 0:
            return {}

        result: Dict[str, CACHE_DATA] = dict()

        with self._cache_mutex:
            for key in key_list:
                if key in self._cache and not self._is_expired(key):
                    result[key] = self._cache.get(key)

            missing_keys = key_list - result.keys()
            if len(missing_keys) > 0:
                new_states = fetcher(list(missing_keys))
                result.update(new_states)
                self._update_no_lock(new_states)

        return result

    def _was_loaded(self, key: str) -> bool:
        return self._cache.get(key) is not None

    def _is_expired(self, key: str) -> bool:
        if key not in self._cache_time:
            return True
        age = datetime.now() - self._cache_time.get(key)
        return age.total_seconds() > self._max_age_in_seconds

    def _get_expired_keys(self) -> List[str]:
        return list(filter(lambda key: self._is_expired(key), self._cache_time.keys()))

    def _update(self, data: Dict[str, CACHE_DATA]):
        with self._cache_mutex:
            self._update_no_lock(data)

    def _update_no_lock(self, data: Dict[str, CACHE_DATA]):
        t = datetime.now()
        for key in data.keys():
            self._cache_time[key] = t
        self._cache.update(data)

    def _invalidate(self, key_list: List[str]):
        with self._cache_mutex:
            for key in key_list:
                del self._cache_time[key]
