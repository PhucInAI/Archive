from typing import Callable, Optional, Generic

from utility.cache.base_cache import BaseCache, CACHE_DATA

_DUMMY_KEY = 'single-item-cache'


class SingleCache(BaseCache[CACHE_DATA], Generic[CACHE_DATA]):

    def __init__(self, max_age: int):
        super().__init__(max_age)

    def get(self, fetcher: Callable[[], CACHE_DATA]) -> Optional[CACHE_DATA]:
        return super()._get(key_list=[_DUMMY_KEY], fetcher=lambda k: {_DUMMY_KEY: fetcher()}).get(_DUMMY_KEY)

    def was_loaded(self) -> bool:
        return super()._was_loaded(_DUMMY_KEY)

    def is_expired(self) -> bool:
        return super()._is_expired(_DUMMY_KEY)

    def update(self, data: CACHE_DATA):
        super()._update({_DUMMY_KEY: data})

    def invalidate(self):
        super()._invalidate([_DUMMY_KEY])
