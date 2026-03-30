from typing import Dict, List, TypeVar, Callable, Generic

from utility.cache.base_cache import BaseCache

CACHE_DATA = TypeVar('CACHE_DATA')


class MultiCache(BaseCache[CACHE_DATA], Generic[CACHE_DATA]):

    def __init__(self, max_age: int):
        super().__init__(max_age)

    def get(self, key_list: List[str], fetcher: Callable[[List[str]], Dict[str, CACHE_DATA]]) -> Dict[str, CACHE_DATA]:
        return super()._get(key_list=key_list, fetcher=fetcher)

    def was_loaded(self, key: str) -> bool:
        return super()._was_loaded(key)

    def get_expired_keys(self) -> List[str]:
        return super()._get_expired_keys()

    def update(self, data: Dict[str, CACHE_DATA]):
        super()._update(data=data)

    def invalidate(self, key_list: List[str]):
        super()._invalidate(key_list=key_list)
