from functools import wraps
from typing import Any, Callable, Dict, Optional
import hashlib
import time
import threading


class QueryCache:
    """查询结果缓存（本地内存缓存，作为Redis缓存的补充）"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = threading.RLock()

    def _get_key(self, query: str, **kwargs) -> str:
        """生成缓存键"""
        key_str = query + str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query: str, **kwargs) -> Optional[Any]:
        """获取缓存"""
        key = self._get_key(query, **kwargs)
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() - entry["timestamp"] < self._ttl:
                    return entry["value"]
                else:
                    del self._cache[key]
        return None

    def set(self, query: str, value: Any, **kwargs) -> None:
        """设置缓存"""
        key = self._get_key(query, **kwargs)
        with self._lock:
            if len(self._cache) >= self._max_size:
                # 删除最旧的缓存
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["timestamp"])
                del self._cache[oldest_key]

            self._cache[key] = {
                "value": value,
                "timestamp": time.time()
            }

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()


def cached(cache_instance: QueryCache, **cache_kwargs):
    """装饰器：缓存函数调用结果"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            query = args[0] if args else ""
            cached_result = cache_instance.get(query, **cache_kwargs)
            if cached_result is not None:
                return cached_result

            result = func(*args, **kwargs)
            cache_instance.set(query, result, **cache_kwargs)
            return result
        return wrapper
    return decorator


# 全局缓存实例
query_cache = QueryCache(max_size=500, ttl_seconds=3600)
