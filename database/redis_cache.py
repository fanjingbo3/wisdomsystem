import json
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from utils.logger_handler import logger


class SimpleCache:
    def __init__(self):
        self.cache = {}
        self.expiry = {}

    def set(self, key: str, value: str):
        self.cache[key] = value
        self.expiry[key] = datetime.now() + timedelta(hours=2)

    def get(self, key: str) -> Optional[str]:
        if key not in self.cache:
            return None
        if datetime.now() > self.expiry[key]:
            del self.cache[key]
            del self.expiry[key]
            return None
        return self.cache[key]

    def expire(self, key: str, ttl: timedelta):
        if key in self.cache:
            self.expiry[key] = datetime.now() + ttl

    def lpush(self, key: str, value: str):
        if key not in self.cache:
            self.cache[key] = []
        self.cache[key].insert(0, value)
        self.expiry[key] = datetime.now() + timedelta(hours=2)

    def lrange(self, key: str, start: int, end: int) -> List[str]:
        if key not in self.cache:
            return []
        if datetime.now() > self.expiry[key]:
            del self.cache[key]
            del self.expiry[key]
            return []
        items = self.cache[key]
        if end == -1:
            end = len(items)
        return items[start:end]

    def ltrim(self, key: str, start: int, end: int):
        if key in self.cache:
            self.cache[key] = self.cache[key][start:end+1]


class RedisCache:
    def __init__(self):
        from utils.config_handler import agent_conf
        
        redis_conf = agent_conf.get("redis", {})
        self.host = redis_conf.get("host", os.getenv("REDIS_HOST", "localhost"))
        self.port = int(redis_conf.get("port", os.getenv("REDIS_PORT", 6379)))
        self.db = int(redis_conf.get("db", os.getenv("REDIS_DB", 0)))
        self.connect_timeout = float(redis_conf.get("connect_timeout", 0.5))
        self.socket_timeout = float(redis_conf.get("socket_timeout", 1))
        
        self._local_cache = {}
        self._local_cache_expiry = {}
        self._local_cache_lock = threading.Lock()
        
        self.client = SimpleCache()
        self.use_redis = False
        
        self._init_async()

    def _init_async(self):
        """异步初始化Redis连接，避免阻塞启动"""
        def _init():
            try:
                import redis
                self.client = redis.Redis(
                    host=self.host, 
                    port=self.port, 
                    db=self.db, 
                    decode_responses=True,
                    socket_connect_timeout=self.connect_timeout, 
                    socket_timeout=self.socket_timeout
                )
                self.client.ping()
                self.use_redis = True
                logger.info(f"[RedisCache] 成功连接Redis: {self.host}:{self.port}")
            except Exception as e:
                self.client = SimpleCache()
                self.use_redis = False
                logger.warning(f"[RedisCache] Redis连接失败，使用本地缓存: {str(e)}")
        
        threading.Thread(target=_init, daemon=True).start()

    def _get_local_cache(self, key: str) -> Optional[str]:
        """从本地内存缓存获取"""
        with self._local_cache_lock:
            if key not in self._local_cache:
                return None
            if datetime.now() > self._local_cache_expiry.get(key, datetime.min):
                del self._local_cache[key]
                del self._local_cache_expiry[key]
                return None
            return self._local_cache[key]

    def _set_local_cache(self, key: str, value: str, ttl: timedelta = timedelta(minutes=5)):
        """设置本地内存缓存"""
        with self._local_cache_lock:
            self._local_cache[key] = value
            self._local_cache_expiry[key] = datetime.now() + ttl

    def set_conversation_context(self, session_id: str, messages: List[Dict[str, Any]], max_messages: int = 10):
        truncated = messages[-max_messages:]
        value = json.dumps(truncated)
        self._set_local_cache(f"context:{session_id}", value, timedelta(hours=2))
        
        if self.client:
            try:
                self.client.set(f"context:{session_id}", value)
                self.client.expire(f"context:{session_id}", timedelta(hours=2))
            except Exception:
                pass

    def get_conversation_context(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        cached = self._get_local_cache(f"context:{session_id}")
        if cached:
            return json.loads(cached)
        
        if self.client:
            try:
                data = self.client.get(f"context:{session_id}")
                if data:
                    self._set_local_cache(f"context:{session_id}", data, timedelta(hours=2))
                    return json.loads(data)
            except Exception:
                pass
        
        return None

    def cache_user_profile(self, user_id: str, profile: Dict[str, Any]):
        value = json.dumps(profile)
        self._set_local_cache(f"profile:{user_id}", value, timedelta(hours=24))
        
        if self.client:
            try:
                self.client.set(f"profile:{user_id}", value)
                self.client.expire(f"profile:{user_id}", timedelta(hours=24))
            except Exception:
                pass

    def get_cached_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        cached = self._get_local_cache(f"profile:{user_id}")
        if cached:
            return json.loads(cached)
        
        if self.client:
            try:
                data = self.client.get(f"profile:{user_id}")
                if data:
                    self._set_local_cache(f"profile:{user_id}", data, timedelta(hours=24))
                    return json.loads(data)
            except Exception:
                pass
        
        return None

    def cache_summary(self, summary_id: str, summary: Dict[str, Any]):
        value = json.dumps(summary)
        self._set_local_cache(f"summary:{summary_id}", value, timedelta(hours=12))
        
        if self.client:
            try:
                self.client.set(f"summary:{summary_id}", value)
                self.client.expire(f"summary:{summary_id}", timedelta(hours=12))
            except Exception:
                pass

    def set_session_user(self, session_id: str, user_id: str):
        self._set_local_cache(f"session:{session_id}:user", user_id, timedelta(hours=2))
        
        if self.client:
            try:
                self.client.set(f"session:{session_id}:user", user_id)
                self.client.expire(f"session:{session_id}:user", timedelta(hours=2))
            except Exception:
                pass

    def get_session_user(self, session_id: str) -> Optional[str]:
        cached = self._get_local_cache(f"session:{session_id}:user")
        if cached:
            return cached
        
        if self.client:
            try:
                data = self.client.get(f"session:{session_id}:user")
                if data:
                    self._set_local_cache(f"session:{session_id}:user", data, timedelta(hours=2))
                    return data
            except Exception:
                pass
        
        return None

    def add_user_session(self, user_id: str, session_id: str):
        if self.client:
            try:
                self.client.lpush(f"user:{user_id}:sessions", session_id)
                self.client.ltrim(f"user:{user_id}:sessions", 0, 9)
            except Exception:
                pass

    def get_user_sessions(self, user_id: str) -> List[str]:
        cached = self._get_local_cache(f"user:{user_id}:sessions")
        if cached:
            return json.loads(cached)
        
        if self.client:
            try:
                data = self.client.lrange(f"user:{user_id}:sessions", 0, -1)
                if data:
                    self._set_local_cache(f"user:{user_id}:sessions", json.dumps(data), timedelta(minutes=1))
                    return data
            except Exception:
                pass
        
        return []

    def set_cache(self, key: str, value: str, ttl_seconds: int = 86400):
        self._set_local_cache(key, value, timedelta(seconds=ttl_seconds))
        
        if self.client:
            try:
                self.client.set(key, value)
                self.client.expire(key, timedelta(seconds=ttl_seconds))
            except Exception:
                pass

    def get_cache(self, key: str) -> Optional[str]:
        cached = self._get_local_cache(key)
        if cached:
            return cached
        
        if self.client:
            try:
                data = self.client.get(key)
                if data:
                    self._set_local_cache(key, data)
                    return data
            except Exception:
                pass
        
        return None