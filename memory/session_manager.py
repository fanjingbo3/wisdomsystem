import json
import os
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any
from database.redis_cache import RedisCache
from utils.logger_handler import logger


class SessionManager:
    _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="session_io")
    
    def __init__(self, redis: RedisCache = None, storage_dir: str = "data/sessions"):
        self.redis = redis if redis else RedisCache()
        self.storage_dir = storage_dir
        self._session_user_map: Dict[str, str] = {}
        self._session_cache: Dict[str, dict] = {}
        self._session_cache_expiry: Dict[str, datetime] = {}
        self._cache_lock = threading.Lock()
        
        os.makedirs(storage_dir, exist_ok=True)

    def create_session(self, user_id: str) -> str:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self._cache_lock:
            self._session_user_map[session_id] = user_id
        
        self.redis.set_session_user(session_id, user_id)
        self.redis.add_user_session(user_id, session_id)
        
        return session_id

    def _save_session_sync(self, session_id: str, messages: List[Dict[str, Any]], user_id: str = None):
        if user_id is None:
            user_id = self._session_user_map.get(session_id)
        if user_id is None:
            user_id = self.redis.get_session_user(session_id)
        if user_id is None:
            user_id = "unknown"

        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "messages": messages,
            "saved_at": datetime.now().isoformat()
        }

        file_path = os.path.join(self.storage_dir, f"{session_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

    def save_session(self, session_id: str, messages: List[Dict[str, Any]], user_id: str = None):
        with self._cache_lock:
            self._session_cache[session_id] = {
                "session_id": session_id,
                "user_id": user_id,
                "messages": messages,
                "saved_at": datetime.now().isoformat()
            }
            self._session_cache_expiry[session_id] = datetime.now()
        
        self._executor.submit(self._save_session_sync, session_id, messages, user_id)

    def load_session(self, session_id: str) -> Optional[dict]:
        with self._cache_lock:
            if session_id in self._session_cache:
                if datetime.now() - self._session_cache_expiry.get(session_id, datetime.min) < timedelta(minutes=5):
                    return self._session_cache[session_id]
        
        file_path = os.path.join(self.storage_dir, f"{session_id}.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                with self._cache_lock:
                    self._session_cache[session_id] = data
                    self._session_cache_expiry[session_id] = datetime.now()
                return data
        
        return None

    def list_user_sessions(self, user_id: str) -> List[dict]:
        session_ids = self.redis.get_user_sessions(user_id)

        if not session_ids:
            session_ids = self._scan_sessions_from_files()

        sessions = []
        for session_id in session_ids:
            cached = None
            with self._cache_lock:
                if session_id in self._session_cache:
                    cached = self._session_cache[session_id]
            
            if cached and cached.get("user_id") == user_id:
                last_msg = cached["messages"][-1]["content"][:30] if cached["messages"] else "空会话"
                sessions.append({
                    "session_id": session_id,
                    "preview": last_msg,
                    "saved_at": cached.get("saved_at", "")
                })
            else:
                file_path = os.path.join(self.storage_dir, f"{session_id}.json")
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if data.get("user_id") == user_id:
                                last_msg = data["messages"][-1]["content"][:30] if data["messages"] else "空会话"
                                sessions.append({
                                    "session_id": session_id,
                                    "preview": last_msg,
                                    "saved_at": data.get("saved_at", "")
                                })
                    except Exception as e:
                        logger.warning(f"[SessionManager] 读取会话文件失败: {session_id}, {str(e)}")

        return sessions

    def _scan_sessions_from_files(self) -> List[str]:
        session_ids = []
        if os.path.exists(self.storage_dir):
            for filename in os.listdir(self.storage_dir):
                if filename.endswith(".json"):
                    session_ids.append(filename.replace(".json", ""))
        return sorted(session_ids, reverse=True)