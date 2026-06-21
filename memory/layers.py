from typing import Optional, Dict, Any, List
from datetime import datetime
from database.sqlite_db import SQLiteDatabase
from database.redis_cache import RedisCache


class ShortTermContext:
    def __init__(self, redis: RedisCache):
        self.redis = redis
        self.max_messages = 10

    def add_message(self, session_id: str, role: str, content: str):
        context = self.redis.get_conversation_context(session_id) or []
        context.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.redis.set_conversation_context(session_id, context, self.max_messages)

    def get_context(self, session_id: str) -> str:
        messages = self.redis.get_conversation_context(session_id) or []
        return "\n".join([f"{m['role']}: {m['content']}" for m in messages])


class UserProfileMemory:
    def __init__(self, db: SQLiteDatabase, redis: RedisCache):
        self.db = db
        self.redis = redis

    def get_or_create(self, user_id: str) -> Dict[str, Any]:
        cached = self.redis.get_cached_profile(user_id)
        if cached:
            return cached

        profile = self.db.get_user_profile(user_id)
        if profile:
            self.redis.cache_user_profile(user_id, profile)
            return profile

        new_profile = {
            "user_id": user_id,
            "name": None,
            "occupation": None,
            "preferences": {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        self.db.save_user_profile(user_id, new_profile)
        self.redis.cache_user_profile(user_id, new_profile)
        return new_profile

    def update_preference(self, user_id: str, key: str, value: Any):
        profile = self.get_or_create(user_id)
        profile["preferences"][key] = value
        profile["updated_at"] = datetime.now().isoformat()
        self.db.save_user_profile(user_id, profile)
        self.redis.cache_user_profile(user_id, profile)


class ConversationSummaryMemory:
    def __init__(self, db: SQLiteDatabase, redis: RedisCache):
        self.db = db
        self.redis = redis

    def extract_summary(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_content = " ".join([m["content"] for m in messages])

        keywords = self._extract_keywords(all_content)
        topics = self._extract_topics(all_content)
        summary_text = all_content[:200] + "..." if len(all_content) > 200 else all_content

        return {
            "keywords": keywords,
            "topics": topics,
            "summary_text": summary_text
        }

    def _extract_keywords(self, text: str) -> List[str]:
        common_keywords = ["扫地机器人", "天气", "报告", "用户", "功能", "使用", "问题", "故障", "维护"]
        return [kw for kw in common_keywords if kw in text]

    def _extract_topics(self, text: str) -> List[str]:
        topics = []
        if "报告" in text:
            topics.append("报告生成")
        if "天气" in text:
            topics.append("天气查询")
        if "功能" in text:
            topics.append("产品功能")
        if "故障" in text:
            topics.append("故障排查")
        if "维护" in text:
            topics.append("维护保养")
        if not topics:
            topics.append("普通对话")
        return topics

    def save_summary(self, user_id: str, session_id: str, messages: List[Dict[str, Any]]):
        summary = self.extract_summary(messages)
        summary_id = f"sum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.db.save_conversation_summary(
            summary_id=summary_id,
            user_id=user_id,
            session_id=session_id,
            keywords=summary["keywords"],
            topics=summary["topics"],
            summary_text=summary["summary_text"]
        )

        self.redis.cache_summary(summary_id, summary)

    def get_user_summaries(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        return self.db.get_user_summaries(user_id, limit)


class ExperienceMemory:
    def __init__(self, rag_service):
        self.rag_service = rag_service

    def add_success_case(self, user_id: str, title: str, content: str, metadata: Dict[str, Any] = {}):
        doc_content = f"【成功案例】{title}\n用户ID: {user_id}\n内容: {content}\n元数据: {metadata}"
        self._add_to_rag(doc_content)

    def add_failure_case(self, user_id: str, title: str, content: str, metadata: Dict[str, Any] = {}):
        doc_content = f"【失败案例】{title}\n用户ID: {user_id}\n内容: {content}\n元数据: {metadata}"
        self._add_to_rag(doc_content)

    def _add_to_rag(self, content: str):
        from langchain_core.documents import Document
        doc = Document(page_content=content, metadata={"type": "experience"})
        if hasattr(self.rag_service, 'vector_store'):
            self.rag_service.vector_store.add_documents([doc])
        print(f"已将经验记录存入RAG: {content[:50]}...")

    def query_experience(self, query: str) -> str:
        if hasattr(self.rag_service, 'rag_summarize'):
            return self.rag_service.rag_summarize(query)
        return ""
