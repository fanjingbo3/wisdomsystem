import numpy as np
from model.factory import embed_model, get_embed_model
import threading
from functools import lru_cache
import hashlib


class SemanticChecker:
    def __init__(self):
        self.embedding_model = embed_model

    @lru_cache(maxsize=2000)
    def _get_embedding(self, text: str) -> np.ndarray:
        """缓存嵌入结果"""
        return np.array(self.embedding_model.embed_documents([text])[0])

    def calculate_similarity(self, text1: str, text2: str) -> float:
        # 跳过空文本，截断过长文本（API限制8192字符）
        if not text1 or len(text1.strip()) == 0 or not text2 or len(text2.strip()) == 0:
            return 0.0
        if len(text1) > 8000:
            text1 = text1[:8000]
        if len(text2) > 8000:
            text2 = text2[:8000]
        embedding1 = self._get_embedding(text1)
        embedding2 = self._get_embedding(text2)

        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def calculate_similarity_batch(self, query: str, texts: list[str]) -> list[float]:
        """批量计算相似度"""
        if not texts:
            return []

        query_embedding = self._get_embedding(query)

        similarities = []
        for text in texts:
            # 跳过空文本，截断过长文本（API限制8192字符）
            if not text or len(text.strip()) == 0:
                similarities.append(0.0)
                continue
            if len(text) > 8000:
                text = text[:8000]
            text_embedding = self._get_embedding(text)
            dot_product = np.dot(query_embedding, text_embedding)
            norm = np.linalg.norm(text_embedding)
            if norm == 0:
                similarities.append(0.0)
            else:
                similarities.append(dot_product / (np.linalg.norm(query_embedding) * norm))

        return similarities

    def check_similarity(self, original: str, rewritten: str, threshold: float = 0.8) -> tuple:
        similarity = self.calculate_similarity(original, rewritten)
        return similarity >= threshold, similarity


# 全局单例，避免每次调用都创建新实例
_semantic_checker = None
_checker_lock = threading.Lock()

def get_semantic_checker() -> SemanticChecker:
    global _semantic_checker
    if _semantic_checker is None:
        with _checker_lock:
            if _semantic_checker is None:
                _semantic_checker = SemanticChecker()
    return _semantic_checker