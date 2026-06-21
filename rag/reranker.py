import numpy as np
from langchain_core.documents import Document
from typing import List
from functools import lru_cache


class Reranker:
    def __init__(self):
        self._embedding_model = None

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            from model.factory import get_embed_model
            self._embedding_model = get_embed_model()
        return self._embedding_model

    @lru_cache(maxsize=2000)
    def _get_embedding(self, text: str) -> np.ndarray:
        """缓存嵌入结果"""
        return np.array(self.embedding_model.embed_documents([text])[0])

    def _calculate_similarity(self, query_embedding: np.ndarray, doc_embedding: np.ndarray) -> float:
        dot_product = np.dot(query_embedding, doc_embedding)
        norm1 = np.linalg.norm(query_embedding)
        norm2 = np.linalg.norm(doc_embedding)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        if not documents:
            return []

        query_embedding = self._get_embedding(query)

        scored_docs = []
        for doc in documents:
            text = doc.page_content
            # 跳过空文档
            if not text or len(text.strip()) == 0:
                continue
            # 截断过长的文档（v2 API限制2048字符）
            if len(text) > 2000:
                text = text[:2000]
            doc_embedding = self._get_embedding(text)
            similarity = self._calculate_similarity(query_embedding, doc_embedding)
            scored_docs.append((doc, similarity))

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs[:top_n]]