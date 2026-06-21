from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from rag.vector_store import VectorStoreService
from rag.query_rewriter import QueryRewriter
from rag.semantic_checker import SemanticChecker
from rag.bm25_retriever import BM25Retriever
from rag.rrf_fusion import rrf_fusion
from rag.reranker import Reranker
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from database.redis_cache import RedisCache
from utils.cache import query_cache
import hashlib


class RagSummarizeService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._vector_store = None
        self._vector_retriever = None
        self._query_rewriter = None
        self._semantic_checker = None
        self._bm25_retriever = None
        self._reranker = None
        self._model = None
        self._chain = None
        self._cache = None

        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)

    def _get_vector_store(self):
        if self._vector_store is None:
            self._vector_store = VectorStoreService()
        return self._vector_store

    def _get_vector_retriever(self):
        if self._vector_retriever is None:
            self._vector_retriever = self._get_vector_store().get_retriever()
        return self._vector_retriever

    def _get_query_rewriter(self):
        if self._query_rewriter is None:
            self._query_rewriter = QueryRewriter()
        return self._query_rewriter

    def _get_semantic_checker(self):
        if self._semantic_checker is None:
            self._semantic_checker = SemanticChecker()
        return self._semantic_checker

    def _get_bm25_retriever(self):
        if self._bm25_retriever is None:
            self._bm25_retriever = BM25Retriever()
        return self._bm25_retriever

    def _get_reranker(self):
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    def _get_model(self):
        if self._model is None:
            self._model = chat_model.materialize()
        return self._model

    def _get_cache(self):
        if self._cache is None:
            self._cache = RedisCache()
        return self._cache

    def _init_chain(self):
        if self._chain is None:
            self._chain = self.prompt_template | self._get_model() | StrOutputParser()
        return self._chain

    def _hybrid_retrieval_single(self, query: str) -> list[Document]:
        """单个Query的混合召回"""
        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(self._get_vector_retriever().invoke, query)
            bm25_future = executor.submit(self._get_bm25_retriever().retrieve, query)
            vector_docs = vector_future.result()
            bm25_docs = bm25_future.result()
        return rrf_fusion(vector_docs, bm25_docs)

    def _hybrid_retrieval_multi(self, queries: list) -> list[Document]:
        """多路Query召回：并行执行，每个Query各自召回，然后融合所有结果"""
        all_docs = []
        
        # 并行执行所有Query的混合召回
        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            futures = {executor.submit(self._hybrid_retrieval_single, q["text"]): q for q in queries}
            for future in as_completed(futures):
                docs = future.result()
                all_docs.extend(docs)
        
        if not all_docs:
            return []
        
        return rrf_fusion(all_docs, [], k=60, top_n=20)

    def retriever_docs(self, query: str) -> list[Document]:
        """完整的检索流程：多路改写 → 多路召回 → Rerank精排"""
        # 1. 多路改写并过滤（相似度≥0.8）
        valid_rewrites = self._get_query_rewriter().rewrite_multi_with_filter(
            query, 
            num_rewrites=3, 
            threshold=0.8
        )
        
        # 2. 准备所有要召回的Query（包括原Query）
        queries_to_retrieve = [{"text": query, "similarity": 1.0}]
        queries_to_retrieve.extend(valid_rewrites)
        
        # 3. 多路召回
        fused_docs = self._hybrid_retrieval_multi(queries_to_retrieve)
        
        # 4. Rerank精排（使用原Query进行精排）
        reranked_docs = self._get_reranker().rerank(query, fused_docs)
        
        return reranked_docs

    def rag_summarize(self, query: str) -> str:
        cache_key = f"rag:{hashlib.md5(query.encode()).hexdigest()}"

        # 1. 先检查本地内存缓存（最快）
        local_result = query_cache.get(query, service="rag")
        if local_result is not None:
            return local_result

        # 2. 再检查Redis缓存
        cached_result = self._get_cache().get_cache(cache_key)
        if cached_result:
            # 回填本地缓存
            query_cache.set(query, cached_result, service="rag")
            return cached_result

        context_docs = self.retriever_docs(query)

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】: {doc.page_content}\n"

        result = self._init_chain().invoke({
            "input": query,
            "context": context,
        })

        # 同时写入本地缓存和Redis缓存
        query_cache.set(query, result, service="rag")
        self._get_cache().set_cache(cache_key, result, ttl_seconds=86400)
        return result


if __name__ == '__main__':
    rag = RagSummarizeService()
    print(rag.rag_summarize("小户型适合哪些扫地机器人"))