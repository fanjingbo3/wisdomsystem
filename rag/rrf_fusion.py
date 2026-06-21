from langchain_core.documents import Document
from typing import List

def rrf_fusion(
    vector_results: List[Document],
    bm25_results: List[Document],
    k: int = 60,
    top_n: int = 10
) -> List[Document]:
    doc_scores = {}

    for rank, doc in enumerate(vector_results, 1):
        doc_id = hash(doc.page_content)
        if doc_id not in doc_scores:
            doc_scores[doc_id] = {"doc": doc, "score": 0}
        doc_scores[doc_id]["score"] += 1 / (k + rank)

    for rank, doc in enumerate(bm25_results, 1):
        doc_id = hash(doc.page_content)
        if doc_id not in doc_scores:
            doc_scores[doc_id] = {"doc": doc, "score": 0}
        doc_scores[doc_id]["score"] += 1 / (k + rank)

    sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_docs[:top_n]]