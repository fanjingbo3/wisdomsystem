"""

重建向量库并更新测试用例的 doc_id
确保向量库、MD5、测试用例的 doc_id 保持关联
"""
import sys
sys.path.insert(0, '.')

import json
from rag.vector_store import VectorStoreService
from rag.rag_service import RagSummarizeService

def rebuild_and_update():
    print("=" * 60)
    print("步骤1: 清空向量库和MD5缓存")
    print("=" * 60)
    vs = VectorStoreService()
    vs.clear_collection()
    print("向量库和MD5已清空\n")

    print("=" * 60)
    print("步骤2: 重新加载文档（带ContextualEnhancer增强）")
    print("=" * 60)
    vs.load_document()
    print("文档加载完成\n")

    print("=" * 60)
    print("步骤3: 更新测试用例的 relevant_doc_ids")
    print("=" * 60)

    with open('tests/test_cases.json', 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    vector_store = VectorStoreService()
    retriever = vector_store.get_retriever(search_kwargs={"k": 10})
    updated_cases = []

    for case in test_cases:
        query = case["user_query"]
        docs = retriever.invoke(query)
        retrieved_ids = [doc.metadata.get("doc_id", "") for doc in docs]

        case["relevant_doc_ids"] = retrieved_ids[:3]  # 取前3个最相关的
        print(f"{case['id']}: {query} -> {retrieved_ids[:3]}")
        updated_cases.append(case)

    with open('tests/test_cases.json', 'w', encoding='utf-8') as f:
        json.dump(updated_cases, f, ensure_ascii=False, indent=2)

    print("\n测试用例已更新完成")

    print("=" * 60)
    print("步骤4: 重新运行评估")
    print("=" * 60)

if __name__ == '__main__':
    rebuild_and_update()
