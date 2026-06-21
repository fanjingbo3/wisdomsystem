import sys
sys.path.insert(0, '.')

import json
from rag.rag_service import RagSummarizeService

with open('tests/test_cases.json', 'r', encoding='utf-8') as f:
    test_cases = json.load(f)

rag_service = RagSummarizeService()

for case in test_cases:
    query = case["user_query"]
    docs = rag_service.retriever_docs(query)
    retrieved_ids = [doc.metadata.get("doc_id", "") for doc in docs]
    case["relevant_doc_ids"] = retrieved_ids[:3]
    print(f"{case['id']}: '{query}' -> {retrieved_ids[:3]}")

with open('tests/test_cases.json', 'w', encoding='utf-8') as f:
    json.dump(test_cases, f, ensure_ascii=False, indent=2)

print("\n测试用例已更新完成！")
