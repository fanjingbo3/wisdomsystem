import sys
sys.path.insert(0, '.')

from rag.vector_store import VectorStoreService
vs = VectorStoreService()
docs = vs.get_all_documents_with_ids()

print('向量库中的doc_ids (前20个):')
for d in docs[:20]:
    print(d['doc_id'])

print('\n测试用例中的relevant_doc_ids:')
import json
with open('tests/test_cases.json', 'r', encoding='utf-8') as f:
    cases = json.load(f)
    for case in cases[:5]:
        print(f"{case['id']}: {case['relevant_doc_ids']}")
