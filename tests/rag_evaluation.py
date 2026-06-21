import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any
from rag.rag_service import RagSummarizeService
from rag.vector_store import VectorStoreService


class RAGEvaluator:
    def __init__(self):
        self.rag_service = RagSummarizeService()
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()

    def load_test_cases(self, test_file: str = None) -> List[Dict[str, Any]]:
        """从文件加载刁钻测试用例"""
        if test_file is None:
            test_file = os.path.join(
                os.path.dirname(__file__),
                "test_cases.json"
            )

        with open(test_file, "r", encoding="utf-8") as f:
            test_cases = json.load(f)

        print(f"加载了 {len(test_cases)} 个刁钻测试用例")
        return test_cases

    def evaluate_recall_at_k(self, test_cases: List[Dict[str, Any]], k: int = 5) -> Dict[str, Any]:
        """
        评估召回率 - 使用 doc_id 精确匹配判断是否召回了预期文档
        """
        results = []
        recall_1_count = 0
        recall_3_count = 0
        recall_5_count = 0
        mrr_sum = 0

        for case in test_cases:
            docs = self.rag_service.retriever_docs(case["user_query"])

            retrieved_doc_ids = [doc.metadata.get("doc_id", "") for doc in docs[:k]]

            expected_doc_ids = case.get("relevant_doc_ids", [])

            hit_at_1 = any(doc_id in expected_doc_ids for doc_id in retrieved_doc_ids[:1])
            hit_at_3 = any(doc_id in expected_doc_ids for doc_id in retrieved_doc_ids[:3])
            hit_at_5 = any(doc_id in expected_doc_ids for doc_id in retrieved_doc_ids[:5])

            if hit_at_1:
                recall_1_count += 1
            if hit_at_3:
                recall_3_count += 1
            if hit_at_5:
                recall_5_count += 1

            mrr = 0
            for i, doc_id in enumerate(retrieved_doc_ids[:k]):
                if doc_id in expected_doc_ids:
                    mrr = 1 / (i + 1)
                    break
            mrr_sum += mrr

            hit_position = None
            for i, doc_id in enumerate(retrieved_doc_ids):
                if doc_id in expected_doc_ids:
                    hit_position = i + 1
                    break

            results.append({
                "id": case["id"],
                "user_query": case["user_query"],
                "note": case.get("note", ""),
                "expected_doc_ids": expected_doc_ids,
                "retrieved_doc_ids": retrieved_doc_ids,
                "hit_at_1": hit_at_1,
                "hit_at_3": hit_at_3,
                "hit_at_5": hit_at_5,
                "hit_position": hit_position,
                "mrr": mrr,
            })

        total = len(test_cases)
        return {
            "total_test_cases": total,
            "recall_at_1": recall_1_count / total if total > 0 else 0,
            "recall_at_3": recall_3_count / total if total > 0 else 0,
            "recall_at_5": recall_5_count / total if total > 0 else 0,
            "mrr": mrr_sum / total if total > 0 else 0,
            "detailed_results": results,
        }

    def evaluate_fidelity(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估忠诚度 - 检查回答是否使用召回的文档
        """
        results = []
        faithful_count = 0

        for case in test_cases[:20]:
            try:
                docs = self.rag_service.retriever_docs(case["user_query"])
                retrieved_contents = [doc.page_content for doc in docs]

                if not retrieved_contents:
                    results.append({
                        "id": case["id"],
                        "user_query": case["user_query"],
                        "is_faithful": None,
                        "reason": "未召回到文档",
                    })
                    continue

                answer = self.rag_service.rag_summarize(case["user_query"])

                is_faithful = self._check_faithfulness(answer, retrieved_contents, case)

                results.append({
                    "id": case["id"],
                    "user_query": case["user_query"],
                    "answer": answer[:200] + "..." if len(answer) > 200 else answer,
                    "is_faithful": is_faithful,
                    "retrieved_count": len(docs),
                })

                if is_faithful:
                    faithful_count += 1

            except Exception as e:
                results.append({
                    "id": case["id"],
                    "user_query": case["user_query"],
                    "answer": f"Error: {str(e)}",
                    "is_faithful": False,
                })

        valid_results = [r for r in results if r.get("is_faithful") is not None]
        fidelity_rate = faithful_count / len(valid_results) if valid_results else 0

        return {
            "total_test_cases": len(results),
            "valid_test_cases": len(valid_results),
            "faithful_count": faithful_count,
            "fidelity_rate": fidelity_rate,
            "detailed_results": results,
        }

    def _check_faithfulness(self, answer: str, retrieved_contents: List[str], case: Dict) -> bool:
        """
        检查回答是否忠实于召回的文档
        """
        has_reference = any(keyword in answer for keyword in [
            "参考资料", "根据知识库", "从文档中", "根据文档",
            "知识库中提到", "文档显示", "根据产品说明"
        ])

        keywords_in_answer = 0
        expected_keywords = case.get("expected_keywords", [])
        for keyword in expected_keywords:
            if keyword in answer:
                keywords_in_answer += 1

        all_doc_text = " ".join(retrieved_contents)
        answer_chars = set(answer.replace(" ", ""))
        doc_chars = set(all_doc_text.replace(" ", ""))
        overlap = len(answer_chars & doc_chars) / len(answer_chars) if answer_chars else 0

        if overlap < 0.05 and keywords_in_answer == 0:
            return False

        if has_reference or keywords_in_answer >= 2 or overlap > 0.1:
            return True

        return True

    def run_full_evaluation(self):
        """运行完整评估"""
        print("=" * 60)
        print("RAG 系统评估（刁钻测试用例）")
        print("=" * 60)

        test_cases = self.load_test_cases()

        print("\n[1/2] 评估召回率 (Recall@K)...")
        recall_results = self.evaluate_recall_at_k(test_cases, k=5)

        print("\n[2/2] 评估忠诚度 (Fidelity)...")
        fidelity_results = self.evaluate_fidelity(test_cases)

        return {
            "recall": recall_results,
            "fidelity": fidelity_results,
            "test_cases": test_cases,
        }

    def print_report(self, results: Dict[str, Any]):
        """打印评估报告"""
        print("\n" + "=" * 60)
        print("评估报告")
        print("=" * 60)

        recall = results["recall"]
        print("\n📈 召回率评估 (Recall@K)")
        print(f"   测试用例总数: {recall['total_test_cases']}")
        print(f"   Recall@1: {recall['recall_at_1']:.1%}  (第一名命中)")
        print(f"   Recall@3: {recall['recall_at_3']:.1%}  (前三命中)")
        print(f"   Recall@5: {recall['recall_at_5']:.1%}  (前五命中)")
        print(f"   MRR:      {recall['mrr']:.3f}  (平均倒数排名)")

        failed_cases = [
            r for r in recall["detailed_results"]
            if not r["hit_at_5"]
        ]
        if failed_cases:
            print(f"\n❌ 召回失败案例 (Top-5 未命中):")
            for case in failed_cases[:5]:
                print(f"   - {case['user_query']}")
                print(f"     预期: {case['expected_doc_ids']}")
                print(f"     实际: {case['retrieved_doc_ids']}")

        fidelity = results["fidelity"]
        print("\n🎯 忠诚度评估 (Fidelity)")
        print(f"   有效测试数: {fidelity['valid_test_cases']}")
        print(f"   忠诚度率: {fidelity['fidelity_rate']:.1%}")

        print("\n" + "=" * 60)

    def save_results(self, results: Dict[str, Any], output_file: str = None):
        """保存评估结果"""
        if output_file is None:
            output_file = "rag_evaluation_results.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n评估结果已保存到 {output_file}")


if __name__ == "__main__":
    import os
    os.environ["DASHSCOPE_API_KEY"] = "YOUR_API_KEY"
    evaluator = RAGEvaluator()

    test_cases = evaluator.load_test_cases()
    test_cases = test_cases[:5]

    print("\n[1/2] 评估召回率 (Recall@K)...")
    recall_results = evaluator.evaluate_recall_at_k(test_cases, k=5)
    print(f"Recall@5: {recall_results['recall_at_5']:.1%}")

    print("\n[2/2] 评估忠诚度 (Fidelity)...")
    fidelity_results = evaluator.evaluate_fidelity(test_cases)
    print(f"Fidelity: {fidelity_results['fidelity_rate']:.1%}")
