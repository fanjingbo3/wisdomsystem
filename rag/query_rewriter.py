from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
import re
import jieba


class QueryRewriter:
    def __init__(self):
        self.light_model = chat_model.materialize()
        self.prompt_template = PromptTemplate.from_template("""
请将用户的问题从三个不同角度改写成适合向量检索的形式，保证多样性且不偏离意图。

三个改写角度：
1. 同义替换版：将关键词替换为同义词或专业术语（如“咋”→“如何”，“豆子”→“颗粒物”）
2. 句式转换版：改变句式结构，如将疑问句转换为陈述句或解决方案式（如“噪音太大”→“如何解决噪音大问题”）
3. 意图补全版：补充省略的主语和上下文信息（如“咋充电”→“扫地机器人如何充电”）

原始问题：{query}

请按以下格式输出三个改写结果：
【同义替换版】：xxx
【句式转换版】：xxx
【意图补全版】：xxx
""")
        self.chain = self.prompt_template | self.light_model | StrOutputParser()

    def _quick_similarity(self, text1: str, text2: str) -> float:
        """快速相似度计算（无需API调用）"""
        set1 = set(text1)
        set2 = set(text2)
        jaccard = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

        len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2)) if max(len(text1), len(text2)) > 0 else 0

        keywords1 = set(jieba.lcut(text1))
        keywords2 = set(jieba.lcut(text2))
        keyword_overlap = len(keywords1 & keywords2) / len(keywords1 | keywords2) if keywords1 | keywords2 else 0

        return 0.4 * jaccard + 0.3 * len_ratio + 0.3 * keyword_overlap

    @lru_cache(maxsize=1000)
    def rewrite(self, query: str) -> str:
        return self.chain.invoke({"query": query})

    def _parse_rewrites(self, output: str) -> list:
        """解析三路改写结果"""
        rewrites = []
        patterns = [
            (r'【同义替换版】[：:]\s*(.+)', '同义替换'),
            (r'【句式转换版】[：:]\s*(.+)', '句式转换'),
            (r'【意图补全版】[：:]\s*(.+)', '意图补全'),
        ]
        for pattern, rewrite_type in patterns:
            match = re.search(pattern, output)
            if match:
                text = match.group(1).strip()
                if text:
                    rewrites.append({"text": text, "type": rewrite_type})
        return rewrites

    def rewrite_multi_with_filter(self, query: str, num_rewrites: int = 3, threshold: float = 0.8):
        """多路改写并过滤，使用三角度提示词一次性改写，然后并行校验语义相似度"""
        from rag.semantic_checker import get_semantic_checker

        checker = get_semantic_checker()

        # 一次性获取三路改写结果
        output = self.chain.invoke({"query": query})
        parsed_rewrites = self._parse_rewrites(output)

        if not parsed_rewrites:
            return []

        filtered_rewrites = []
        for rewrite in parsed_rewrites:
            quick_score = self._quick_similarity(query, rewrite["text"])
            if quick_score >= 0.4:
                filtered_rewrites.append(rewrite)

        if not filtered_rewrites:
            return []

        # 并行校验语义相似度
        def _check_similarity(rewrite_item):
            similarity = checker.calculate_similarity(query, rewrite_item["text"])
            return {**rewrite_item, "similarity": similarity}

        valid_rewrites = []
        with ThreadPoolExecutor(max_workers=len(filtered_rewrites)) as executor:
            futures = {executor.submit(_check_similarity, item): item for item in filtered_rewrites}
            for future in as_completed(futures):
                result = future.result()
                if result["similarity"] >= threshold:
                    valid_rewrites.append(result)

        return valid_rewrites