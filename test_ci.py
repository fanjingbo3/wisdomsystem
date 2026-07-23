"""
CI测试：路由分支、工具调用、异常处理（不依赖API Key）
"""
import sys
sys.path.insert(0, '.')

import unittest
from unittest.mock import patch, MagicMock


class TestRouter(unittest.TestCase):
    """测试路由分支逻辑"""

    def test_l1_simple_patterns(self):
        """L1规则：简单闲聊关键词匹配"""
        from agent.router import classify_query
        
        simple_cases = [
            "你好", "您好", "hi", "hello", "嗨",
            "早上好", "下午好", "晚上好",
            "谢谢", "感谢", "thanks",
            "再见", "拜拜", "bye",
            "你是谁", "你叫什么",
            "你能做什么", "你有什么功能",
            "好的", "收到", "明白了",
        ]
        
        for query in simple_cases:
            result = classify_query(query)
            self.assertEqual(result, "simple", f"Query '{query}' should be classified as 'simple'")

    def test_l1_simple_tool(self):
        """L1规则：单工具意图（天气/位置）"""
        from agent.router import classify_query
        
        tool_cases = [
            "今天天气怎么样", "明天会下雨吗", "温度多少",
            "湿度多少", "我在哪", "我的位置", "城市是哪里",
        ]
        
        for query in tool_cases:
            result = classify_query(query)
            self.assertEqual(result, "simple_tool", f"Query '{query}' should be classified as 'simple_tool'")

    def test_l1_complex(self):
        """L1规则：复杂问题关键词匹配"""
        from agent.router import classify_query
        
        complex_cases = [
            "滚刷多久换一次", "滤网怎么清洁",
            "给我生成使用报告", "月度统计",
            "机器人报错了", "导航出问题",
            "推荐一款扫地机", "吸力不够",
        ]
        
        for query in complex_cases:
            result = classify_query(query)
            self.assertEqual(result, "complex", f"Query '{query}' should be classified as 'complex'")

    def test_l1_cross_domain(self):
        """L1规则：跨域问题（工具+产品关键词）"""
        from agent.router import classify_query
        
        cross_cases = [
            "今天天气适合用机器人吗",
            "下雨了还能扫地吗",
            "北京适合用扫拖机器人吗",
        ]
        
        for query in cross_cases:
            result = classify_query(query)
            self.assertEqual(result, "complex", f"Query '{query}' should be classified as 'complex' (cross-domain)")

    def test_l2_fallback(self):
        """L2分类：模型调用失败时降级为complex"""
        from agent.router import classify_query
        
        with patch('model.factory.get_light_chat_model') as mock_model:
            mock_model.return_value.invoke.side_effect = Exception("Model unavailable")
            
            result = classify_query("随便问一个问题")
            self.assertEqual(result, "complex", "When L2 fails, should fall back to 'complex'")


class TestTools(unittest.TestCase):
    """测试工具函数调用"""

    def test_get_weather(self):
        """测试天气工具"""
        from agent.tools.agent_tools import get_weather
        
        result = get_weather.invoke({"city": "深圳"})
        self.assertIsInstance(result, str)
        self.assertIn("深圳", result)
        self.assertIn("天气", result)

    def test_get_user_location(self):
        """测试位置工具"""
        from agent.tools.agent_tools import get_user_location
        
        result = get_user_location.invoke({})
        self.assertIsInstance(result, str)
        self.assertIn(result, ["深圳", "合肥", "杭州"])

    def test_get_user_id(self):
        """测试用户ID工具"""
        from agent.tools.agent_tools import get_user_id
        
        result = get_user_id.invoke({})
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("10"))

    def test_get_current_month(self):
        """测试月份工具"""
        from agent.tools.agent_tools import get_current_month
        
        result = get_current_month.invoke({})
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("2025-"))

    def test_fetch_external_data_valid(self):
        """测试获取外部数据（有效数据）"""
        from agent.tools.agent_tools import fetch_external_data
        
        result = fetch_external_data.invoke({"user_id": "1001", "month": "2025-01"})
        self.assertIsInstance(result, dict)
        self.assertIn("特征", result)

    def test_fetch_external_data_invalid(self):
        """测试获取外部数据（无效数据）"""
        from agent.tools.agent_tools import fetch_external_data
        
        result = fetch_external_data.invoke({"user_id": "9999", "month": "2025-01"})
        self.assertEqual(result, "")

    def test_fill_context_for_report(self):
        """测试报告上下文填充工具"""
        from agent.tools.agent_tools import fill_context_for_report
        
        result = fill_context_for_report.invoke({})
        self.assertEqual(result, "fill_context_for_report已调用")


class TestAgentToolsErrorHandling(unittest.TestCase):
    """测试Agent工具异常处理"""

    def test_call_knowledge_expert_error(self):
        """测试知识专家调用失败时的异常处理"""
        from agent.agent_tools import call_knowledge_expert
        
        with patch('agent.sub_agents.get_knowledge_expert') as mock_get_expert:
            mock_expert = MagicMock()
            mock_get_expert.return_value = mock_expert
            mock_expert.invoke.side_effect = Exception("Knowledge expert failed")
            
            result = call_knowledge_expert.invoke({"query": "test"})
            self.assertIn("知识专家调用失败", result)

    def test_call_report_expert_error(self):
        """测试报告专家调用失败时的异常处理"""
        from agent.agent_tools import call_report_expert
        
        with patch('agent.sub_agents.get_report_expert') as mock_get_expert:
            mock_expert = MagicMock()
            mock_get_expert.return_value = mock_expert
            mock_expert.invoke.side_effect = Exception("Report expert failed")
            
            result = call_report_expert.invoke({"query": "test"})
            self.assertIn("报告专家调用失败", result)

    def test_call_general_expert_error(self):
        """测试通用专家调用失败时的异常处理"""
        from agent.agent_tools import call_general_expert
        
        with patch('agent.sub_agents.get_general_expert') as mock_get_expert:
            mock_expert = MagicMock()
            mock_get_expert.return_value = mock_expert
            mock_expert.invoke.side_effect = Exception("General expert failed")
            
            result = call_general_expert.invoke({"query": "test"})
            self.assertIn("通用专家调用失败", result)

    def test_rag_service_unavailable(self):
        """测试RAG服务未初始化时的错误处理"""
        from agent.tools.agent_tools import rag_summarize
        
        with patch('agent.tools.agent_tools._get_rag_service') as mock_get_rag:
            mock_get_rag.return_value = None
            
            result = rag_summarize.invoke({"query": "test"})
            self.assertIn("RAG服务未初始化", result)


class TestDataFlow(unittest.TestCase):
    """测试数据流和依赖注入"""

    def test_external_data_file_not_found(self):
        """测试外部数据文件不存在时的异常抛出"""
        from agent.tools.agent_tools import generate_external_data
        
        with patch('agent.tools.agent_tools.agent_conf') as mock_conf:
            mock_conf.__getitem__.return_value = "nonexistent_path.csv"
            
            with self.assertRaises(FileNotFoundError):
                generate_external_data()


if __name__ == '__main__':
    unittest.main()
