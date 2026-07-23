"""
CI测试：路由分支、工具调用、异常处理（不依赖API Key）
"""
import sys
sys.path.insert(0, '.')

import unittest


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

    def test_get_user_id(self):
        """测试用户ID工具"""
        from agent.tools.agent_tools import get_user_id
        
        result = get_user_id.invoke({})
        self.assertIsInstance(result, str)

    def test_get_current_month(self):
        """测试月份工具"""
        from agent.tools.agent_tools import get_current_month
        
        result = get_current_month.invoke({})
        self.assertIsInstance(result, str)


class TestProjectStructure(unittest.TestCase):
    """测试项目结构完整性"""

    def test_required_files_exist(self):
        """测试必要文件存在"""
        import os
        
        required_files = [
            'app.py',
            'requirements.txt',
            'model/factory.py',
            'agent/react_agent.py',
            'agent/router.py',
            'agent/agent_tools.py',
            'agent/tools/agent_tools.py',
            '.gitignore',
        ]
        
        for f in required_files:
            self.assertTrue(os.path.exists(f), f"Missing required file: {f}")


if __name__ == '__main__':
    unittest.main()
