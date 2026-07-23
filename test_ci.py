"""
CI测试：基础检查（不依赖任何外部服务）
"""
import os
import sys
import unittest


class TestBasic(unittest.TestCase):
    """基础测试"""

    def test_project_structure(self):
        """测试项目结构完整性"""
        required_files = [
            'app.py',
            'requirements.txt',
            'model/factory.py',
            'agent/react_agent.py',
            'agent/router.py',
            'agent/agent_tools.py',
            'agent/tools/agent_tools.py',
            '.gitignore',
            'config/agent.yml',
            'config/rag.yml',
            'config/chroma.yml',
            'config/prompts.yml',
        ]
        
        for f in required_files:
            self.assertTrue(os.path.exists(f), f"Missing required file: {f}")

    def test_python_syntax(self):
        """测试Python文件语法"""
        import ast
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in ['.git', '.venv', '__pycache__', 'node_modules']]
            for f in files:
                if f.endswith('.py'):
                    filepath = os.path.join(root, f)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as file:
                            ast.parse(file.read())
                    except SyntaxError as e:
                        self.fail(f"Syntax error in {filepath}: {e}")

    def test_gitignore_exists(self):
        """测试.gitignore存在"""
        self.assertTrue(os.path.exists('.gitignore'), ".gitignore should exist")


if __name__ == '__main__':
    unittest.main()
