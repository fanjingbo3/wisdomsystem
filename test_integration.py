"""
测试基础功能
"""
import sys
sys.path.insert(0, '.')

import os
import unittest


class TestBasic(unittest.TestCase):
    
    def test_env_not_exposed(self):
        """测试敏感文件不存在"""
        self.assertFalse(os.path.exists('.env'), "ERROR: .env file should not be committed!")
    
    def test_project_structure(self):
        """测试项目结构完整性"""
        required_files = [
            'app.py',
            'requirements.txt',
            'model/factory.py',
            'agent/react_agent.py',
            'rag/vector_store.py',
            '.gitignore',
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


if __name__ == '__main__':
    unittest.main()