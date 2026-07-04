"""直接测试通义千问 API 是否正常，定位 HTTP 错误的具体原因。"""
import os
import sys
import time
import dashscope

sys.path.insert(0, ".")

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

print(f"DASHSCOPE_API_KEY: {os.getenv('DASHSCOPE_API_KEY', 'NOT SET')[:10]}...")

# 测试 1: 直接用 dashscope SDK 调 qwen-max
print("\n=== 测试 1: dashscope SDK 直接调 qwen-max ===")
try:
    resp = dashscope.Generation.call(
        model="qwen-max",
        messages=[{"role": "user", "content": "你好，用一句话回答"}],
        result_format="message",
    )
    print(f"status_code: {resp.status_code}")
    print(f"code: {resp.code}")
    print(f"message: {resp.message}")
    if resp.status_code == 200:
        print(f"回答: {resp.output.choices[0].message.content[:100]}")
    else:
        print(f"完整响应: {resp}")
except Exception as e:
    print(f"异常: {type(e).__name__}: {e}")

# 测试 2: 用 dashscope SDK 调 qwen-turbo
print("\n=== 测试 2: dashscope SDK 直接调 qwen-turbo ===")
try:
    resp = dashscope.Generation.call(
        model="qwen-turbo",
        messages=[{"role": "user", "content": "你好"}],
        result_format="message",
    )
    print(f"status_code: {resp.status_code}")
    if resp.status_code == 200:
        print(f"回答: {resp.output.choices[0].message.content[:100]}")
    else:
        print(f"code: {resp.code}, message: {resp.message}")
except Exception as e:
    print(f"异常: {type(e).__name__}: {e}")

# 测试 3: 用 langchain ChatTongyi 调 qwen-max（带工具）
print("\n=== 测试 3: langchain ChatTongyi 调 qwen-max（带工具调用）===")
try:
    from langchain_community.chat_models import ChatTongyi
    from langchain_core.tools import tool

    @tool
    def get_weather(city: str) -> str:
        """获取天气"""
        return f"{city}今天晴天，26度"

    model = ChatTongyi(model="qwen-max", temperature=0)
    model_with_tools = model.bind_tools([get_weather])

    resp = model_with_tools.invoke("杭州天气怎么样？")
    print(f"content: {resp.content[:100] if resp.content else '(空)'}")
    print(f"tool_calls: {resp.tool_calls}")
except Exception as e:
    print(f"异常: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# 测试 4: 连续调用 3 次（检测限流）
print("\n=== 测试 4: 连续调用 3 次 qwen-max（检测限流）===")
for i in range(3):
    try:
        t0 = time.time()
        resp = dashscope.Generation.call(
            model="qwen-max",
            messages=[{"role": "user", "content": f"说第{i+1}句话"}],
            result_format="message",
        )
        dt = time.time() - t0
        if resp.status_code == 200:
            print(f"  [{i+1}] 成功 ({dt:.2f}s): {resp.output.choices[0].message.content[:50]}")
        else:
            print(f"  [{i+1}] 失败 ({dt:.2f}s): code={resp.code}, message={resp.message}")
    except Exception as e:
        print(f"  [{i+1}] 异常: {type(e).__name__}: {e}")
    time.sleep(1)
