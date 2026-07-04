"""调试脚本：打印 _stream_complex 产生的所有 chunks，定位思考过程不完整的问题。"""
import time
import sys

# 强制使用当前目录的代码
sys.path.insert(0, ".")

from agent.react_agent import ReactAgent

print("=" * 60)
print("等待预热完成...")
print("=" * 60)

agent = ReactAgent()

# 等预热完成
import threading
time.sleep(15)  # 给预热 15 秒

queries = [
    "扫地机器人滚刷多久换一次",
    "我这儿的天气适合清洗机器人吗",
]

for q in queries:
    print("\n" + "=" * 60)
    print(f"问题: {q}")
    print("=" * 60)

    t0 = time.time()
    t_first = None
    chunk_count = 0
    marker_count = {"dispatch": 0, "tool_call": 0, "tool_return": 0, "expert_return": 0}
    full_text = ""

    for chunk in agent.execute_stream(q, user_id="debug", session_id="debug"):
        chunk_count += 1
        if t_first is None:
            t_first = time.time() - t0

        # 标记类型统计
        if chunk.startswith("\n\U0001f91d"):
            marker_count["dispatch"] += 1
            tag = "DISPATCH"
        elif chunk.startswith("\n\U0001f527"):
            marker_count["tool_call"] += 1
            tag = "TOOL_CALL"
        elif chunk.startswith("\n\u2705 [工具返回"):
            marker_count["tool_return"] += 1
            tag = "TOOL_RETURN"
        elif chunk.startswith("\n\u2705 [") and "专家返回" in chunk:
            marker_count["expert_return"] += 1
            tag = "EXPERT_RETURN"
        else:
            tag = "CONTENT"

        # 标记 chunks 完整打印（替换换行便于阅读）
        display = chunk.replace("\n", "\\n")
        if len(display) > 120:
            display = display[:120] + "..."
        print(f"  [{chunk_count:3d}] ({tag:13s}) {display}")

        if tag == "CONTENT":
            full_text += chunk

    t_total = time.time() - t0
    print(f"\n--- 统计 ---")
    print(f"  首字/首标记时间: {t_first:.2f}s" if t_first else "  无输出")
    print(f"  总耗时: {t_total:.2f}s")
    print(f"  chunk 总数: {chunk_count}")
    print(f"  标记: 派发={marker_count['dispatch']}, 调用工具={marker_count['tool_call']}, "
          f"工具返回={marker_count['tool_return']}, 专家返回={marker_count['expert_return']}")
    print(f"  正文长度: {len(full_text)} 字符")
    print(f"  正文内容: {full_text[:300]}{'...' if len(full_text) > 300 else ''}")
