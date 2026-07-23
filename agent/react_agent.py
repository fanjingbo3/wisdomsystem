import threading
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from utils.prompt_loader import load_supervisor_prompts
from utils.logger_handler import logger


_AGENT_TOOL_NAMES = {
    "call_knowledge_expert": "知识专家",
    "call_report_expert": "报告专家",
    "call_general_expert": "通用专家",
}

_SIMPLE_PROMPT = (
    "你是「智扫通」扫地机器人智能客服。用简洁友好的中文回应用户的闲聊、问候、身份询问等。"
    "不要编造产品参数或使用建议——涉及产品知识的问题请建议用户详细描述问题。"
)

_SIMPLE_TOOL_PROMPT = (
    "你是「智扫通」扫地机器人智能客服的天气/位置助手。"
    "用户询问天气时，必须先调用 get_user_location 获取用户所在城市，"
    "再调用 get_weather 查询该城市天气，然后根据工具返回的结果回答。"
    "不要编造天气数据，不要询问用户城市，直接调用工具获取。"
    "用户询问位置时，直接调用 get_user_location 获取。回答简洁友好。"
)


class ReactAgent:
    def __init__(self):
        self.base_system_prompt = load_supervisor_prompts()
        self._memory_manager = None
        self._agent = None
        self._agent_fallback = None
        self._simple_tool_agent = None
        self._prewarmed = False
        self._prewarm_lock = threading.Lock()
        self._start_prewarm()

    def get_memory_manager(self):
        if self._memory_manager is None:
            from memory.memory_manager import MemoryManager
            self._memory_manager = MemoryManager()
        return self._memory_manager

    def _get_agent(self, use_fallback: bool = False):
        if use_fallback:
            if self._agent_fallback is None:
                from agent.supervisor import build_supervisor
                self._agent_fallback = build_supervisor(use_fallback=True)
            return self._agent_fallback
        if self._agent is None:
            from agent.supervisor import build_supervisor
            self._agent = build_supervisor(use_fallback=False)
        return self._agent

    def _get_simple_tool_agent(self):
        """懒加载：单 agent + 天气/位置工具（qwen-turbo），不走 supervisor。"""
        if self._simple_tool_agent is None:
            from langgraph.prebuilt import create_react_agent
            from model.factory import get_light_chat_model
            from agent.tools.agent_tools import get_weather, get_user_location
            self._simple_tool_agent = create_react_agent(
                model=get_light_chat_model(),
                tools=[get_weather, get_user_location],
                name="simple_tool_agent",
                prompt=_SIMPLE_TOOL_PROMPT,
            )
        return self._simple_tool_agent

    def _start_prewarm(self):
        """启动后台预热线程：初始化 ChatTongyi + qwen-turbo + RAG + supervisor。"""
        def _prewarm():
            try:
                from utils.config_handler import rag_conf
                main_model_name = rag_conf.get("chat_model_name", "未配置")
                light_model_name = rag_conf.get("light_chat_model_name", "未配置")
                logger.info("[Prewarm] 开始后台预热...")
                from model.factory import chat_model, light_chat_model
                chat_model.materialize()
                logger.info(f"[Prewarm] 主模型({main_model_name}) 已就绪")
                light_chat_model.materialize()
                logger.info(f"[Prewarm] 轻量模型({light_model_name}) 已就绪")
                from agent.tools.agent_tools import _prewarm_rag_service
                _prewarm_rag_service()
                logger.info("[Prewarm] RAG 服务已就绪")
                self._get_agent()
                logger.info("[Prewarm] supervisor(ReAct) 已就绪")
                self._get_simple_tool_agent()
                logger.info("[Prewarm] simple_tool_agent 已就绪")
                with self._prewarm_lock:
                    self._prewarmed = True
                logger.info("[Prewarm] 预热全部完成")
            except Exception as e:
                logger.warning(f"[Prewarm] 预热失败（不影响功能，首次提问时仍会懒加载）: {e}")

        threading.Thread(target=_prewarm, daemon=True).start()

    def _stream_simple(self, query: str, user_id: str, session_id: str):
        """简单问题：用 qwen-turbo 直接流式回答。"""
        from model.factory import get_light_chat_model

        self.get_memory_manager().add_message(session_id, "user", query)

        model = get_light_chat_model()
        messages = [
            SystemMessage(content=_SIMPLE_PROMPT),
            {"role": "user", "content": query},
        ]

        final_response = ""
        try:
            for chunk in model.stream(messages):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    final_response += content
                    for char in content:
                        yield char
        except Exception as e:
            logger.error(f"[simple] qwen-turbo 回答失败: {e}", exc_info=True)
            yield f"抱歉，回答时出现错误：{e}"
            return

        if final_response:
            self.get_memory_manager().add_message(session_id, "assistant", final_response)

    def _stream_simple_tool(self, query: str, user_id: str, session_id: str):
        """工具类简单问题：单 agent + function calling 直答（不走 supervisor）。"""
        self.get_memory_manager().add_message(session_id, "user", query)

        input_dict = {
            "messages": [
                SystemMessage(content=_SIMPLE_TOOL_PROMPT),
                {"role": "user", "content": query},
            ]
        }

        final_response = ""
        pending_ai_content = None

        for chunk in self._get_simple_tool_agent().stream(input_dict, stream_mode="values"):
            latest_message = chunk["messages"][-1]

            if isinstance(latest_message, AIMessage):
                if pending_ai_content is not None:
                    for char in pending_ai_content:
                        yield char
                    final_response = pending_ai_content
                    pending_ai_content = None

                if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                    for tc in latest_message.tool_calls:
                        tool_name = tc.get('name', 'unknown')
                        tool_args = tc.get('args', {})
                        args_str = ", ".join([f"{k}={v}" for k, v in tool_args.items()])
                        yield f"\n\U0001f527 [调用工具: {tool_name}({args_str})]\n"
                elif latest_message.content:
                    content = latest_message.content.strip()
                    if content:
                        pending_ai_content = content

            elif isinstance(latest_message, ToolMessage):
                pending_ai_content = None
                tool_name = getattr(latest_message, 'name', 'unknown')
                content = getattr(latest_message, 'content', '')
                content_preview = content[:150] + "..." if len(content) > 150 else content
                yield f"\n\u2705 [工具返回: {tool_name}] {content_preview}\n"

        if pending_ai_content is not None:
            for char in pending_ai_content:
                yield char
            final_response = pending_ai_content

        if final_response:
            self.get_memory_manager().add_message(session_id, "assistant", final_response)

    def _stream_complex_impl(self, query: str, user_id: str, session_id: str, use_fallback: bool):
        """supervisor 多 agent 流程的实际实现（messages 模式，逐 token 流式）。"""
        logger.info(f"[complex] 开始处理 (use_fallback={use_fallback}, query={query[:40]})")
        if use_fallback:
            system_prompt = load_supervisor_prompts(fallback=True)
        else:
            system_prompt = self.base_system_prompt

        context = self.get_memory_manager().build_full_context(user_id, session_id)
        memory_context = context if context else "无"
        sys_content = system_prompt.replace("{memory_context}", memory_context)

        input_dict = {
            "messages": [
                SystemMessage(content=sys_content),
                {"role": "user", "content": query},
            ]
        }

        if not use_fallback:
            self.get_memory_manager().add_message(session_id, "user", query)

        final_response = ""
        seen_tool_call_ids = set()

        logger.info("[complex] 启动 supervisor stream...")
        for chunk_data in self._get_agent(use_fallback=use_fallback).stream(
            input_dict,
            stream_mode="messages",
            subgraphs=True,
            recursion_limit=10,
            configurable={"thread_id": session_id},
        ):
            if not isinstance(chunk_data, tuple) or len(chunk_data) != 2:
                continue
            # stream(subgraphs=True) 格式: (namespace_tuple, (message_chunk, metadata_dict))
            namespace, (chunk, metadata) = chunk_data
            if not isinstance(metadata, dict):
                metadata = {}
            is_top_level = len(namespace) == 0

            chunk_type = type(chunk).__name__

            if chunk_type in ("AIMessageChunk", "AIMessage"):
                tool_calls = getattr(chunk, 'tool_calls', None) or []
                for tc in tool_calls:
                    tc_id = tc.get("id", "")
                    if tc_id and tc_id in seen_tool_call_ids:
                        continue
                    if tc_id:
                        seen_tool_call_ids.add(tc_id)

                    tool_name = tc.get("name", "unknown")
                    tool_args = tc.get("args", {})
                    args_str = ", ".join([f"{k}={v}" for k, v in tool_args.items()])
                    if tool_name in _AGENT_TOOL_NAMES:
                        yield f"\n\U0001f91d [派发给: {_AGENT_TOOL_NAMES[tool_name]}]\n"
                    else:
                        yield f"\n\U0001f527 [调用工具: {tool_name}({args_str})]\n"

                content = getattr(chunk, 'content', '')
                if content and is_top_level:
                    final_response += content
                    for char in content:
                        yield char

            elif chunk_type == "ToolMessage":
                tool_name = getattr(chunk, 'name', 'unknown')
                content = getattr(chunk, 'content', '')
                content_preview = content[:150] + "..." if len(content) > 150 else content
                if tool_name in _AGENT_TOOL_NAMES:
                    yield f"\n\u2705 [{_AGENT_TOOL_NAMES[tool_name]}返回] {content_preview}\n"
                else:
                    yield f"\n\u2705 [工具返回: {tool_name}] {content_preview}\n"

        logger.info(f"[complex] stream 结束 (final_response长度={len(final_response)})")
        if final_response:
            self.get_memory_manager().add_message(session_id, "assistant", final_response)

    def _stream_complex(self, query: str, user_id: str, session_id: str):
        """复杂问题：先走 ReAct 版 supervisor，失败时降级到 fallback 版重试。"""
        try:
            yield from self._stream_complex_impl(query, user_id, session_id, use_fallback=False)
        except Exception as e:
            logger.warning(f"[complex] ReAct 模式失败，降级到 fallback: {e}", exc_info=True)
            from agent.agent_tools import set_fallback_mode
            set_fallback_mode(True)
            try:
                yield "\n\n[正在切换到稳定模式重新生成...]\n\n"
                yield from self._stream_complex_impl(query, user_id, session_id, use_fallback=True)
            finally:
                set_fallback_mode(False)

    def execute_stream(self, query: str, user_id: str = "default", session_id: str = "default"):
        """入口：三路路由——simple 走 qwen-turbo 直答，simple_tool 走单 agent+工具，complex 走 supervisor。"""
        from agent.router import classify_query

        route = classify_query(query)

        if route == "simple":
            yield from self._stream_simple(query, user_id, session_id)
        elif route == "simple_tool":
            yield from self._stream_simple_tool(query, user_id, session_id)
        else:
            yield from self._stream_complex(query, user_id, session_id)


if __name__ == '__main__':
    import time
    agent = ReactAgent()

    print("=== 简单问题测试 ===")
    t = time.time()
    for chunk in agent.execute_stream("你好"):
        print(chunk, end="", flush=True)
    print(f"\n[耗时: {time.time()-t:.2f}s]")

    print("\n=== 复杂问题测试 ===")
    t = time.time()
    for chunk in agent.execute_stream("扫地机器人滚刷多久换一次"):
        print(chunk, end="", flush=True)
    print(f"\n[耗时: {time.time()-t:.2f}s]")
