from langchain_core.messages import SystemMessage, ToolMessage, ToolCall, AIMessage
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)


class ReactAgent:
    def __init__(self):
        self.base_system_prompt = load_system_prompts()
        self._memory_manager = None
        self._agent = None

    def get_memory_manager(self):
        if self._memory_manager is None:
            from memory.memory_manager import MemoryManager
            self._memory_manager = MemoryManager()
        return self._memory_manager

    def _get_agent(self):
        if self._agent is None:
            from langgraph.prebuilt import create_react_agent
            self._agent = create_react_agent(
                model=chat_model.materialize(),
                tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                       get_current_month, fetch_external_data, fill_context_for_report],
            )
        return self._agent

    def execute_stream(self, query: str, user_id: str = "default", session_id: str = "default"):
        context = self.get_memory_manager().build_full_context(user_id, session_id)
        if context:
            sys_content = f"{self.base_system_prompt}\n\n记忆上下文:\n{context}"
        else:
            sys_content = self.base_system_prompt

        input_dict = {
            "messages": [
                SystemMessage(content=sys_content),
                {"role": "user", "content": query},
            ]
        }

        self.get_memory_manager().add_message(session_id, "user", query)

        final_response = ""
        last_content_len = 0
        
        for chunk in self._get_agent().stream(input_dict, stream_mode="values"):
            latest_message = chunk["messages"][-1]

            if isinstance(latest_message, AIMessage):
                if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                    for tc in latest_message.tool_calls:
                        tool_name = tc.get('name', 'unknown')
                        tool_args = tc.get('args', {})
                        args_str = ", ".join([f"{k}={v}" for k, v in tool_args.items()])
                        yield f"\n\U0001f527 [调用工具: {tool_name}({args_str})]\n"
                elif latest_message.content:
                    content = latest_message.content.strip()
                    if content:
                        new_content = content[last_content_len:]
                        if new_content:
                            for char in new_content:
                                yield char
                        last_content_len = len(content)
                        final_response = content

            elif isinstance(latest_message, ToolMessage):
                tool_name = getattr(latest_message, 'name', 'unknown')
                content = getattr(latest_message, 'content', '')
                content_preview = content[:150] + "..." if len(content) > 150 else content
                yield f"\n\u2705 [工具返回: {tool_name}] {content_preview}\n"

        if final_response:
            self.get_memory_manager().add_message(session_id, "assistant", final_response)


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)