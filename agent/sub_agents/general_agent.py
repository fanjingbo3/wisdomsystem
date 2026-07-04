from langgraph.prebuilt import create_react_agent
from model.factory import light_chat_model
from agent.tools.agent_tools import get_weather, get_user_location
from utils.prompt_loader import load_general_agent_prompts

_react_instance = None
_fallback_instance = None


def get_general_expert(use_fallback: bool = False):
    """获取通用专家 agent。use_fallback=True 时使用兜底 prompt（硬编码流程版）。"""
    global _react_instance, _fallback_instance
    tools = [get_weather, get_user_location]
    if use_fallback:
        if _fallback_instance is None:
            _fallback_instance = create_react_agent(
                model=light_chat_model.materialize(),
                tools=tools,
                name="general_expert",
                prompt=load_general_agent_prompts(fallback=True),
            )
        return _fallback_instance
    if _react_instance is None:
        _react_instance = create_react_agent(
            model=light_chat_model.materialize(),
            tools=tools,
            name="general_expert",
            prompt=load_general_agent_prompts(fallback=False),
        )
    return _react_instance


def general_expert():
    """向后兼容的模块级访问（默认 ReAct 版）。"""
    return get_general_expert(use_fallback=False)
