from langgraph.prebuilt import create_react_agent
from model.factory import chat_model
from agent.tools.agent_tools import (
    fetch_external_data,
    fill_context_for_report,
    get_user_id,
    get_current_month,
)
from utils.prompt_loader import load_report_agent_prompts

_react_instance = None
_fallback_instance = None


def get_report_expert(use_fallback: bool = False):
    """获取报告专家 agent。use_fallback=True 时使用兜底 prompt（硬编码流程版）。"""
    global _react_instance, _fallback_instance
    tools = [fetch_external_data, fill_context_for_report, get_user_id, get_current_month]
    if use_fallback:
        if _fallback_instance is None:
            _fallback_instance = create_react_agent(
                model=chat_model.materialize(),
                tools=tools,
                name="report_expert",
                prompt=load_report_agent_prompts(fallback=True),
            )
        return _fallback_instance
    if _react_instance is None:
        _react_instance = create_react_agent(
            model=chat_model.materialize(),
            tools=tools,
            name="report_expert",
            prompt=load_report_agent_prompts(fallback=False),
        )
    return _react_instance


def report_expert():
    """向后兼容的模块级访问（默认 ReAct 版）。"""
    return get_report_expert(use_fallback=False)
