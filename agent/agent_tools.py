from langchain_core.tools import tool
from utils.logger_handler import logger

_fallback_mode = False


def set_fallback_mode(enabled: bool):
    """切换全局 fallback 模式。ReAct 失败时设为 True，使用硬编码流程版 prompt。"""
    global _fallback_mode
    _fallback_mode = enabled
    logger.info(f"[Fallback] 模式切换: {'兜底(硬编码流程)' if enabled else 'ReAct'}")


@tool
def call_knowledge_expert(query: str) -> str:
    """派发给知识问答专家。用于：扫地机器人故障排查、维护保养、选购指南、产品知识、
    使用技巧类问题。专家会调用 RAG 检索知识库后回答。"""
    from agent.sub_agents import get_knowledge_expert
    try:
        expert = get_knowledge_expert(use_fallback=_fallback_mode)
        result = expert.invoke({"messages": [{"role": "user", "content": query}]})
        return result["messages"][-1].content
    except Exception as e:
        logger.error(f"[call_knowledge_expert] 知识专家调用失败: {str(e)}", exc_info=True)
        return f"知识专家调用失败: {str(e)}"


@tool
def call_report_expert(query: str) -> str:
    """派发给报告生成专家。用于：用户使用报告生成、个性化使用建议、月度使用数据分析类问题。"""
    from agent.sub_agents import get_report_expert
    try:
        expert = get_report_expert(use_fallback=_fallback_mode)
        result = expert.invoke({"messages": [{"role": "user", "content": query}]})
        return result["messages"][-1].content
    except Exception as e:
        logger.error(f"[call_report_expert] 报告专家调用失败: {str(e)}", exc_info=True)
        return f"报告专家调用失败: {str(e)}"


@tool
def call_general_expert(query: str) -> str:
    """派发给通用闲聊专家。用于：天气查询、用户定位、闲聊、与产品无关的通用问题。"""
    from agent.sub_agents import get_general_expert
    try:
        expert = get_general_expert(use_fallback=_fallback_mode)
        result = expert.invoke({"messages": [{"role": "user", "content": query}]})
        return result["messages"][-1].content
    except Exception as e:
        logger.error(f"[call_general_expert] 通用专家调用失败: {str(e)}", exc_info=True)
        return f"通用专家调用失败: {str(e)}"
