from langgraph.prebuilt import create_react_agent
from model.factory import chat_model
from utils.prompt_loader import load_supervisor_prompts
from agent.agent_tools import call_knowledge_expert, call_report_expert, call_general_expert


def build_supervisor(use_fallback: bool = False):
    """构建 Supervisor Agent（普通 ReAct Agent + 3 个 agent-tool）。

    Supervisor 负责意图分析，通过调用 call_knowledge_expert / call_report_expert /
    call_general_expert 三个 agent-tool 把任务派发给对应子 Agent。
    跨域查询通过串行调用多个 agent-tool 实现，中间结果在 supervisor 消息历史中累积。

    use_fallback=True 时使用兜底 prompt（硬编码派发规则版），ReAct 失败时降级使用。
    """
    return create_react_agent(
        model=chat_model.materialize(),
        tools=[call_knowledge_expert, call_report_expert, call_general_expert],
        name="supervisor",
        prompt=load_supervisor_prompts(fallback=use_fallback),
    )
