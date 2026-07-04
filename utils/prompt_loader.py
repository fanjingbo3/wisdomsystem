from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

def load_system_prompts():
    try:
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompts]在yaml配置项中没有main_prompt_path配置项")
        raise e

    try:
        return open(system_prompt_path,"r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompts]解析系统提示词出错，{str(e)}")
        raise e


def load_rag_prompts():
    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompts]在yaml配置项中没有rag_summarize_prompt_path配置项")
        raise e

    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompts]解析RAG提示词出错，{str(e)}")
        raise e


def load_report_prompts():
    try:
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompts]在yaml配置项中没有report_prompt_path配置项")
        raise e

    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompts]解析报告生成提示词出错，{str(e)}")
        raise e


def load_supervisor_prompts(fallback: bool = False):
    key = "supervisor_fallback_path" if fallback else "supervisor_prompt_path"
    try:
        prompt_path = get_abs_path(prompts_conf[key])
    except KeyError as e:
        logger.error(f"[load_supervisor_prompts]在yaml配置项中没有{key}配置项")
        raise e

    try:
        return open(prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_supervisor_prompts]解析Supervisor提示词出错，{str(e)}")
        raise e


def load_knowledge_agent_prompts(fallback: bool = False):
    key = "knowledge_agent_fallback_path" if fallback else "knowledge_agent_prompt_path"
    try:
        prompt_path = get_abs_path(prompts_conf[key])
    except KeyError as e:
        logger.error(f"[load_knowledge_agent_prompts]在yaml配置项中没有{key}配置项")
        raise e

    try:
        return open(prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_knowledge_agent_prompts]解析知识专家提示词出错，{str(e)}")
        raise e


def load_report_agent_prompts(fallback: bool = False):
    key = "report_agent_fallback_path" if fallback else "report_agent_prompt_path"
    try:
        prompt_path = get_abs_path(prompts_conf[key])
    except KeyError as e:
        logger.error(f"[load_report_agent_prompts]在yaml配置项中没有{key}配置项")
        raise e

    try:
        return open(prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_agent_prompts]解析报告专家提示词出错，{str(e)}")
        raise e


def load_general_agent_prompts(fallback: bool = False):
    key = "general_agent_fallback_path" if fallback else "general_agent_prompt_path"
    try:
        prompt_path = get_abs_path(prompts_conf[key])
    except KeyError as e:
        logger.error(f"[load_general_agent_prompts]在yaml配置项中没有{key}配置项")
        raise e

    try:
        return open(prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_general_agent_prompts]解析通用专家提示词出错，{str(e)}")
        raise e

if __name__=='__main__':
    print(load_report_prompts())

    