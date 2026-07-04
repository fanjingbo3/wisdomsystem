import os
import threading
import concurrent.futures
from utils.logger_handler import logger
from langchain_core.tools import tool

import random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path

_rag_service = None
_rag_ready = threading.Event()
_rag_initializing = False
_rag_lock = threading.Lock()

def _prewarm_rag_service():
    global _rag_service, _rag_initializing
    with _rag_lock:
        if _rag_initializing or _rag_service is not None:
            return
        _rag_initializing = True
    
    try:
        from rag.rag_service import RagSummarizeService
        from model.factory import get_embed_model, get_chat_model
        service = RagSummarizeService()
        
        # 第一阶段：并行初始化核心模型（单例，避免多线程竞争）
        logger.info("[RAG预热] 初始化核心模型...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(get_embed_model),
                executor.submit(get_chat_model),
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        
        # 第二阶段：并行初始化各个组件（核心模型已就绪）
        logger.info("[RAG预热] 初始化RAG组件...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(lambda: service._get_vector_retriever()),
                executor.submit(lambda: service._get_bm25_retriever()._ensure_loaded()),
                executor.submit(lambda: service._get_query_rewriter()),
                executor.submit(lambda: service._get_reranker()),
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        
        _rag_service = service
        logger.info("[RAG预热] RAG服务预热完成")
    except Exception as e:
        logger.error(f"[RAG预热] RAG服务预热失败: {str(e)}", exc_info=True)
    finally:
        _rag_initializing = False
        _rag_ready.set()

def _get_rag_service():
    global _rag_service
    if _rag_service is None:
        if not _rag_initializing:
            threading.Thread(target=_prewarm_rag_service, daemon=True).start()
        _rag_ready.wait(timeout=30)
    return _rag_service

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010",]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]

external_data = {}


@tool
def rag_summarize(query: str) -> str:
    """从向量存储中检索参考资料"""
    rag_service = _get_rag_service()
    if rag_service is None:
        logger.error("[rag_summarize] RAG服务未初始化，请检查API Key配置")
        return "RAG服务未初始化，请检查API Key配置或网络连接"
    return rag_service.rag_summarize(query)


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气，以消息字符串的形式返回"""
    return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"


@tool
def get_user_location() -> str:
    """获取用户所在城市的名称，以纯字符串形式返回"""
    return random.choice(["深圳", "合肥", "杭州"])


@tool
def get_user_id() -> str:
    """获取用户的ID，以纯字符串形式返回"""
    return random.choice(user_ids)


@tool
def get_current_month() -> str:
    """获取当前月份，以纯字符串形式返回"""
    return random.choice(month_arr)


def generate_external_data():
    """
    {
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        ...
    }
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].replace('"', "")
                feature: str = arr[1].replace('"', "")
                efficiency: str = arr[2].replace('"', "")
                consumables: str = arr[3].replace('"', "")
                comparison: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool
def fetch_external_data(user_id: str, month: str) -> str:
    """从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回，如果未检索到返回空字符串"""
    generate_external_data()

    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""


@tool
def fill_context_for_report():
    """无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息"""
    return "fill_context_for_report已调用"