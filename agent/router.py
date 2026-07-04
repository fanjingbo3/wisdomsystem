"""前置路由器：L1 规则分类 + L2 qwen-turbo 轻量分类。

三路分类：
- simple：纯闲聊/问候，走 qwen-turbo 直答（~2s）
- simple_tool：单工具意图（天气/位置），走单 agent + function calling（~4s）
- complex：复杂/跨域问题，走 supervisor 多 agent 流程（~12s）

L1 纯关键词规则，<1ms，0 次 LLM 调用，覆盖约 70% 请求。
L2 仅在 L1 不确定时触发，用 qwen-turbo 做意图分类（~1s）。
"""
from typing import Literal
from utils.logger_handler import logger

RouteType = Literal["simple", "simple_tool", "complex"]

# L1 规则：简单闲聊关键词（精确匹配短语或短问候）
SIMPLE_PATTERNS = {
    "你好", "您好", "hi", "hello", "嗨",
    "早上好", "下午好", "晚上好", "早安",
    "谢谢", "感谢", "thanks", "多谢", "辛苦了", "谢了",
    "再见", "拜拜", "bye", "byebye", "回头见",
    "你是谁", "你叫什么", "自我介绍", "介绍你自己",
    "你能做什么", "你有什么功能", "你都会什么", "你能帮我什么",
    "好的", "收到", "明白了", "知道了", "嗯", "哦",
}

# L1 规则：单工具意图关键词（天气/位置类，命中后走单 agent + 工具直答）
SIMPLE_TOOL_KEYWORDS = {
    "天气", "下雨", "降雨", "温度", "湿度", "位置", "在哪", "城市",
}

# 产品关键词（与工具关键词组合出现时判定为跨域 complex）
PRODUCT_KEYWORDS = {
    "机器人", "扫地", "扫拖", "清洁", "滚刷", "滤网", "边刷",
    "水箱", "抹布", "尘盒", "传感器", "导航", "建图", "续航",
    "吸力", "路径", "定时", "预约", "故障", "噪音", "维护", "保养",
    "选购", "推荐", "安装", "充电", "错误", "报警", "卡住", "漏扫",
    "碰撞", "适合用", "能不能用", "可以用吗",
}

# L1 规则：复杂问题关键词（命中任意一个即判为 complex）
COMPLEX_KEYWORDS = {
    # 报告类
    "报告", "使用记录", "使用情况", "月度", "统计", "使用数据",
    # 产品知识类（不含天气/位置，那些已移到 SIMPLE_TOOL_KEYWORDS）
    "故障", "噪音", "滚刷", "滤网", "边刷", "清洁", "维护", "保养",
    "选购", "推荐", "安装", "充电", "错误", "报警", "卡住", "漏扫",
    "碰撞", "水箱", "抹布", "尘盒", "传感器", "导航", "建图", "续航",
    "扫地机", "扫拖", "机器人", "吸力", "路径", "定时", "预约",
}


def _classify_l1(query: str) -> RouteType | None:
    """L1 规则分类。返回 'simple'/'simple_tool'/'complex'，不确定返回 None。"""
    query_stripped = query.strip().lower()

    if not query_stripped:
        return None

    # 精确匹配简单闲聊（去除标点后比较）
    query_clean = query_stripped.rstrip("。！？.!?,，")
    if query_clean in SIMPLE_PATTERNS:
        return "simple"

    # 检测工具关键词和产品关键词
    has_tool_kw = any(kw in query_stripped for kw in SIMPLE_TOOL_KEYWORDS)
    has_product_kw = any(kw in query_stripped for kw in PRODUCT_KEYWORDS)

    # 工具关键词 + 产品关键词 = 跨域问题 → complex（如"天气适合用机器人吗"）
    if has_tool_kw and has_product_kw:
        return "complex"

    # 纯工具问题（只有天气/位置，无产品词）→ simple_tool（如"今天天气怎么样"）
    if has_tool_kw:
        return "simple_tool"

    # 复杂关键词命中 → complex
    for kw in COMPLEX_KEYWORDS:
        if kw in query_stripped:
            return "complex"

    return None


def _classify_l2(query: str) -> RouteType:
    """L2 qwen-turbo 轻量分类。仅在 L1 不确定时触发。"""
    from model.factory import get_light_chat_model

    prompt = (
        "你是一个意图分类器。判断以下用户问题属于哪一类：\n"
        "- 简单：闲聊、问候、感谢、身份询问等可以直接回答的问题\n"
        "- 工具：天气查询、位置查询等只需调用一个简单工具的问题（不涉及产品知识）\n"
        "- 复杂：需要查询产品知识、生成报告、故障排查、跨域问题等\n"
        f"用户问题：{query}\n"
        "回答（仅输出'简单'、'工具'或'复杂'）："
    )

    try:
        model = get_light_chat_model()
        resp = model.invoke(prompt)
        content = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()

        if "简单" in content:
            logger.info(f"[Router] L2 判定: simple (query={query[:30]}, resp={content[:20]})")
            return "simple"
        elif "工具" in content:
            logger.info(f"[Router] L2 判定: simple_tool (query={query[:30]}, resp={content[:20]})")
            return "simple_tool"
        else:
            logger.info(f"[Router] L2 判定: complex (query={query[:30]}, resp={content[:20]})")
            return "complex"
    except Exception as e:
        logger.warning(f"[Router] L2 分类失败，降级为 complex: {e}")
        return "complex"


def classify_query(query: str) -> RouteType:
    """路由入口：L1 规则 → L2 qwen-turbo。返回 'simple' / 'simple_tool' / 'complex'。"""
    result = _classify_l1(query)
    if result is not None:
        logger.info(f"[Router] L1 判定: {result} (query={query[:30]})")
        return result

    return _classify_l2(query)


if __name__ == "__main__":
    test_cases = [
        ("你好", "simple"),
        ("谢谢", "simple"),
        ("你是谁", "simple"),
        ("今天天气怎么样", "simple_tool"),
        ("我在哪个城市", "simple_tool"),
        ("今天温度多少", "simple_tool"),
        ("滚刷多久换一次", "complex"),
        ("给我生成使用报告", "complex"),
        ("今天天气适合用机器人吗", "complex"),
    ]
    for q, expected in test_cases:
        result = classify_query(q)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{q}' -> {result} (期望: {expected})")
