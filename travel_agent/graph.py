import os
import json

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from travel_agent.state import TravelContext, TravelRequest, TravelState

from travel_agent.tools import (
    calculate_budget,
    search_attractions,
    search_hotels_mcp,
    search_travel_knowledge,
)

load_dotenv()

TOOLS = [
    search_attractions,
    search_travel_knowledge,
    search_hotels_mcp,
    calculate_budget,
]

SYSTEM_PROMPT = """
你是一名旅行规划 Agent。

你的职责是根据用户需求生成可执行、预算清楚的旅行建议。
目前本地数据只支持成都和北京。

规划旅行时：
1. 必须依次调用 search_attractions、search_hotels_mcp、search_travel_knowledge、calculate_budget。
2. 每次只调用一个工具，取得前一个工具的返回结果后再调用下一个，不得在同一次响应中并行调用多个工具。
3. calculate_budget 必须在景点和酒店工具都已经返回结果后调用。
4. 工具没有返回数据时，明确说明原因，不要编造真实信息。
5. 最终回答使用中文，包含推荐景点、酒店、基础预算和建议。
6. 最终方案只能推荐工具实际返回的景点和酒店。
7. 禁止补充工具结果中不存在的景点、酒店、价格和门票信息。
8. calculate_budget 返回的只是住宿与门票基础预算，不包含餐饮、交通和购物。
9. 不得仅根据基础预算就声称全部旅行费用一定没有超出总预算。
10. 如果数据不足，明确说明“当前本地数据暂不支持”，不要依靠常识补充。
11. 旅行住宿晚数默认等于旅行天数减一。
12. 预算数字必须直接引用 calculate_budget 返回的字段，禁止手工重新计算。
13. 不要提供工具未返回的酒店位置、交通价格、预订渠道或具体美食名称。
14. 最终答案中不得出现“工具结果与手工计算不一致”这样的区间表达；以工具结果为准。
15. 确定候选景点后，调用 search_travel_knowledge 检索景点介绍和来源。
16. 景点介绍必须来自 search_travel_knowledge 返回的 content。
17. 最终方案必须列出知识库返回的 source。
18. 如果知识库没有返回资料，不要自行补充景点介绍。
19. search_attractions 返回候选景点，但不代表这些景点已经获得知识库介绍。
20. 只有 search_travel_knowledge 实际返回的景点，才能写介绍并列出来源。
21. 每个带介绍的景点都必须对应一个 source；没有 source 时只能列出名称、类别和票价。
22. 酒店只能使用 search_hotels_mcp 返回的名称和每晚价格，不得补充位置、交通、评分或设施。
23. 不得声称“全部信息来自知识库”，除非所有事实都能在工具返回值中找到。
24. 酒店必须通过 search_hotels_mcp 查询。
25. 最终酒店信息必须包含 MCP 返回的 source。
26. 如果 MCP 调用失败，明确说明酒店服务暂时不可用，不得自行编造酒店。
27. 最终行程中的具名景点，只能来自 search_attractions 返回结果。
28. search_travel_knowledge 只用于补充已选景点的信息和来源，不能把检索到的其他景点自动加入行程。
29. 不得用“例如”“可以去”等方式补充工具没有返回的景点。
30. 调用 calculate_budget 时，attraction_ticket_per_person 必须等于所有最终选中景点票价之和，不能只传其中一个景点的票价。
31. search_attractions 的城市和偏好由 TravelState 自动注入，调用时不要构造参数。
32. search_hotels_mcp 的城市和每晚预算由 TravelState 自动计算，调用时不要构造参数。
33. 酒店预算策略为：总预算的 50% 分配给住宿，再除以住宿晚数得到每晚价格上限。
34. search_travel_knowledge 的城市和候选景点由 TravelState 以及 search_attractions 的返回结果自动提供，调用时不要构造参数。
35. calculate_budget 的人数、天数、酒店和景点费用由 TravelState 以及已有工具结果自动提供，调用时不要构造参数。
"""

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("缺少 GROQ_API_KEY，请检查项目根目录下的 .env")

model = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    temperature=0,
    timeout=60,
    max_retries=2,
)
request_extractor = model.with_structured_output(TravelRequest)
model_with_tools = model.bind_tools(TOOLS)

def extract_request(
    state: TravelState,
    runtime: Runtime[TravelContext],
) -> dict:
    """提取本轮需求，并合并线程状态和用户长期偏好。"""
    user_message = state["messages"][-1]

    request = request_extractor.invoke(
        [
            SystemMessage(
                content=(
                    "只从用户当前这条消息中提取旅行需求。"
                    "没有明确提供的信息保持为 null，不要自行猜测。"
                    "将多个旅行偏好拆分为字符串列表。"
                    "“不想太赶”或“轻松一点”应提取为轻松节奏。"
                )
            ),
            user_message,
        ]
    )

    new_data = request.model_dump()

    # 读取同一线程中的已有状态。
    for field in ("city", "days", "people", "budget", "pace"):
        if new_data.get(field) is None:
            new_data[field] = state.get(field)

    # 读取当前用户跨线程保存的长期偏好。
    namespace = ("travel_users", runtime.context.user_id)
    memory_item = runtime.store.get(namespace, "profile")

    saved_profile = memory_item.value if memory_item else {}
    saved_preferences = saved_profile.get("preferences", [])
    saved_pace = saved_profile.get("pace")

    current_preferences = new_data.get("preferences") or []
    thread_preferences = state.get("preferences") or []

    new_data["preferences"] = list(
        dict.fromkeys(
            [
                *saved_preferences,
                *thread_preferences,
                *current_preferences,
            ]
        )
    )

    if new_data.get("pace") is None:
        new_data["pace"] = saved_pace

    days = new_data.get("days")
    new_data["nights"] = max(days - 1, 0) if days else None

    # 使用固定 key 更新用户画像，避免重复创建大量记录。
    runtime.store.put(
        namespace,
        "profile",
        {
            "preferences": new_data["preferences"],
            "pace": new_data.get("pace"),
        },
    )

    return new_data

REQUIRED_FIELDS = {
    "city": "目的地城市",
    "days": "旅行天数",
    "people": "旅行人数",
    "budget": "总预算",
}


def validate_request(state: TravelState) -> dict:
    """检查规划旅行所需的核心字段。"""
    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if state.get(field) is None
    ]

    return {"missing_fields": missing_fields}


def route_after_validation(state: TravelState) -> str:
    """根据需求完整程度决定下一步。"""
    if state.get("missing_fields"):
        return "ask_clarification"

    return "agent"


def ask_clarification(state: TravelState) -> dict:
    """生成需要用户补充的信息。"""
    missing_names = [
        REQUIRED_FIELDS[field]
        for field in state["missing_fields"]
    ]

    content = (
        "为了给你生成可靠的旅行方案，请补充以下信息："
        + "、".join(missing_names)
        + "。"
    )

    return {"messages": [AIMessage(content=content)]}

def call_model(state: TravelState) -> dict:
    response = model_with_tools.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"],
        ]
    )
    return {"messages": [response]}

def route_after_agent(state: TravelState) -> str:
    """根据 Agent 最后一次响应决定继续调用工具、结束或补救空响应。"""
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    content = last_message.content
    if isinstance(content, str):
        has_content = bool(content.strip())
    else:
        has_content = bool(content)

    if has_content:
        return "end"

    # 模型没有继续调用工具，但最终正文为空
    return "finalize"


def finalize_answer(state: TravelState) -> dict:
    """使用不绑定工具的模型，根据已有工具结果生成最终旅行方案。"""
    finalizer_prompt = SystemMessage(
        content="""
你是旅行规划 Agent 的最终答案生成器。

此前的 Agent 已经完成工具调用。请根据对话中的 ToolMessage 结果，
直接生成完整的中文旅行方案。

要求：
1. 不再调用任何工具。
2. 只能使用工具已经返回的景点、酒店、价格和知识库信息。
3. 包含行程安排、酒店推荐、基础预算和预算剩余说明。
4. 住宿晚数按照工具返回的 nights 字段。
5. 预算以 calculate_budget 最后一次返回的结果为准。
6. 如果知识库结果包含 source，请列出相关来源。
7. 不要声称餐饮、交通和购物已经包含在基础预算中。
8. 不要输出思考过程，直接输出最终方案。
9. 酒店价格默认为每间每晚；当前默认两人同住一间房。
   住宿费用 = hotel_price_per_night × nights，不乘 people。
10. 预算数字和计算公式必须与 calculate_budget 返回的字段完全一致，
    不得自行重新计算或改变计算口径。
11. 最终行程中的景点名称必须逐一出现在 search_attractions 的返回结果中。
    RAG 返回但 search_attractions 没有选中的景点，不得加入行程。
12. 不得推荐或举例说明任何工具没有返回的景点。
13. 如果最终行程包含多个收费景点，
    calculate_budget 的 attraction_ticket_per_person
    必须是这些景点每人门票的总和。
14. 如果预算工具没有计入某个景点的门票，
    最终行程也不得加入该景点。
"""
    )

    # route_after_agent 只会在最后一条消息内容为空时进入这里，
    # 因此去掉这条空消息，但保留此前所有工具调用及返回结果。
    history = state["messages"][:-1]

    # 这里使用原始 model，而不是 model_with_tools。
    # Finalizer 因此只能写答案，不能再次调用工具。
    response = model.invoke(
        [
            finalizer_prompt,
            *history,
        ]
    )

    return {"messages": [response]}

def build_travel_graph(checkpointer=None, store=None):
    builder = StateGraph(
        TravelState,
        context_schema=TravelContext,
    )

    builder.add_node("extract_request", extract_request)
    builder.add_node("validate_request", validate_request)
    builder.add_node("ask_clarification", ask_clarification)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("finalize", finalize_answer)

    builder.add_edge(START, "extract_request")

    builder.add_edge("extract_request", "validate_request")

    builder.add_conditional_edges(
        "validate_request",
        route_after_validation,
        {
            "ask_clarification": "ask_clarification",
            "agent": "agent",
        },
    )

    builder.add_edge("ask_clarification", END)

    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "finalize": "finalize",
            "end": END,
        },
    )

    builder.add_edge("tools", "agent")
    builder.add_edge("finalize", END)

    return builder.compile(
        checkpointer=checkpointer,
        store=store,
    )