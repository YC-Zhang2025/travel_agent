import os
import json

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from travel_agent.grounding import (
    find_unsupported_answer_phrases,
)
from travel_agent.state import (
    AnswerReview,
    TravelContext,
    TravelRequest,
    TravelState,
)

from travel_agent.tools import (
    calculate_budget,
    get_latest_tool_result,
    search_attractions,
    search_hotels_mcp,
    search_travel_knowledge,
)

load_dotenv()

TOOLS = [
    search_attractions,
    search_hotels_mcp,
    search_travel_knowledge,
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

answer_reviewer = model.with_structured_output(AnswerReview)

MAX_ANSWER_REVISIONS = 1

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
    namespace = ("travel_users", runtime.context["user_id"],)
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
    new_data["answer_is_grounded"] = None
    new_data["validation_feedback"] = None
    new_data["revision_count"] = 0

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

def build_answer_evidence(state: TravelState) -> dict:
    """从 State 和 ToolMessage 中整理最终答案唯一可以使用的证据。"""
    messages = state["messages"]

    attractions = get_latest_tool_result(
        messages,
        "search_attractions",
    ) or []

    hotel_result = get_latest_tool_result(
        messages,
        "search_hotels_mcp",
    ) or {}

    knowledge = get_latest_tool_result(
        messages,
        "search_travel_knowledge",
    ) or []

    budget_result = get_latest_tool_result(
        messages,
        "calculate_budget",
    ) or {}

    remaining_budget = None

    if (
        state.get("budget") is not None
        and budget_result.get("total_cost") is not None
    ):
        remaining_budget = (
            float(state["budget"])
            - float(budget_result["total_cost"])
        )

    return {
        "request": {
            "city": state.get("city"),
            "days": state.get("days"),
            "nights": state.get("nights"),
            "people": state.get("people"),
            "total_budget": state.get("budget"),
            "preferences": state.get("preferences") or [],
            "pace": state.get("pace"),
        },
        "selected_attractions": attractions,
        "hotel_search": hotel_result,
        "knowledge": knowledge,
        "budget_calculation": budget_result,
        "remaining_budget": remaining_budget,
    }

def route_after_agent(state: TravelState) -> str:
    """工具调用继续执行；普通文本响应统一交给最终写作节点。"""
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return "finalize"


def finalize_answer(state: TravelState) -> dict:
    """只依据结构化证据生成最终旅行方案。"""
    evidence = build_answer_evidence(state)

    finalizer_prompt = SystemMessage(
        content="""
你是旅行规划 Agent 的最终答案生成器。

你只能使用用户消息中提供的需求，以及“可靠证据”JSON 中的事实。
不得使用常识补充任何地点、活动、位置关系、价格、开放时间、
交通方式、预订状态、预订渠道、设施或具体食物。

要求：

1. 只能推荐 selected_attractions 中的景点。
2. 景点介绍只能引用 knowledge 中对应景点的 content。
3. 酒店只能选择 budget_calculation.selected_hotel。
4. 酒店名称和价格必须来自 hotel_search。
5. 酒店来源必须使用 hotel_search.source，并且只能称为信息来源或数据来源，
   不能称为预订来源、预订渠道或预订平台。
6. 预算数字必须使用 budget_calculation 中的字段。
7. remaining_budget 只是扣除住宿与门票后的余额。
8. 不得声称剩余预算足以覆盖餐饮、交通、购物或整趟旅行。
9. 不得声称酒店已经预订。
10. 允许使用的普通行程动作只有：
    抵达、入住、休息、游览已选景点、自由时间、返程。
11. 不得出现证据中没有的具名地点或其他具体活动。
12. 数据不足时直接说明“当前数据不足”，不得自行补充。
13. 最终回答使用中文，包含行程、酒店、基础预算和信息来源。
14. 不要声称“没有进行任何推测”，只需直接给出方案。
"""
    )

    evidence_message = HumanMessage(
        content=(
            "下面是生成答案时唯一允许使用的可靠证据：\n"
            + json.dumps(
                evidence,
                ensure_ascii=False,
                indent=2,
            )
        )
    )

    response = model.invoke(
        [
            finalizer_prompt,
            evidence_message,
        ]
    )

    return {"messages": [response]}

def validate_answer(state: TravelState) -> dict:
    """检查最终答案中的事实是否都能由可靠证据支持。"""
    evidence = build_answer_evidence(state)
    final_answer = state["messages"][-1].content

    validation_prompt = SystemMessage(
        content="""
你是严格的旅行方案事实校验器。

请对比“可靠证据”和“待校验答案”，判断答案中的所有事实性内容
是否都能由可靠证据直接支持。

以下情况必须判定为不合格：

1. 出现证据中没有的景点、酒店、地点或具体活动。
2. 添加证据中没有的位置关系、交通、设施、开放时间或价格。
3. 把酒店数据来源描述成预订来源、预订渠道或预订平台，
   或者声称酒店已经预订。
4. 声称剩余预算足以覆盖整趟旅行、餐饮、交通或购物。
5. 声称“满足所有需求”“没有推测”“保证准确”等无法验证的结论。
6. 景点介绍超出 knowledge.content 的信息。
7. 预算数字与 budget_calculation 不一致。

以下内容可以接受：

1. Markdown 表格、标题和排版。
2. 直接来自证据的数字和信息。
3. 根据 total_budget 和 total_cost 得到的 remaining_budget。
4. 抵达、入住、休息、游览已选景点、自由时间和返程等中性安排。

如果不合格，feedback 必须明确指出应删除或修改的内容。
如果合格，feedback 返回空字符串。
"""
    )

    review_message = HumanMessage(
        content=(
            "可靠证据：\n"
            + json.dumps(
                evidence,
                ensure_ascii=False,
                indent=2,
            )
            + "\n\n待校验答案：\n"
            + str(final_answer)
        )
    )

    review = answer_reviewer.invoke(
        [
            validation_prompt,
            review_message,
        ]
    )

    deterministic_violations = (
        find_unsupported_answer_phrases(str(final_answer))
    )
    feedback_parts = []

    if review.feedback:
        feedback_parts.append(review.feedback)

    if deterministic_violations:
        feedback_parts.append(
            "删除或改写以下无依据表达："
            + "、".join(deterministic_violations)
            + "。MCP source 只能称为信息来源或数据来源。"
        )

    return {
        "answer_is_grounded": (
            review.is_grounded
            and not deterministic_violations
        ),
        "validation_feedback": "\n".join(feedback_parts),
    }

def revise_answer(state: TravelState) -> dict:
    """根据校验反馈重新生成一次最终答案。"""
    evidence = build_answer_evidence(state)
    previous_answer = state["messages"][-1].content
    feedback = state.get("validation_feedback") or ""

    revision_prompt = SystemMessage(
        content="""
你是旅行方案修订器。

请根据校验反馈修订答案。修订后的所有事实必须来自可靠证据。

要求：

1. 删除所有证据无法支持的内容。
2. 不得添加新的地点、活动或事实。
3. “source”只能描述为信息来源或数据来源，
   不能描述为预订来源、预订渠道或预订平台。
4. 不得声称酒店已经预订。
5. 不得声称剩余预算足以覆盖整趟旅行。
6. 不得声称满足所有需求或没有进行推测。
7. 保留正确的行程、酒店、预算和来源信息。
8. 只输出修订后的完整旅行方案，不要解释修改过程。
"""
    )

    revision_message = HumanMessage(
        content=(
            "可靠证据：\n"
            + json.dumps(
                evidence,
                ensure_ascii=False,
                indent=2,
            )
            + "\n\n校验反馈：\n"
            + feedback
            + "\n\n原答案：\n"
            + str(previous_answer)
        )
    )

    response = model.invoke(
        [
            revision_prompt,
            revision_message,
        ]
    )

    return {
        "messages": [response],
        "revision_count": (
            int(state.get("revision_count") or 0) + 1
        ),
    }

def route_after_answer_validation(
    state: TravelState,
) -> str:
    """校验通过则结束，否则最多修订一次。"""
    if state.get("answer_is_grounded"):
        return "end"

    revision_count = int(
        state.get("revision_count") or 0
    )

    if revision_count >= MAX_ANSWER_REVISIONS:
        return "end"

    return "revise"

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
    builder.add_node("validate_answer", validate_answer)
    builder.add_node("revise_answer", revise_answer)

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
        },
    )

    builder.add_edge("tools", "agent")
    builder.add_edge("finalize", "validate_answer")

    builder.add_conditional_edges(
        "validate_answer",
        route_after_answer_validation,
        {
            "end": END,
            "revise": "revise_answer",
        },
    )

    builder.add_edge(
        "revise_answer",
        "validate_answer",
    )

    return builder.compile(
        checkpointer=checkpointer,
        store=store,
    )
