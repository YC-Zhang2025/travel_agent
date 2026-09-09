import json
import re

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from travel_agent.mcp_client import search_hotels_via_mcp
from travel_agent.rag import retrieve_travel_knowledge

HOTEL_BUDGET_RATIO = 0.5
ATTRACTIONS = {
    "成都": [
        {"name": "宽窄巷子", "tag": "文化", "ticket": 0},
        {"name": "大熊猫繁育研究基地", "tag": "自然", "ticket": 55},
        {"name": "锦里", "tag": "美食", "ticket": 0},
        {"name": "武侯祠", "tag": "历史", "ticket": 50},
    ],
    "北京": [
        {"name": "故宫", "tag": "历史", "ticket": 60},
        {"name": "颐和园", "tag": "自然", "ticket": 30},
        {"name": "南锣鼓巷", "tag": "美食", "ticket": 0},
    ],
}

HOTELS = {
    "成都": [
        {"name": "春熙路舒适酒店", "price_per_night": 320},
        {"name": "宽窄巷子精品酒店", "price_per_night": 480},
    ],
    "北京": [
        {"name": "前门快捷酒店", "price_per_night": 420},
        {"name": "国贸商务酒店", "price_per_night": 680},
    ],
}


def filter_attractions(
    city: str,
    preferences: list[str] | None,
) -> list[dict]:
    """根据已经结构化的城市和偏好过滤本地景点数据。"""
    attractions = ATTRACTIONS.get(city, [])
    preferences = preferences or []

    keywords = []

    for preference in preferences:
        keywords.extend(
            keyword
            for keyword in re.split(
                r"[\s、,，/和]+",
                preference.strip(),
            )
            if keyword
        )

    if not keywords:
        return attractions

    return [
        item
        for item in attractions
        if any(
            keyword in item["tag"]
            or keyword in item["name"]
            for keyword in keywords
        )
    ]

def get_latest_tool_result(
    messages: list,
    tool_name: str,
):
    """从消息历史中获取指定工具最后一次返回的结构化结果。"""
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue

        if message.name != tool_name:
            continue

        content = message.content

        if isinstance(content, (dict, list)):
            return content

        if not isinstance(content, str):
            return None

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None

    return None


@tool
def search_attractions(runtime: ToolRuntime) -> list[dict]:
    """根据当前结构化旅行需求查询景点，参数由 TravelState 自动提供。"""
    state = runtime.state

    return filter_attractions(
        city=state.get("city") or "",
        preferences=state.get("preferences") or [],
    )


@tool
def search_hotels_mcp(runtime: ToolRuntime) -> dict:
    """根据 TravelState 中的预算，通过 MCP Server 查询酒店。"""
    state = runtime.state

    city = state.get("city") or ""
    budget = state.get("budget")
    nights = state.get("nights") or 0

    if budget is None or nights <= 0:
        max_price_per_night = 0.0
    else:
        max_price_per_night = round(
            float(budget) * HOTEL_BUDGET_RATIO / nights,
            2,
        )

    return search_hotels_via_mcp(
        city=city,
        max_price_per_night=max_price_per_night,
    )

def compute_budget(
    days: int,
    people: int,
    hotel_price_per_night: float,
    attraction_ticket_per_person: float,
) -> dict:
    """纯 Python 预算计算，不依赖大模型或 LangGraph。"""
    nights = max(days - 1, 0)

    hotel_cost = hotel_price_per_night * nights
    ticket_cost = attraction_ticket_per_person * people
    total_cost = hotel_cost + ticket_cost

    return {
        "days": days,
        "nights": nights,
        "people": people,
        "hotel_price_per_night": float(hotel_price_per_night),
        "hotel_cost": float(hotel_cost),
        "attraction_ticket_per_person": float(
            attraction_ticket_per_person
        ),
        "ticket_cost": float(ticket_cost),
        "total_cost": float(total_cost),
        "excluded_items": [
            "餐饮",
            "交通",
            "购物",
        ],
    }


@tool
def calculate_budget(runtime: ToolRuntime) -> dict:
    """根据 State 和已有工具结果确定性计算住宿与门票预算。"""
    state = runtime.state
    messages = state["messages"]

    attractions = get_latest_tool_result(
        messages,
        "search_attractions",
    )
    hotel_result = get_latest_tool_result(
        messages,
        "search_hotels_mcp",
    )

    attractions = attractions or []
    hotel_result = hotel_result or {}

    hotels = hotel_result.get("hotels", [])

    selected_hotel = min(
        hotels,
        key=lambda hotel: hotel["price_per_night"],
        default=None,
    )

    hotel_price = (
        float(selected_hotel["price_per_night"])
        if selected_hotel
        else 0.0
    )

    ticket_per_person = sum(
        float(attraction.get("ticket", 0))
        for attraction in attractions
    )

    result = compute_budget(
        days=int(state.get("days") or 0),
        people=int(state.get("people") or 0),
        hotel_price_per_night=hotel_price,
        attraction_ticket_per_person=ticket_per_person,
    )

    result["selected_hotel"] = (
        selected_hotel["name"]
        if selected_hotel
        else None
    )
    result["selected_attractions"] = [
        attraction["name"]
        for attraction in attractions
    ]

    return result

@tool
def search_travel_knowledge(
    city: str,
    query: str,
) -> list[dict]:
    """从本地向量知识库检索指定城市的景点资料和来源。"""
    return retrieve_travel_knowledge(
        query=query,
        city=city,
        limit=4,
    )