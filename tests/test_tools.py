import json

from travel_agent.tools import (
    calculate_budget,
    filter_attractions,
)


def parse_result(result):
    """兼容工具返回 Python 对象或 JSON 字符串两种情况。"""
    if isinstance(result, str):
        return json.loads(result)
    return result


def test_calculate_budget():
    result = calculate_budget.invoke(
        {
            "days": 3,
            "people": 2,
            "hotel_price_per_night": 320,
            "attraction_ticket_per_person": 105,
        }
    )
    result = parse_result(result)

    assert result["days"] == 3
    assert result["nights"] == 2
    assert result["people"] == 2
    assert result["hotel_cost"] == 640
    assert result["ticket_cost"] == 210
    assert result["total_cost"] == 850


def test_calculate_budget_for_one_day():
    result = calculate_budget.invoke(
        {
            "days": 1,
            "people": 2,
            "hotel_price_per_night": 320,
            "attraction_ticket_per_person": 50,
        }
    )
    result = parse_result(result)

    assert result["nights"] == 0
    assert result["hotel_cost"] == 0
    assert result["ticket_cost"] == 100
    assert result["total_cost"] == 100


def test_search_attractions_with_multiple_preferences():
    result = filter_attractions(
        city="成都",
        preferences=["美食", "文化"],
    )

    names = {item["name"] for item in result}

    assert "锦里" in names
    assert "宽窄巷子" in names


def test_search_attractions_unknown_city():
    result = filter_attractions(
        city="不存在的城市",
        preferences=["美食"],
    )

    assert result == []