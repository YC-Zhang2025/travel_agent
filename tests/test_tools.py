from travel_agent.tools import (
    calculate_budget,
    compute_budget,
    filter_attractions,
    search_attractions,
    search_hotels_mcp,
    search_travel_knowledge,
)

def test_calculate_budget():
    result = compute_budget(
        days=3,
        people=2,
        hotel_price_per_night=320,
        attraction_ticket_per_person=105,
    )

    assert result["days"] == 3
    assert result["nights"] == 2
    assert result["people"] == 2
    assert result["hotel_cost"] == 640
    assert result["ticket_cost"] == 210
    assert result["total_cost"] == 850


def test_calculate_budget_for_one_day():
    result = compute_budget(
        days=1,
        people=2,
        hotel_price_per_night=320,
        attraction_ticket_per_person=50,
    )

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

def test_state_driven_tools_hide_runtime_parameters():
    state_driven_tools = [
        search_attractions,
        search_hotels_mcp,
        search_travel_knowledge,
        calculate_budget,
    ]

    for agent_tool in state_driven_tools:
        assert agent_tool.args == {}