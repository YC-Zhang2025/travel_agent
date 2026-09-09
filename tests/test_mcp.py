import json

from travel_agent.mcp_client import (
    search_hotels_via_mcp as search_hotels_mcp,
)

def call_tool(arguments: dict):
    """兼容 LangChain Tool 和普通 Python 函数。"""
    if hasattr(search_hotels_mcp, "invoke"):
        result = search_hotels_mcp.invoke(arguments)
    else:
        result = search_hotels_mcp(**arguments)

    if isinstance(result, str):
        return json.loads(result)

    return result


def test_mcp_returns_filtered_hotels():
    result = call_tool(
        {
            "city": "成都",
            "max_price_per_night": 400,
        }
    )

    assert result["city"] == "成都"
    assert result["max_price_per_night"] == 400
    assert result["data_type"] == "local_demo_data"
    assert result["source"].startswith("mcp://")

    assert len(result["hotels"]) == 1
    assert result["hotels"][0]["name"] == "春熙路舒适酒店"
    assert result["hotels"][0]["price_per_night"] == 320


def test_mcp_applies_price_limit():
    result = call_tool(
        {
            "city": "成都",
            "max_price_per_night": 300,
        }
    )

    assert result["hotels"] == []


def test_mcp_unknown_city_returns_empty_list():
    result = call_tool(
        {
            "city": "不存在的城市",
            "max_price_per_night": 1000,
        }
    )

    assert result["city"] == "不存在的城市"
    assert result["hotels"] == []
    assert result["source"].startswith("mcp://")