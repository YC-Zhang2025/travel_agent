import asyncio
import sys
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]


async def _search_hotels_async(
    city: str,
    max_price_per_night: float,
) -> dict[str, Any]:
    """通过 stdio MCP 调用独立酒店服务。"""
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_servers.hotel_server"],
        cwd=PROJECT_ROOT,
    )

    async with Client(server_parameters) as client:
        tools_result = await client.list_tools()

        available_tools = {
            tool.name
            for tool in tools_result.tools
        }

        if "search_hotels_mcp" not in available_tools:
            raise RuntimeError(
                "MCP Server 未提供 search_hotels_mcp 工具"
            )

        result = await client.call_tool(
            "search_hotels_mcp",
            {
                "city": city,
                "max_price_per_night": max_price_per_night,
            },
        )

        if result.is_error:
            raise RuntimeError(
                f"MCP 酒店查询失败：{result.content}"
            )

        if result.structured_content is None:
            raise RuntimeError(
                "MCP 酒店查询没有返回结构化结果"
            )

        return result.structured_content


def search_hotels_via_mcp(
    city: str,
    max_price_per_night: float,
) -> dict[str, Any]:
    """同步入口，供当前同步 LangGraph CLI 调用。"""
    return asyncio.run(
        _search_hotels_async(
            city=city,
            max_price_per_night=max_price_per_night,
        )
    )