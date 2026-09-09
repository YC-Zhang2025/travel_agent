import asyncio

from mcp import Client

from mcp_servers.hotel_server import mcp


async def main() -> None:
    # 直接连接 Server 对象，用于单元测试。
    async with Client(mcp) as client:
        tools_result = await client.list_tools()

        print("--- MCP 工具列表 ---")
        for tool in tools_result.tools:
            print(f"名称：{tool.name}")
            print(f"说明：{tool.description}")
            print(f"输入 Schema：{tool.input_schema}")
            print(f"输出 Schema：{tool.output_schema}")

        result = await client.call_tool(
            "search_hotels_mcp",
            {
                "city": "成都",
                "max_price_per_night": 400,
            },
        )

        print("\n--- MCP 调用结果 ---")
        print("是否错误：", result.is_error)
        print("文本内容：", result.content)
        print("结构化结果：", result.structured_content)

if __name__ == "__main__":
    asyncio.run(main())