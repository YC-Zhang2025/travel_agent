from mcp.server import MCPServer
from pydantic import BaseModel


mcp = MCPServer(
    "travel-hotel-service",
    instructions=(
        "提供项目本地演示酒店数据。"
        "所有价格均为项目样例，不代表实时酒店报价。"
    ),
)

class Hotel(BaseModel):
    name: str
    price_per_night: float


class HotelSearchResult(BaseModel):
    city: str
    max_price_per_night: float
    hotels: list[Hotel]
    source: str
    data_type: str


HOTELS = {
    "成都": [
        {
            "name": "春熙路舒适酒店",
            "price_per_night": 320,
        },
        {
            "name": "宽窄巷子精品酒店",
            "price_per_night": 480,
        },
    ],
    "北京": [
        {
            "name": "前门快捷酒店",
            "price_per_night": 420,
        },
        {
            "name": "国贸商务酒店",
            "price_per_night": 680,
        },
    ],
}


@mcp.tool()
def search_hotels_mcp(
    city: str,
    max_price_per_night: float,
) -> HotelSearchResult:
    """查询指定城市中不超过每晚预算的本地演示酒店。"""
    hotels = HOTELS.get(city, [])

    matched_hotels = [
        Hotel(**hotel)
        for hotel in hotels
        if hotel["price_per_night"] <= max_price_per_night
    ]

    return HotelSearchResult(
        city=city,
        max_price_per_night=max_price_per_night,
        hotels=matched_hotels,
        source=f"mcp://travel-hotel-service/hotels/{city}",
        data_type="local_demo_data",
    )

if __name__ == "__main__":
    # 默认启动 stdio transport。
    # stdin/stdout 用于传输 MCP JSON-RPC 消息。
    mcp.run()