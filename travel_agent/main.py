import argparse
import sqlite3
from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from travel_agent.graph import build_travel_graph
from travel_agent.state import TravelContext

def main() -> None:
    parser = argparse.ArgumentParser(description="智能旅行规划 Agent")
    parser.add_argument(
        "--query",
        required=True,
        help="用自然语言描述旅行需求",
    )
    parser.add_argument(
        "--thread-id",
        default="default",
        help="会话线程 ID；相同 ID 会恢复之前的状态",
    )
    parser.add_argument(
        "--user-id",
        default="default-user",
        help="用户 ID；相同用户可跨线程共享旅行偏好",
    )
    args = parser.parse_args()

    database_path = Path("data/checkpoints.sqlite")
    memory_path = Path("data/memory.sqlite")

    database_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_connection = sqlite3.connect(
        database_path,
        check_same_thread=False,
    )

    memory_connection = sqlite3.connect(
        memory_path,
        check_same_thread=False,
        isolation_level=None,
    )

    try:
        checkpointer = SqliteSaver(checkpoint_connection)

        store = SqliteStore(memory_connection)
        store.setup()

        graph = build_travel_graph(
            checkpointer=checkpointer,
            store=store,
        )

        config = {
            "configurable": {
                "thread_id": args.thread_id,
            }
        }

        result = graph.invoke(
            {"messages": [HumanMessage(content=args.query)]},
            config=config,
            context=TravelContext(user_id=args.user_id),
        )
    finally:
        checkpoint_connection.close()
        memory_connection.close()

    print("\n--- 结构化旅行需求 ---")
    print(f"城市：{result.get('city')}")
    print(f"天数：{result.get('days')}")
    print(f"住宿晚数：{result.get('nights')}")
    print(f"人数：{result.get('people')}")
    print(f"预算：{result.get('budget')}")
    print(f"偏好：{result.get('preferences')}")
    print(f"节奏：{result.get('pace')}")
    for message in result["messages"]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                print(
                    f"[Agent 调用工具] {tool_call['name']} "
                    f"参数={tool_call['args']}"
                )
        elif isinstance(message, ToolMessage):
            print(
                f"[工具返回] {message.name}: "
                f"{message.content}"
            )

    print("\n--- 最终方案 ---")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()