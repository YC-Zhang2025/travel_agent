import os
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

DEFAULT_CHECKPOINT_PATH = Path("data/checkpoints.sqlite")
DEFAULT_MEMORY_PATH = Path("data/memory.sqlite")


class TravelAgentService:
    """管理旅行 Agent 的数据库连接、Graph 生命周期和调用入口。"""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        memory_path: str | Path = DEFAULT_MEMORY_PATH,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.memory_path = Path(memory_path)
        self._checkpoint_connection: sqlite3.Connection | None = None
        self._memory_connection: sqlite3.Connection | None = None
        self._graph: Any | None = None
        self._invoke_lock = RLock()

    def open(self) -> "TravelAgentService":
        """打开数据库并构建一次可复用的 LangGraph。"""
        if self._graph is not None:
            return self

        # 延迟导入 Graph，使健康检查和 API 契约测试不依赖模型密钥。
        from travel_agent.graph import build_travel_graph

        self.checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.memory_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._checkpoint_connection = sqlite3.connect(
            self.checkpoint_path,
            check_same_thread=False,
        )
        self._memory_connection = sqlite3.connect(
            self.memory_path,
            check_same_thread=False,
            isolation_level=None,
        )

        checkpointer = SqliteSaver(
            self._checkpoint_connection
        )
        store = SqliteStore(self._memory_connection)
        store.setup()

        self._graph = build_travel_graph(
            checkpointer=checkpointer,
            store=store,
        )
        return self

    def close(self) -> None:
        """关闭服务持有的数据库连接。"""
        self._graph = None

        if self._checkpoint_connection is not None:
            self._checkpoint_connection.close()
            self._checkpoint_connection = None

        if self._memory_connection is not None:
            self._memory_connection.close()
            self._memory_connection = None

    def invoke(
        self,
        *,
        query: str,
        user_id: str,
        thread_id: str,
    ) -> dict:
        """使用统一入口运行一次旅行规划请求。"""
        if self._graph is None:
            raise RuntimeError(
                "TravelAgentService 尚未打开，请先调用 open()"
            )

        if not query.strip():
            raise ValueError("query 不能为空")
        if not user_id.strip():
            raise ValueError("user_id 不能为空")
        if not thread_id.strip():
            raise ValueError("thread_id 不能为空")

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        # SQLite 版本先串行执行 Graph，避免多个请求共用连接时
        # 发生嵌套事务冲突。后续可替换为数据库连接池。
        with self._invoke_lock:
            return self._graph.invoke(
                {
                    "messages": [
                        HumanMessage(content=query)
                    ]
                },
                config=config,
                context={"user_id": user_id},
            )

    def __enter__(self) -> "TravelAgentService":
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def create_service_from_env() -> TravelAgentService:
    """根据环境变量创建服务，未设置时使用项目默认路径。"""
    return TravelAgentService(
        checkpoint_path=os.getenv(
            "TRAVEL_CHECKPOINT_DB",
            str(DEFAULT_CHECKPOINT_PATH),
        ),
        memory_path=os.getenv(
            "TRAVEL_MEMORY_DB",
            str(DEFAULT_MEMORY_PATH),
        ),
    )
