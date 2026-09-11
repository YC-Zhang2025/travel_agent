from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from travel_agent.api import create_app


class FakeTravelAgentService:
    """不调用数据库和大模型的 API 测试替身。"""

    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self.invoke_calls = []

    def open(self):
        self.opened = True
        return self

    def close(self) -> None:
        self.closed = True

    def invoke(self, *, query, user_id, thread_id) -> dict:
        self.invoke_calls.append(
            {
                "query": query,
                "user_id": user_id,
                "thread_id": thread_id,
            }
        )
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_attractions",
                            "args": {},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="成都旅行方案"),
            ],
            "city": "成都",
            "days": 3,
            "nights": 2,
            "people": 2,
            "budget": 3000.0,
            "preferences": ["美食"],
            "pace": "轻松",
            "missing_fields": [],
            "answer_is_grounded": True,
            "validation_feedback": "",
            "revision_count": 0,
        }


def test_health_endpoint_manages_service_lifecycle():
    service = FakeTravelAgentService()
    app = create_app(service_factory=lambda: service)

    with TestClient(app) as client:
        response = client.get("/health")
        assert service.opened is True

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "travel-agent",
    }
    assert service.closed is True


def test_plan_endpoint_returns_structured_result():
    service = FakeTravelAgentService()
    app = create_app(service_factory=lambda: service)

    with TestClient(app) as client:
        response = client.post(
            "/v1/trips/plan",
            json={
                "query": "  去成都3天  ",
                "user_id": " user-001 ",
                "thread_id": " trip-001 ",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["city"] == "成都"
    assert body["tool_names"] == ["search_attractions"]
    assert body["answer_is_grounded"] is True
    assert body["final_answer"] == "成都旅行方案"
    assert service.invoke_calls == [
        {
            "query": "去成都3天",
            "user_id": "user-001",
            "thread_id": "trip-001",
        }
    ]


def test_plan_endpoint_rejects_blank_query():
    service = FakeTravelAgentService()
    app = create_app(service_factory=lambda: service)

    with TestClient(app) as client:
        response = client.post(
            "/v1/trips/plan",
            json={
                "query": "   ",
                "user_id": "user-001",
                "thread_id": "trip-001",
            },
        )

    assert response.status_code == 422
    assert service.invoke_calls == []
