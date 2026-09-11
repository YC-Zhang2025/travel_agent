from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class PlanRequest(BaseModel):
    """创建旅行方案的 HTTP 请求。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(max_length=2000)
    user_id: str = Field(max_length=128)
    thread_id: str = Field(max_length=128)

    @field_validator("query", "user_id", "thread_id")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("不能为空")
        return normalized


class HealthResponse(BaseModel):
    """服务健康状态。"""

    status: Literal["ok"] = "ok"
    service: str = "travel-agent"


class TravelPlanResponse(BaseModel):
    """旅行 Agent 的结构化 HTTP 响应。"""

    city: str | None = None
    days: int | None = None
    nights: int | None = None
    people: int | None = None
    budget: float | None = None
    preferences: list[str] = Field(default_factory=list)
    pace: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    answer_is_grounded: bool | None = None
    validation_feedback: str | None = None
    revision_count: int = 0
    tool_names: list[str] = Field(default_factory=list)
    final_answer: str
