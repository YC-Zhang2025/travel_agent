from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from typing_extensions import TypedDict

class TravelRequest(BaseModel):
    """从用户自然语言中提取的旅行需求。"""

    city: str | None = Field(
        default=None,
        description="目的地城市",
    )
    days: int | None = Field(
        default=None,
        ge=1,
        description="旅行天数",
    )
    people: int | None = Field(
        default=None,
        ge=1,
        description="旅行人数",
    )
    budget: float | None = Field(
        default=None,
        ge=0,
        description="总预算，单位为人民币元",
    )
    preferences: list[str] | None = Field(
    default=None,
    description="本轮明确提到的旅行偏好；未提及时为 null",
    )
    pace: str | None = Field(
        default=None,
        description="行程节奏，例如轻松、适中、紧凑",
    )

class AnswerReview(BaseModel):
    """最终旅行方案的事实依据检查结果。"""

    is_grounded: bool = Field(
        description="最终答案中的事实是否全部能由工具结果支持",
    )
    feedback: str = Field(
        default="",
        description="不合格时需要修正的问题；合格时为空字符串",
    )

class TravelState(MessagesState):
    """旅行 Agent 在各节点之间共享的状态。"""

    city: str | None
    days: int | None
    nights: int | None
    people: int | None
    budget: float | None
    preferences: list[str]
    pace: str | None
    missing_fields: list[str]
    answer_is_grounded: bool | None
    validation_feedback: str | None
    revision_count: int

class TravelContext(TypedDict):
    """一次调用中不会改变的用户身份信息。"""

    user_id: str