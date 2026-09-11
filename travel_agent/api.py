import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, HTTPException, Request
from langchain_core.messages import AIMessage

from travel_agent.api_schemas import (
    HealthResponse,
    PlanRequest,
    TravelPlanResponse,
)
from travel_agent.service import (
    TravelAgentService,
    create_service_from_env,
)


logger = logging.getLogger(__name__)
router = APIRouter()


def collect_tool_names(result: dict) -> list[str]:
    """按首次出现顺序收集 Agent 调用过的工具。"""
    names = [
        tool_call["name"]
        for message in result["messages"]
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
    ]
    return list(dict.fromkeys(names))


def build_plan_response(result: dict) -> TravelPlanResponse:
    """把 LangGraph State 转换成稳定的 HTTP 响应。"""
    return TravelPlanResponse(
        city=result.get("city"),
        days=result.get("days"),
        nights=result.get("nights"),
        people=result.get("people"),
        budget=result.get("budget"),
        preferences=result.get("preferences") or [],
        pace=result.get("pace"),
        missing_fields=result.get("missing_fields") or [],
        answer_is_grounded=result.get("answer_is_grounded"),
        validation_feedback=result.get("validation_feedback"),
        revision_count=int(result.get("revision_count") or 0),
        tool_names=collect_tool_names(result),
        final_answer=str(result["messages"][-1].content),
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
)
def health_check() -> HealthResponse:
    """供本地开发、容器和部署平台检查服务状态。"""
    return HealthResponse()


@router.post(
    "/v1/trips/plan",
    response_model=TravelPlanResponse,
    tags=["travel"],
)
def plan_trip(
    payload: PlanRequest,
    request: Request,
) -> TravelPlanResponse:
    """运行一次旅行规划请求。"""
    service: TravelAgentService = (
        request.app.state.travel_service
    )

    try:
        result = service.invoke(
            query=payload.query,
            user_id=payload.user_id,
            thread_id=payload.thread_id,
        )
    except Exception as exc:
        logger.exception("旅行规划请求执行失败")
        raise HTTPException(
            status_code=503,
            detail="旅行规划服务暂时不可用",
        ) from exc

    return build_plan_response(result)


def create_app(
    service_factory: Callable[
        [], TravelAgentService
    ] = create_service_from_env,
) -> FastAPI:
    """创建支持依赖替换和生命周期管理的 FastAPI 应用。"""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        service = service_factory()
        service.open()
        application.state.travel_service = service

        try:
            yield
        finally:
            service.close()

    application = FastAPI(
        title="Travel Agent API",
        version="0.1.0",
        description=(
            "基于 LangGraph、Memory、RAG 和 MCP 的旅行规划服务"
        ),
        lifespan=lifespan,
    )
    application.include_router(router)
    return application


app = create_app()
