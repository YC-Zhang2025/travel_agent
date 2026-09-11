import uuid
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langsmith import testing as t

from travel_agent.grounding import (
    find_unsupported_answer_phrases,
)
from travel_agent.service import TravelAgentService


EVALUATION_CASES = [
    {
        "name": "chengdu_complete_request",
        "query": (
            "去成都3天，两个人，预算3000元，"
            "喜欢美食和文化，行程轻松，请基于知识库规划。"
        ),
        "expected_state": {
            "city": "成都",
            "days": 3,
            "people": 2,
            "budget": 3000.0,
        },
        "expected_answer_phrases": [
            "宽窄巷子",
            "锦里",
            "640",
            "mcp://travel-hotel-service/hotels/成都",
            "local://attractions/chengdu/",
        ],
        "required_tools": [
            "search_attractions",
            "search_hotels_mcp",
            "search_travel_knowledge",
            "calculate_budget",
        ],
        "expects_grounded_answer": True,
    },
    {
        "name": "beijing_complete_request",
        "query": (
            "去北京3天，两个人，预算4000元，"
            "喜欢历史和美食，行程轻松，请基于知识库规划。"
        ),
        "expected_state": {
            "city": "北京",
            "days": 3,
            "people": 2,
            "budget": 4000.0,
        },
        "expected_answer_phrases": [
            "故宫",
            "南锣鼓巷",
            "960",
            "mcp://travel-hotel-service/hotels/北京",
            "local://attractions/beijing/",
        ],
        "required_tools": [
            "search_attractions",
            "search_hotels_mcp",
            "search_travel_knowledge",
            "calculate_budget",
        ],
        "expects_grounded_answer": True,
    },
    {
        "name": "missing_city_clarification",
        "query": "3天，两个人，预算3000元。",
        "expected_state": {
            "city": None,
            "days": 3,
            "people": 2,
            "budget": 3000.0,
        },
        "expected_answer_phrases": [
            "为了给你生成可靠的旅行方案",
            "目的地城市",
        ],
        "required_tools": [],
        "expects_grounded_answer": False,
    },
]


def create_test_service(tmp_path: Path) -> TravelAgentService:
    """为每个评测测试创建隔离的短期和长期记忆数据库。"""
    return TravelAgentService(
        checkpoint_path=tmp_path / "checkpoints.sqlite",
        memory_path=tmp_path / "memory.sqlite",
    )


def collect_tool_names(result: dict) -> list[str]:
    """从结构化消息中收集本次运行的工具调用名称。"""
    return [
        tool_call["name"]
        for message in result["messages"]
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
    ]


def summarize_result(result: dict) -> dict:
    """生成适合上传到 LangSmith 的结构化评测结果。"""
    final_answer = str(result["messages"][-1].content)
    return {
        "city": result.get("city"),
        "days": result.get("days"),
        "people": result.get("people"),
        "budget": result.get("budget"),
        "preferences": result.get("preferences"),
        "pace": result.get("pace"),
        "answer_is_grounded": result.get(
            "answer_is_grounded"
        ),
        "revision_count": result.get("revision_count", 0),
        "tool_names": collect_tool_names(result),
        "final_answer": final_answer,
    }


@pytest.mark.e2e
@pytest.mark.evaluation
@pytest.mark.langsmith
@pytest.mark.parametrize(
    "case",
    EVALUATION_CASES,
    ids=lambda case: case["name"],
)
def test_travel_agent_dataset(case, tmp_path):
    """进程内运行 Agent，并把结构化结果和评分同步到 LangSmith。"""
    unique_id = uuid.uuid4().hex

    t.log_inputs(
        {
            "case_name": case["name"],
            "query": case["query"],
        }
    )
    t.log_reference_outputs(
        {
            "expected_state": case["expected_state"],
            "expected_answer_phrases": (
                case["expected_answer_phrases"]
            ),
            "required_tools": case["required_tools"],
            "expects_grounded_answer": (
                case["expects_grounded_answer"]
            ),
        }
    )

    with create_test_service(tmp_path) as service:
        result = service.invoke(
            query=case["query"],
            user_id=f"dataset-user-{unique_id}",
            thread_id=f"dataset-thread-{unique_id}",
        )

    summary = summarize_result(result)
    final_answer = summary["final_answer"]
    tool_names = set(summary["tool_names"])

    state_ok = all(
        summary.get(field) == expected_value
        for field, expected_value
        in case["expected_state"].items()
    )
    answer_content_ok = all(
        phrase in final_answer
        for phrase in case["expected_answer_phrases"]
    )
    expected_content_ok = state_ok and answer_content_ok
    tool_path_ok = all(
        tool_name in tool_names
        for tool_name in case["required_tools"]
    )

    if case["expects_grounded_answer"]:
        behavior_ok = (
            summary["answer_is_grounded"] is True
            and not find_unsupported_answer_phrases(
                final_answer
            )
        )
    else:
        behavior_ok = (
            "目的地城市" in final_answer
            and not tool_names
        )

    t.log_outputs(summary)
    t.log_feedback(key="process_success", score=True)
    t.log_feedback(
        key="expected_content",
        score=expected_content_ok,
    )
    t.log_feedback(key="tool_path", score=tool_path_ok)
    t.log_feedback(
        key="grounded_or_clarified",
        score=behavior_ok,
    )

    assert expected_content_ok, summary
    assert tool_path_ok, summary
    assert behavior_ok, summary


@pytest.mark.e2e
@pytest.mark.evaluation
@pytest.mark.langsmith
def test_cross_thread_long_term_memory(tmp_path):
    """验证同一用户在不同 thread 中可以继承长期偏好和节奏。"""
    unique_id = uuid.uuid4().hex
    user_id = f"memory-user-{unique_id}"

    seed_query = "我喜欢美食和文化，行程轻松。"
    recall_query = (
        "去北京3天，两个人，预算4000元，"
        "请基于知识库规划。"
    )

    t.log_inputs(
        {
            "case_name": "cross_thread_long_term_memory",
            "seed_query": seed_query,
            "recall_query": recall_query,
        }
    )
    t.log_reference_outputs(
        {
            "expected_preferences": ["美食", "文化"],
            "expected_pace": "轻松",
            "threads_must_differ": True,
        }
    )

    with create_test_service(tmp_path) as service:
        seed_result = service.invoke(
            query=seed_query,
            user_id=user_id,
            thread_id=f"memory-seed-{unique_id}",
        )
        recall_result = service.invoke(
            query=recall_query,
            user_id=user_id,
            thread_id=f"memory-recall-{unique_id}",
        )

    seed_summary = summarize_result(seed_result)
    recall_summary = summarize_result(recall_result)
    preferences = recall_summary["preferences"] or []
    tool_names = set(recall_summary["tool_names"])

    memory_ok = (
        "美食" in preferences
        and "文化" in preferences
        and recall_summary["pace"] == "轻松"
    )
    expected_content_ok = (
        recall_summary["city"] == "北京"
        and memory_ok
    )
    required_tools = {
        "search_attractions",
        "search_hotels_mcp",
        "search_travel_knowledge",
        "calculate_budget",
    }
    tool_path_ok = required_tools.issubset(tool_names)
    behavior_ok = (
        recall_summary["answer_is_grounded"] is True
        and not find_unsupported_answer_phrases(
            recall_summary["final_answer"]
        )
    )

    t.log_outputs(
        {
            "seed": seed_summary,
            "recall": recall_summary,
        }
    )
    t.log_feedback(key="process_success", score=True)
    t.log_feedback(
        key="expected_content",
        score=expected_content_ok,
    )
    t.log_feedback(key="tool_path", score=tool_path_ok)
    t.log_feedback(
        key="grounded_or_clarified",
        score=behavior_ok,
    )
    t.log_feedback(key="memory_recall", score=memory_ok)

    assert expected_content_ok, recall_summary
    assert tool_path_ok, recall_summary
    assert behavior_ok, recall_summary
