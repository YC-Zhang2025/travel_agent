import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from langsmith import testing as t


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_CASES = [
    {
        "name": "chengdu_complete_request",
        "query": (
            "去成都3天，两个人，预算3000元，"
            "喜欢美食和文化，行程轻松，请基于知识库规划。"
        ),
        "expected_phrases": [
            "城市：成都",
            "宽窄巷子",
            "锦里",
            "640",
            "校验通过： True",
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
        "name": "missing_city_clarification",
        "query": "3天，两个人，预算3000元。",
        "expected_phrases": [
            "城市：None",
            "为了给你生成可靠的旅行方案",
            "目的地城市",
        ],
        "required_tools": [],
        "expects_grounded_answer": False,
    },
]


@pytest.mark.e2e
@pytest.mark.evaluation
@pytest.mark.langsmith
@pytest.mark.parametrize(
    "case",
    EVALUATION_CASES,
    ids=lambda case: case["name"],
)
def test_travel_agent_dataset(case):
    """运行旅行 Agent，并把结果和评分同步到 LangSmith。"""
    unique_id = uuid.uuid4().hex

    t.log_inputs(
        {
            "case_name": case["name"],
            "query": case["query"],
        }
    )

    t.log_reference_outputs(
        {
            "expected_phrases": case["expected_phrases"],
            "required_tools": case["required_tools"],
            "expects_grounded_answer": (
                case["expects_grounded_answer"]
            ),
        }
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "travel_agent.main",
            "--user-id",
            f"dataset-user-{unique_id}",
            "--thread-id",
            f"dataset-thread-{unique_id}",
            "--query",
            case["query"],
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )

    stdout = completed.stdout
    diagnostic_output = (
        completed.stdout
        + "\n"
        + completed.stderr
    )

    expected_content_ok = all(
        phrase in stdout
        for phrase in case["expected_phrases"]
    )

    tool_path_ok = all(
        f"[Agent 调用工具] {tool_name}" in stdout
        for tool_name in case["required_tools"]
    )

    if case["expects_grounded_answer"]:
        behavior_ok = "校验通过： True" in stdout
    else:
        behavior_ok = (
            "目的地城市" in stdout
            and "[Agent 调用工具]" not in stdout
        )

    process_ok = completed.returncode == 0

    t.log_outputs(
        {
            "stdout": stdout,
            "returncode": completed.returncode,
        }
    )

    t.log_feedback(
        key="process_success",
        score=process_ok,
    )
    t.log_feedback(
        key="expected_content",
        score=expected_content_ok,
    )
    t.log_feedback(
        key="tool_path",
        score=tool_path_ok,
    )
    t.log_feedback(
        key="grounded_or_clarified",
        score=behavior_ok,
    )

    assert process_ok, diagnostic_output
    assert expected_content_ok, diagnostic_output
    assert tool_path_ok, diagnostic_output
    assert behavior_ok, diagnostic_output