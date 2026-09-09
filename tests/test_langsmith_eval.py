import os
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
        "name": "beijing_complete_request",
        "query": (
            "去北京3天，两个人，预算4000元，"
            "喜欢历史和美食，行程轻松，请基于知识库规划。"
        ),
        "expected_phrases": [
            "城市：北京",
            "故宫",
            "南锣鼓巷",
            "960",
            "校验通过： True",
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
        "expected_phrases": [
            "城市：None",
            "为了给你生成可靠的旅行方案",
            "目的地城市",
        ],
        "required_tools": [],
        "expects_grounded_answer": False,
    },
]


def build_test_environment(tmp_path: Path) -> dict[str, str]:
    """为每个评测测试创建隔离的短期和长期记忆数据库。"""
    environment = os.environ.copy()
    environment["TRAVEL_CHECKPOINT_DB"] = str(
        tmp_path / "checkpoints.sqlite"
    )
    environment["TRAVEL_MEMORY_DB"] = str(
        tmp_path / "memory.sqlite"
    )
    environment["LANGCHAIN_CALLBACKS_BACKGROUND"] = "false"
    return environment


def run_agent(
    *,
    query: str,
    user_id: str,
    thread_id: str,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """通过真实 CLI 运行一次 Agent。"""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "travel_agent.main",
            "--user-id",
            user_id,
            "--thread-id",
            thread_id,
            "--query",
            query,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
        env=environment,
    )


@pytest.mark.e2e
@pytest.mark.evaluation
@pytest.mark.langsmith
@pytest.mark.parametrize(
    "case",
    EVALUATION_CASES,
    ids=lambda case: case["name"],
)
def test_travel_agent_dataset(case, tmp_path):
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

    completed = run_agent(
        query=case["query"],
        user_id=f"dataset-user-{unique_id}",
        thread_id=f"dataset-thread-{unique_id}",
        environment=build_test_environment(tmp_path),
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


@pytest.mark.e2e
@pytest.mark.evaluation
@pytest.mark.langsmith
def test_cross_thread_long_term_memory(tmp_path):
    """验证同一用户在不同 thread 中可以继承长期偏好和节奏。"""
    unique_id = uuid.uuid4().hex
    user_id = f"memory-user-{unique_id}"
    environment = build_test_environment(tmp_path)

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

    seed = run_agent(
        query=seed_query,
        user_id=user_id,
        thread_id=f"memory-seed-{unique_id}",
        environment=environment,
    )
    recall = run_agent(
        query=recall_query,
        user_id=user_id,
        thread_id=f"memory-recall-{unique_id}",
        environment=environment,
    )

    diagnostic_output = (
        "--- seed stdout ---\n"
        + seed.stdout
        + "\n--- seed stderr ---\n"
        + seed.stderr
        + "\n--- recall stdout ---\n"
        + recall.stdout
        + "\n--- recall stderr ---\n"
        + recall.stderr
    )

    preference_line = next(
        (
            line
            for line in recall.stdout.splitlines()
            if line.startswith("偏好：")
        ),
        "",
    )

    process_ok = (
        seed.returncode == 0
        and recall.returncode == 0
    )
    memory_ok = (
        "美食" in preference_line
        and "文化" in preference_line
        and "节奏：轻松" in recall.stdout
    )
    expected_content_ok = (
        "城市：北京" in recall.stdout
        and memory_ok
    )

    required_tools = [
        "search_attractions",
        "search_hotels_mcp",
        "search_travel_knowledge",
        "calculate_budget",
    ]
    tool_path_ok = all(
        f"[Agent 调用工具] {tool_name}" in recall.stdout
        for tool_name in required_tools
    )
    behavior_ok = "校验通过： True" in recall.stdout

    t.log_outputs(
        {
            "seed_stdout": seed.stdout,
            "recall_stdout": recall.stdout,
            "seed_returncode": seed.returncode,
            "recall_returncode": recall.returncode,
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
    t.log_feedback(
        key="memory_recall",
        score=memory_ok,
    )

    assert process_ok, diagnostic_output
    assert expected_content_ok, diagnostic_output
    assert tool_path_ok, diagnostic_output
    assert behavior_ok, diagnostic_output
