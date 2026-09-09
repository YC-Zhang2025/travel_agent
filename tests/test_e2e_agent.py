import json
import re
import subprocess
import sys
import uuid
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUERY = (
    "去成都3天，两个人，预算3000元，喜欢美食和文化，"
    "行程轻松，请基于知识库规划。"
)


@pytest.mark.e2e
def test_complete_travel_agent_flow():
    thread_id = f"eval-{uuid.uuid4().hex}"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "travel_agent.main",
            "--user-id",
            "eval-user",
            "--thread-id",
            thread_id,
            "--query",
            QUERY,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )

    output = completed.stdout + "\n" + completed.stderr
    print(output)

    # 1. 程序必须正常退出
    assert completed.returncode == 0, output

    # 2. 需求必须被正确抽取
    assert "城市：成都" in output
    assert "天数：3" in output
    assert "人数：2" in output
    assert "预算：3000" in output

    # 3. Agent 必须调用四个关键工具
    required_tools = {
        "search_attractions",
        "search_hotels_mcp",
        "calculate_budget",
        "search_travel_knowledge",
    }

    for tool_name in required_tools:
        assert (
            f"[Agent 调用工具] {tool_name}" in output
        ), f"缺少工具调用：{tool_name}\n{output}"

    # 4. 最终答案必须存在，且不能只是复述问题
    final_marker = "--- 最终方案 ---"
    assert final_marker in output

    final_answer = output.rsplit(final_marker, maxsplit=1)[1].strip()

    assert final_answer
    assert final_answer != QUERY
    assert len(final_answer) >= 100

    # 5. 最终答案必须引用 MCP 和 RAG 来源
    assert "mcp://travel-hotel-service" in final_answer
    assert "local://attractions/" in final_answer

    # 6. 读取最后一次预算工具的结构化结果
    budget_results = re.findall(
        r"^\[工具返回\] calculate_budget: (.+)$",
        output,
        flags=re.MULTILINE,
    )

    assert budget_results, "没有找到 calculate_budget 的返回结果"

    budget = json.loads(budget_results[-1])
    total_cost = budget["total_cost"]

    # 兼容 640 与 640.0 两种显示形式
    total_cost_candidates = {
        str(total_cost),
        f"{total_cost:g}",
    }

    assert any(
        value in final_answer
        for value in total_cost_candidates
    ), (
        f"最终答案没有使用预算工具返回的总费用：{total_cost}\n"
        f"{final_answer}"
    )

    # 7. 必须说明基础预算不包含哪些项目
    for excluded_item in budget["excluded_items"]:
        assert excluded_item in final_answer