from travel_agent.grounding import (
    find_unsupported_answer_phrases,
)


def test_detects_unsupported_mcp_source_label():
    answer = (
        "酒店：春熙路舒适酒店。\n"
        "预订渠道：mcp://travel-hotel-service/hotels/成都"
    )

    violations = find_unsupported_answer_phrases(answer)

    assert violations == ["预订渠道"]


def test_allows_mcp_information_source_label():
    answer = (
        "酒店：春熙路舒适酒店。\n"
        "信息来源：mcp://travel-hotel-service/hotels/成都"
    )

    violations = find_unsupported_answer_phrases(answer)

    assert violations == []
