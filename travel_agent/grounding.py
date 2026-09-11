"""最终答案的确定性事实约束。"""

UNSUPPORTED_ANSWER_PHRASES = (
    "预订来源",
    "预订渠道",
    "预订平台",
    "酒店已经预订",
    "酒店已预订",
    "足以覆盖",
    "足以满足",
    "未进行任何推测",
    "没有进行任何推测",
)


def find_unsupported_answer_phrases(answer: str) -> list[str]:
    """返回答案中命中的明确违规表达。"""
    return [
        phrase
        for phrase in UNSUPPORTED_ANSWER_PHRASES
        if phrase in answer
    ]
