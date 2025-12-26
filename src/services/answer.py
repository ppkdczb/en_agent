from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Literal

from src.model import ClozeTest, ReadingTask
from src.utils import LLMService, parse_json_content, remove_think_tags

ChoiceLabel = Literal["A", "B", "C", "D"]


class AnswerParseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ClozeAnswerResult:
    answers: list[ChoiceLabel]
    raw: str
    time_used: float


@dataclass(frozen=True, slots=True)
class ReadingAnswerItem:
    id: int
    answer: ChoiceLabel


@dataclass(frozen=True, slots=True)
class ReadingAnswerResult:
    answers: list[ReadingAnswerItem]
    raw: str
    time_used: float


def _normalize_choice(value: object) -> ChoiceLabel:
    text = str(value).strip().upper()
    if text not in {"A", "B", "C", "D"}:
        raise AnswerParseError(f"invalid choice label: {value!r}")
    return text  # type: ignore[return-value]


def _extract_json_dict(raw_text: str) -> dict:
    """
    Extract a JSON object from LLM output.

    This intentionally avoids tool/function calling so it can work across providers.
    """
    cleaned = remove_think_tags(raw_text).strip()
    candidate = parse_json_content(cleaned).strip()
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: find the first {...} block.
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise AnswerParseError("no JSON object found in LLM output")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise AnswerParseError(f"failed to parse JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise AnswerParseError("JSON root must be an object")
    return parsed


def build_cloze_prompt(task: ClozeTest) -> str:
    """
    Build the user prompt for a cloze task.

    Convention:
    - `task.content` contains placeholders like `<question_1>`, `<question_2>`, ...
    - `task.options[i]` corresponds to `<question_{i+1}>`
    """
    lines: list[str] = []
    lines.append("You are taking an English cloze test.")
    lines.append("Fill each blank with the best option (A/B/C/D).")
    lines.append("")
    lines.append("Text:")
    lines.append(task.content)
    lines.append("")
    lines.append("Options:")
    for i, opt in enumerate(task.options, start=1):
        lines.append(f"<question_{i}>")
        lines.append(f"A. {opt.A}")
        lines.append(f"B. {opt.B}")
        lines.append(f"C. {opt.C}")
        lines.append(f"D. {opt.D}")
        lines.append("")
    lines.append("Output ONLY valid JSON (no markdown, no extra keys):")
    lines.append('{"answers":["A","B","C"]}')
    return "\n".join(lines).strip()


def answer_cloze(llm: LLMService, task: ClozeTest) -> ClozeAnswerResult:
    system_prompt = "You are a careful English test taker. Return only the requested JSON."
    user_prompt = build_cloze_prompt(task)

    start = time.perf_counter()
    raw = llm.invoke(user_prompt=user_prompt, system_prompt=system_prompt)
    time_used = time.perf_counter() - start

    payload = _extract_json_dict(str(raw))
    answers_obj = payload.get("answers")
    if not isinstance(answers_obj, list):
        raise AnswerParseError("JSON must contain 'answers' as a list")

    answers: list[ChoiceLabel] = [_normalize_choice(x) for x in answers_obj]
    if len(answers) != len(task.options):
        raise AnswerParseError(
            f"answers length mismatch: got {len(answers)}, expected {len(task.options)}"
        )
    return ClozeAnswerResult(answers=answers, raw=str(raw), time_used=time_used)


def build_reading_prompt(task: ReadingTask) -> str:
    lines: list[str] = []
    lines.append("You are taking an English reading comprehension multiple-choice test.")
    lines.append("Choose the best answer (A/B/C/D) for each question.")
    lines.append("")
    lines.append("Passage:")
    lines.append(task.content)
    lines.append("")
    lines.append("Questions:")
    for q in task.questions:
        lines.append(f"Q{q.id}. {q.prompt}")
        lines.append(f"A. {q.options.A}")
        lines.append(f"B. {q.options.B}")
        lines.append(f"C. {q.options.C}")
        lines.append(f"D. {q.options.D}")
        lines.append("")
    lines.append("Output ONLY valid JSON (no markdown, no extra keys):")
    lines.append('{"answers":[{"id":1,"answer":"A"},{"id":2,"answer":"D"}]}')
    return "\n".join(lines).strip()


def answer_reading(llm: LLMService, task: ReadingTask) -> ReadingAnswerResult:
    system_prompt = "You are a careful English test taker. Return only the requested JSON."
    user_prompt = build_reading_prompt(task)

    start = time.perf_counter()
    raw = llm.invoke(user_prompt=user_prompt, system_prompt=system_prompt)
    time_used = time.perf_counter() - start

    payload = _extract_json_dict(str(raw))
    answers_obj = payload.get("answers")
    if not isinstance(answers_obj, list):
        raise AnswerParseError("JSON must contain 'answers' as a list")

    answers: list[ReadingAnswerItem] = []
    for item in answers_obj:
        if not isinstance(item, dict):
            raise AnswerParseError("each answers item must be an object")
        qid = item.get("id")
        ans = item.get("answer")
        if not isinstance(qid, int):
            raise AnswerParseError("answers[].id must be an int")
        answers.append(ReadingAnswerItem(id=qid, answer=_normalize_choice(ans)))

    expected_ids = [q.id for q in task.questions]
    got_ids = [a.id for a in answers]
    if sorted(got_ids) != sorted(expected_ids):
        raise AnswerParseError(f"answers question ids mismatch: got={got_ids}, expected={expected_ids}")

    return ReadingAnswerResult(answers=answers, raw=str(raw), time_used=time_used)
