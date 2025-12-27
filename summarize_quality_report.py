import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ItemSummary:
    type: str  # "cloze" | "reading"
    source: str
    name: str
    score: float
    aspect_scores: Optional[Dict[str, float]]
    ai_accuracy: Optional[float]
    ai_correct: Optional[int]
    ai_total: Optional[int]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _compute_ai_accuracy(item: Dict[str, Any]) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    llm = item.get("llm_judge")
    if not isinstance(llm, dict):
        return None, None, None

    if item.get("type") == "cloze":
        per_blank = llm.get("per_blank")
        if isinstance(per_blank, list) and per_blank:
            matches = [b.get("matches_provided") for b in per_blank]
            total = len(matches)
            correct = sum(1 for m in matches if m is True)
            return correct / total, correct, total

        predicted = llm.get("predicted_answers")
        mismatch = llm.get("answer_mismatch_indices")
        if isinstance(predicted, list) and predicted:
            total = len(predicted)
            if isinstance(mismatch, list):
                correct = total - len(mismatch)
                return correct / total, correct, total
            return None, None, total

    if item.get("type") == "reading":
        per_question = llm.get("per_question")
        if isinstance(per_question, list) and per_question:
            matches = [q.get("matches_provided") for q in per_question]
            total = len(matches)
            correct = sum(1 for m in matches if m is True)
            return correct / total, correct, total

        predicted = llm.get("predicted_answers")
        mismatch = llm.get("answer_mismatch_ids")
        if isinstance(predicted, dict) and predicted:
            total = len(predicted)
            if isinstance(mismatch, list):
                correct = total - len(mismatch)
                return correct / total, correct, total
            return None, None, total

    return None, None, None


def _extract_aspect_scores(item: Dict[str, Any]) -> Optional[Dict[str, float]]:
    llm = item.get("llm_judge")
    if not isinstance(llm, dict):
        return None
    scores = llm.get("overall_scores")
    if not isinstance(scores, dict) or not scores:
        return None
    out: Dict[str, float] = {}
    for k, v in scores.items():
        fv = _safe_float(v)
        if fv is None:
            continue
        out[str(k)] = fv
    return out or None


def summarize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    items = report.get("items") if isinstance(report.get("items"), list) else []

    item_summaries: List[ItemSummary] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        t = str(it.get("type", ""))
        if t not in {"cloze", "reading"}:
            continue

        source = str(it.get("source_file") or it.get("path") or "")
        if t == "cloze":
            name = f"id={it.get('id')}"
        else:
            name = str(it.get("title") or "")

        score = _safe_float(it.get("score")) or 0.0
        aspect_scores = _extract_aspect_scores(it)
        acc, correct, total = _compute_ai_accuracy(it)
        item_summaries.append(
            ItemSummary(
                type=t,
                source=source,
                name=name,
                score=score,
                aspect_scores=aspect_scores,
                ai_accuracy=acc,
                ai_correct=correct,
                ai_total=total,
            )
        )

    def _avg(xs: List[float]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    cloze_items = [x for x in item_summaries if x.type == "cloze"]
    reading_items = [x for x in item_summaries if x.type == "reading"]

    cloze_scores = [x.score for x in cloze_items]
    reading_scores = [x.score for x in reading_items]

    cloze_accs = [x.ai_accuracy for x in cloze_items if x.ai_accuracy is not None]
    reading_accs = [x.ai_accuracy for x in reading_items if x.ai_accuracy is not None]

    def _avg_aspects(items_: List[ItemSummary]) -> Optional[Dict[str, float]]:
        # Average each aspect across items that contain it.
        bucket: Dict[str, List[float]] = {}
        for it in items_:
            if not it.aspect_scores:
                continue
            for k, v in it.aspect_scores.items():
                bucket.setdefault(k, []).append(float(v))
        if not bucket:
            return None
        return {k: _avg(vs) for k, vs in sorted(bucket.items(), key=lambda x: x[0])}

    summary = {
        "generated_at": report.get("generated_at"),
        "model": report.get("model"),
        "source_settings": report.get("settings", {}),
        "counts": {
            "cloze": len(cloze_items),
            "reading": len(reading_items),
            "total": len(item_summaries),
        },
        "overall": {
            "cloze_avg_score": _avg(cloze_scores),
            "reading_avg_score": _avg(reading_scores),
            "cloze_avg_aspect_scores": _avg_aspects(cloze_items),
            "reading_avg_aspect_scores": _avg_aspects(reading_items),
            "cloze_avg_ai_accuracy": _avg([x for x in cloze_accs if x is not None]) if cloze_accs else None,
            "reading_avg_ai_accuracy": _avg([x for x in reading_accs if x is not None]) if reading_accs else None,
        },
        "per_passage": [
            {
                "type": x.type,
                "source": x.source,
                "name": x.name,
                "score": x.score,
                "aspect_scores": x.aspect_scores,
                "ai_accuracy": x.ai_accuracy,
                "ai_correct": x.ai_correct,
                "ai_total": x.ai_total,
            }
            for x in item_summaries
        ],
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize key metrics from quality_report_*.json")
    parser.add_argument("report_pos", nargs="?", default=None, help="Path to a quality_report_*.json (positional).")
    parser.add_argument("--report", default=None, help="Path to a quality_report_*.json (optional).")
    parser.add_argument("--out", default=None, help="Write summary JSON to this path (optional).")
    args = parser.parse_args()

    report_arg = args.report or args.report_pos
    if not report_arg:
        parser.error("Please provide a report path (positional) or via --report.")

    report_path = Path(report_arg)
    report = _read_json(report_path)
    summary = summarize_report(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved summary: {out_path}")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
