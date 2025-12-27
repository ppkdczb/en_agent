import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar
from dotenv import load_dotenv

load_dotenv()

def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root / "src"))


_add_src_to_path()

from config import Config  # noqa: E402
from model import ClozeTest, ReadingTask  # noqa: E402
from services.quality_reviewer import (  # noqa: E402
    dump_report_json,
    issues_to_dict,
    llm_judge_cloze,
    llm_judge_reading,
    now_tag,
    overall_score,
    rule_check_cloze,
    rule_check_reading,
)
from utils import LLMService  # noqa: E402


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


T = TypeVar("T")


def _unwrap_list_container(raw: Any) -> Any:
    if isinstance(raw, dict):
        for key in ("items", "data", "cloze", "clozes", "reading", "readings", "tasks"):
            v = raw.get(key)
            if isinstance(v, list):
                return v
    return raw


def _load_models_from_file(path: Path, model_cls: Type[T]) -> List[T]:
    raw = _unwrap_list_container(load_json(path))
    if isinstance(raw, list):
        return [model_cls(**x) for x in raw]
    if isinstance(raw, dict):
        return [model_cls(**raw)]
    raise ValueError(f"Unsupported JSON top-level in {path}: {type(raw).__name__}")


def iter_json_files(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.rglob("*.json") if p.is_file()])


def review_cloze_files(files: List[Path], llm: Optional[LLMService]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    per_item: List[Dict[str, Any]] = []
    totals = {"count": 0, "fatal": 0, "warning": 0, "info": 0, "avg_score": 0.0}
    scores: List[float] = []

    for path in files:
        items = _load_models_from_file(path, ClozeTest)
        for idx, item in enumerate(items):
            issues, stats = rule_check_cloze(item)

            llm_judge = None
            if llm is not None:
                llm_judge = llm_judge_cloze(item, llm).model_dump()

            score = overall_score(issues, llm_judge.get("overall_score") if llm_judge else None)
            scores.append(score)

            per_item.append(
                {
                    "type": "cloze",
                    "path": f"{path}#{idx}",
                    "source_file": str(path),
                    "index_in_file": idx,
                    "id": item.id,
                    "score": score,
                    "rule_issues": issues_to_dict(issues),
                    "rule_stats": stats,
                    "llm_judge": llm_judge,
                }
            )

            totals["count"] += 1
            totals["fatal"] += stats.get("fatal_count", 0)
            totals["warning"] += stats.get("warning_count", 0)
            totals["info"] += stats.get("info_count", 0)

    totals["avg_score"] = (sum(scores) / len(scores)) if scores else 0.0
    return per_item, totals


def review_reading_files(files: List[Path], llm: Optional[LLMService]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    per_item: List[Dict[str, Any]] = []
    totals = {"count": 0, "fatal": 0, "warning": 0, "info": 0, "avg_score": 0.0}
    scores: List[float] = []

    for path in files:
        tasks = _load_models_from_file(path, ReadingTask)
        for idx, task in enumerate(tasks):
            issues, stats = rule_check_reading(task)

            llm_judge = None
            if llm is not None:
                llm_judge = llm_judge_reading(task, llm).model_dump()

            score = overall_score(issues, llm_judge.get("overall_score") if llm_judge else None)
            scores.append(score)

            per_item.append(
                {
                    "type": "reading",
                    "path": f"{path}#{idx}",
                    "source_file": str(path),
                    "index_in_file": idx,
                    "title": task.title,
                    "score": score,
                    "rule_issues": issues_to_dict(issues),
                    "rule_stats": stats,
                    "llm_judge": llm_judge,
                }
            )

            totals["count"] += 1
            totals["fatal"] += stats.get("fatal_count", 0)
            totals["warning"] += stats.get("warning_count", 0)
            totals["info"] += stats.get("info_count", 0)

    totals["avg_score"] = (sum(scores) / len(scores)) if scores else 0.0
    return per_item, totals


def main() -> int:
    parser = argparse.ArgumentParser(description="Review AI-generated cloze + reading quality (考研英语).")
    parser.add_argument("--cloze-dir", default="data/clozetest", help="Directory of ClozeTest json files.")
    parser.add_argument("--reading-dir", default="data/readintask", help="Directory of ReadingTask json files.")
    parser.add_argument("--out", default=None, help="Output report path (json). Default: outputs/quality_report_<ts>.json")
    parser.add_argument("--llm", action="store_true", help="Enable LLM judging (requires API config).")
    args = parser.parse_args()

    config = Config()
    llm = LLMService(config) if args.llm else None

    cloze_files = iter_json_files(Path(args.cloze_dir))
    reading_files = iter_json_files(Path(args.reading_dir))

    cloze_items, cloze_totals = review_cloze_files(cloze_files, llm)
    reading_items, reading_totals = review_reading_files(reading_files, llm)

    report = {
        "generated_at": now_tag(),
        "settings": {
            "target": "考研英语",
            "language": "English",
            "llm_enabled": bool(args.llm),
            "model_provider": getattr(config, "model_provider", None) if args.llm else None,
            "model_version": getattr(config, "model_version", None) if args.llm else None,
        },
        "totals": {"cloze": cloze_totals, "reading": reading_totals},
        "items": cloze_items + reading_items,
    }

    out_path = Path(args.out) if args.out else Path("outputs") / f"quality_report_{report['generated_at']}.json"
    dump_report_json(out_path, report)
    print(f"Saved report: {out_path}")

    print(
        "Summary:",
        f"cloze_files={cloze_totals['count']} avg_score={cloze_totals['avg_score']:.1f} fatal={cloze_totals['fatal']} warning={cloze_totals['warning']} info={cloze_totals['info']}",
        f"reading_files={reading_totals['count']} avg_score={reading_totals['avg_score']:.1f} fatal={reading_totals['fatal']} warning={reading_totals['warning']} info={reading_totals['info']}",
        sep="\n- ",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
