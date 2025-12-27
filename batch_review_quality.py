import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

load_dotenv()


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root / "src"))


_add_src_to_path()

from config import Config  # noqa: E402
from services.quality_reviewer import dump_report_json, now_tag  # noqa: E402
from utils import LLMService  # noqa: E402

from review_quality import iter_json_files, review_cloze_files, review_reading_files  # noqa: E402


def _infer_model_name(path: Path) -> str:
    """
    Infer model name from filename like:
    - DeepseekV3.2_ClozeTest.json -> DeepseekV3.2
    - GPT5.2_ReadingComprehension.json -> GPT5.2
    - official__ClozeTest.json -> official
    """
    stem = path.stem  # without .json
    if "_" in stem:
        return stem.split("_", 1)[0].strip() or "unknown"
    return stem.strip() or "unknown"


def _group_by_model(files: List[Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = defaultdict(list)
    for p in files:
        groups[_infer_model_name(p)].append(p)
    return dict(groups)


def _all_models(*group_dicts: Dict[str, List[Path]]) -> List[str]:
    models: Set[str] = set()
    for g in group_dicts:
        models.update(g.keys())
    return sorted(models)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch review all models' json files under data/, producing one report per model.")
    parser.add_argument("--cloze-dir", default="data/clozetest/1", help="Directory containing ClozeTest json files.")
    parser.add_argument("--reading-dir", default="data/readintask/1", help="Directory containing ReadingTask json files.")
    parser.add_argument("--out-dir", default="outputs/by_model", help="Directory to write per-model reports.")
    parser.add_argument("--llm", action="store_true", help="Enable LLM judging (requires API config).")
    args = parser.parse_args()

    config = Config()
    llm = LLMService(config) if args.llm else None
    ts = now_tag()

    cloze_files = iter_json_files(Path(args.cloze_dir))
    reading_files = iter_json_files(Path(args.reading_dir))

    cloze_by_model = _group_by_model(cloze_files)
    reading_by_model = _group_by_model(reading_files)

    models = _all_models(cloze_by_model, reading_by_model)
    if not models:
        print("No json files found.")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        model_cloze = cloze_by_model.get(model, [])
        model_reading = reading_by_model.get(model, [])

        cloze_items, cloze_totals = review_cloze_files(model_cloze, llm)
        reading_items, reading_totals = review_reading_files(model_reading, llm)

        report = {
            "generated_at": ts,
            "model": model,
            "settings": {
                "target": "考研英语",
                "language": "English",
                "llm_enabled": bool(args.llm),
                "model_provider": getattr(config, "model_provider", None) if args.llm else None,
                "judge_model_version": getattr(config, "model_version", None) if args.llm else None,
            },
            "totals": {"cloze": cloze_totals, "reading": reading_totals},
            "items": cloze_items + reading_items,
        }

        out_path = out_dir / f"quality_report_{model}_{ts}.json"
        dump_report_json(out_path, report)
        print(f"Saved report ({model}): {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

