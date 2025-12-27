import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from summarize_quality_report import summarize_report


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_reports(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.rglob("*.json") if p.is_file() and "quality_report_" in p.name])


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a summary for every quality_report_*.json in a directory.")
    parser.add_argument("--reports-dir", default="outputs/by_model", help="Directory containing per-model reports.")
    parser.add_argument("--out-dir", default=None, help="Directory to write summary json files. Default: <reports-dir>/summaries")
    parser.add_argument(
        "--combined-out",
        default=None,
        help="Optional path to write a combined summary list JSON (one entry per report).",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    report_files = _iter_reports(reports_dir)
    if not report_files:
        print(f"No reports found in: {reports_dir}")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else (reports_dir / "summaries")
    out_dir.mkdir(parents=True, exist_ok=True)

    combined: List[Dict[str, Any]] = []

    for report_path in report_files:
        report = _read_json(report_path)
        summary: Dict[str, Any] = summarize_report(report)

        model = str(report.get("model") or "unknown")
        ts = str(report.get("generated_at") or report_path.stem.replace("quality_report_", ""))
        out_path = out_dir / f"summary_{model}_{ts}.json"
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved summary: {out_path}")

        combined.append(
            {
                "report": str(report_path),
                "summary": str(out_path),
                "model": model,
                "generated_at": summary.get("generated_at"),
                "counts": summary.get("counts"),
                "overall": summary.get("overall"),
            }
        )

    if args.combined_out:
        combined_out = Path(args.combined_out)
        combined_out.parent.mkdir(parents=True, exist_ok=True)
        combined_out.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved combined index: {combined_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
