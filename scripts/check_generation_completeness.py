#!/usr/bin/env python3
# scripts/check_generation_completeness.py

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                # skip bad line
                continue


def pick_str(rec: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for k in keys:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return default


def pick_bool(rec: Dict[str, Any], keys: List[str], default: bool = False) -> bool:
    for k in keys:
        v = rec.get(k)
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)) and v in (0, 1):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "t", "1", "yes", "y"):
                return True
            if s in ("false", "f", "0", "no", "n"):
                return False
    return default


def is_empty_text(s: Any) -> bool:
    if s is None:
        return True
    if not isinstance(s, str):
        s = str(s)
    return len(s.strip()) == 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scan model_outputs jsonl files and summarize empty/trunc counts per generation model."
    )
    ap.add_argument(
        "--dir",
        default="data/model_outputs",
        help="Directory containing generation jsonl files (default: data/model_outputs)",
    )
    ap.add_argument(
        "--pattern",
        default="*.jsonl",
        help="Glob pattern under --dir (default: *.jsonl)",
    )
    ap.add_argument(
        "--out",
        default="results/analysis/generation_completeness.csv",
        help="Output CSV path (default: results/analysis/generation_completeness.csv)",
    )
    ap.add_argument(
        "--min-total",
        type=int,
        default=1,
        help="Only show models with at least N items (default: 1)",
    )
    args = ap.parse_args()

    in_dir = Path(args.dir)
    files = sorted(in_dir.rglob(args.pattern)) if "**" in args.pattern else sorted(in_dir.glob(args.pattern))
    if not files:
        print(f"[WARN] No files matched: dir={in_dir} pattern={args.pattern}")
        return

    rows: List[Dict[str, Any]] = []

    for fp in files:
        # infer language from filename if possible (best-effort)
        fname = fp.name.lower()
        inferred_lang = "zh" if "_zh" in fname or fname.startswith("zh_") else ("en" if "_en" in fname or fname.startswith("en_") else "")

        for rec in iter_jsonl(fp):
            model = pick_str(rec, ["model", "answer_model"], default="(unknown_model)")
            backend = pick_str(rec, ["backend", "answer_backend"], default="(unknown_backend)")
            lang = pick_str(rec, ["language", "lang"], default=inferred_lang or "(unknown_lang)")

            ans = rec.get("answer")
            ans_raw = rec.get("answer_raw")

            empty = is_empty_text(ans)
            trunc = pick_bool(rec, ["suspected_truncation", "trunc", "is_trunc"], default=False)

            api_error = rec.get("api_error")
            has_api_error = not is_empty_text(api_error)

            rows.append(
                {
                    "file": fp.name,
                    "language": lang,
                    "backend": backend,
                    "model": model,
                    "empty": empty,
                    "trunc": trunc,
                    "empty_and_trunc": bool(empty and trunc),
                    "api_error": has_api_error,
                    "answer_len": 0 if ans is None else len(str(ans)),
                    "answer_raw_len": 0 if ans_raw is None else len(str(ans_raw)),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        print("[WARN] No records parsed from matched files.")
        return

    # Aggregate per model (and backend, language)
    gcols = ["backend", "model"]
    # if you also want by language, keep it here; otherwise remove it.
    gcols_lang = ["backend", "model", "language"]

    def _agg(group_cols: List[str]) -> pd.DataFrame:
        out = (
            df.groupby(group_cols, dropna=False)
            .agg(
                n_total=("empty", "size"),
                n_empty=("empty", "sum"),
                n_trunc=("trunc", "sum"),
                n_empty_and_trunc=("empty_and_trunc", "sum"),
                n_api_error=("api_error", "sum"),
                avg_answer_len=("answer_len", "mean"),
                p50_answer_len=("answer_len", "median"),
            )
            .reset_index()
        )
        out["empty_rate"] = (out["n_empty"] / out["n_total"]).round(4)
        out["trunc_rate"] = (out["n_trunc"] / out["n_total"]).round(4)
        out["api_error_rate"] = (out["n_api_error"] / out["n_total"]).round(4)
        return out

    summary = _agg(gcols_lang)  # 按语言分开统计（更符合你现在的诊断需求）
    summary = summary[summary["n_total"] >= int(args.min_total)].copy()

    # Sort: prioritize problems
    summary = summary.sort_values(
        by=["trunc_rate", "empty_rate", "api_error_rate", "n_total"],
        ascending=[False, False, False, False],
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    # Print top rows
    print(f"[INFO] Scanned files: {len(files)} | records: {len(df)}")
    print(f"[DONE] Wrote summary CSV: {out_path}\n")

    show_cols = [
        "backend",
        "model",
        "language",
        "n_total",
        "n_empty",
        "empty_rate",
        "n_trunc",
        "trunc_rate",
        "n_empty_and_trunc",
        "n_api_error",
        "api_error_rate",
        "avg_answer_len",
        "p50_answer_len",
    ]
    # display top 30
    with pd.option_context("display.max_rows", 60, "display.max_columns", 50, "display.width", 140):
        print(summary[show_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
