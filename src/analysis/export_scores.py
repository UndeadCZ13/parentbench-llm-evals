# src/analysis/export_scores.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.judges.judge_prompts import RUBRIC_KEYS
from src.analysis.scoring_utils import aggregate_runs



def process_jsonl(in_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            raw_runs = rec.get("raw_judge_runs") or []
            agg = aggregate_runs(raw_runs)

            row: Dict[str, Any] = {
                "scenario_id": rec.get("scenario_id"),
                "scenario_text": rec.get("scenario_text"),
                "answer_model": rec.get("answer_model"),
                "answer_backend": rec.get("answer_backend"),
                "judge_model": rec.get("judge_model"),
                "judge_backend": rec.get("judge_backend"),
                "n_repeats": rec.get("n_repeats", len(raw_runs)),
                "comment": agg.comment,
            }

            # 均值
            for k in RUBRIC_KEYS:
                row[k] = agg.avg_scores.get(k)

            # 也可以顺手把 std 导出（可选）
            for k in RUBRIC_KEYS:
                row[f"{k}_std"] = agg.std_scores.get(k)

            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="run_judging 产生的 jsonl 文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 CSV 路径；默认放到 results/scores/ 下，名字跟输入类似",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    df = process_jsonl(in_path)

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path("results/scores")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (in_path.stem + ".csv")

    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ 导出完成：{out_path.resolve()}")


if __name__ == "__main__":
    main()
