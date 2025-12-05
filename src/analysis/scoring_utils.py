# src/analysis/scoring_utils.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Iterable, Optional
import statistics

from src.judges.judge_prompts import RUBRIC_KEYS


@dataclass
class ScoreAggregate:
    """
    聚合后的评分结果（对多次 judge run 做统计）
    """
    avg_scores: Dict[str, Optional[float]]
    std_scores: Dict[str, Optional[float]]
    comment: Optional[str]


def aggregate_runs(
    raw_runs: List[Dict[str, Any]],
    rubric_keys: Iterable[str] = RUBRIC_KEYS,
) -> ScoreAggregate:
    """
    对多次 judge 结果做统计聚合：
      - 每个 rubric 计算 mean / std（如果没有有效值则为 None）
      - comment 选取第一条非空 comment
    """
    avg_scores: Dict[str, Optional[float]] = {}
    std_scores: Dict[str, Optional[float]] = {}

    for key in rubric_keys:
        vals: List[float] = []
        for run in raw_runs:
            v = run.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))

        if vals:
            avg_scores[key] = sum(vals) / len(vals)
            std_scores[key] = (
                statistics.pstdev(vals) if len(vals) > 1 else 0.0
            )
        else:
            avg_scores[key] = None
            std_scores[key] = None

    # 简单策略：拿第一条非空 comment
    comment: Optional[str] = None
    for run in raw_runs:
        c = run.get("comment")
        if isinstance(c, str) and c.strip():
            comment = c.strip()
            break

    return ScoreAggregate(
        avg_scores=avg_scores,
        std_scores=std_scores,
        comment=comment,
    )
