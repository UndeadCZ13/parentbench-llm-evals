#!/usr/bin/env python3
# scripts/retry_bad_generations.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.model_caller import call_model
from src.run_generation import (
    build_final_only_prompt,
    strip_reasoning,
    suspected_truncation,
    is_reasoning_model,
)
from src.config import (
    MAX_TOKENS_DEFAULT,
    MAX_TOKENS_REASONING_MODEL,
    RETRY_TOKEN_MULTIPLIER,
    STRIP_REASONING,
)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
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


def is_empty_text(x: Any) -> bool:
    if x is None:
        return True
    if not isinstance(x, str):
        x = str(x)
    return len(x.strip()) == 0


def get_uid(rec: Dict[str, Any]) -> str:
    return pick_str(rec, ["scenario_uid", "scenario_id", "id"], default="")


def get_lang(rec: Dict[str, Any]) -> str:
    # prefer explicit language in record; fallback from filename handled outside
    return pick_str(rec, ["language", "lang"], default="en").lower()


def load_scenarios_map(scenarios_path: Path) -> Dict[str, Dict[str, Any]]:
    mp: Dict[str, Dict[str, Any]] = {}
    for s in iter_jsonl(scenarios_path):
        uid = get_uid(s)
        if uid:
            mp[uid] = s
    return mp


def should_retry(rec: Dict[str, Any], mode: str) -> bool:
    ans = rec.get("answer")
    empty = is_empty_text(ans)
    trunc = pick_bool(rec, ["suspected_truncation"], default=False)

    if mode == "empty":
        return empty
    if mode == "trunc":
        return trunc
    # both (default)
    return bool(empty or trunc)



def regenerate_one(
    scenario_rec: Dict[str, Any],
    backend: str,
    model_name: str,
    attempts: int = 15,
    max_retries_per_call: int = 15,
) -> Dict[str, Any]:
    """
    Return fields to update: answer_raw/answer/... + metadata about retries.
    We try up to `attempts` attempts until non-empty and not trunc.
    """
    lang = get_lang(scenario_rec)
    user_prompt = pick_str(scenario_rec, ["scenario_text", "prompt", "question", "scenario"], default="")
    if not user_prompt:
        return {
            "answer_raw": "",
            "answer": "",
            "suspected_truncation": True,
            "api_error": "scenario_text missing",
            "retry_attempts": 0,
            "retry_success": False,
        }

    reasoning = is_reasoning_model(model_name)
    base_max_tokens = MAX_TOKENS_REASONING_MODEL if reasoning else MAX_TOKENS_DEFAULT

    # Allow token growth if trunc happens.
    max_tokens = base_max_tokens

    for a in range(1, attempts + 1):
                # build_final_only_prompt 兼容旧/新签名：
        # - old: build_final_only_prompt(user_prompt, lang)
        # - new: build_final_only_prompt(user_prompt, lang, model_name)
        try:
            prompt = build_final_only_prompt(user_prompt, lang, model_name)  # new
        except TypeError:
            prompt = build_final_only_prompt(user_prompt, lang)  # old


        resp = call_model(
            prompt=prompt,
            backend=backend,
            model=model_name,
            max_tokens=max_tokens,
            max_retries=max_retries_per_call,  # <- 每次调用内部也给足 retry
        )

        raw = resp.get("text", "") or ""
        finish_reason = resp.get("finish_reason")
        api_error = resp.get("error")

        cleaned = strip_reasoning(raw) if STRIP_REASONING else {
            "final": raw,
            "stripped": False,
            "removed_chars": 0,
        }
        final = cleaned["final"]
        trunc = suspected_truncation(final, finish_reason)
        empty = is_empty_text(final)

        # success condition: non-empty + not trunc
        if (not empty) and (not trunc):
            return {
                "answer_raw": raw,
                "answer": final,
                "finish_reason": finish_reason,
                "api_error": api_error,
                "reasoning_stripped": cleaned.get("stripped", False),
                "removed_reasoning_chars": cleaned.get("removed_chars", 0),
                "suspected_truncation": trunc,
                "retry_attempts": a,
                "retry_success": True,
                "retry_final_max_tokens": max_tokens,
            }

        # if trunc -> increase max_tokens for next attempt
        if trunc:
            max_tokens = int(max_tokens * RETRY_TOKEN_MULTIPLIER)

    # failed after all attempts
    return {
        "answer_raw": raw if "raw" in locals() else "",
        "answer": final if "final" in locals() else "",
        "finish_reason": finish_reason if "finish_reason" in locals() else None,
        "api_error": api_error if "api_error" in locals() else "retry_exhausted",
        "reasoning_stripped": cleaned.get("stripped", False) if "cleaned" in locals() else False,
        "removed_reasoning_chars": cleaned.get("removed_chars", 0) if "cleaned" in locals() else 0,
        "suspected_truncation": True,
        "retry_attempts": attempts,
        "retry_success": False,
        "retry_final_max_tokens": max_tokens,
    }


def process_file(
    in_path: Path,
    scenarios_map: Dict[str, Dict[str, Any]],
    attempts: int,
    max_retries_per_call: int,
    out_path: Path,
    dry_run: bool = False,
    mode: str = "both",
) -> Tuple[int, int, int]:
    """
    Only write an output file if there exists any record that needs retry (per mode).
    Returns: (n_total, n_bad, n_fixed)
    """
    records = list(iter_jsonl(in_path))
    n_total = len(records)

    # pre-scan: count bad records under the chosen mode
    bad_indices: List[int] = []
    for i, rec in enumerate(records):
        uid = get_uid(rec)
        if not uid:
            continue
        if should_retry(rec, mode):
            bad_indices.append(i)

    n_bad = len(bad_indices)
    if n_bad == 0:
        print(f"[SKIP] {in_path.name}: no records to retry (mode={mode}). No output file generated.")
        return n_total, 0, 0

    n_fixed = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx, rec in enumerate(records):
            uid = get_uid(rec)
            if not uid or idx not in set(bad_indices):
                # untouched record
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            backend = pick_str(rec, ["backend", "answer_backend"], default="").lower()
            if backend == "local":
                backend = "ollama"
            model_name = pick_str(rec, ["model", "answer_model"], default="")

            scenario_rec = scenarios_map.get(uid)
            if scenario_rec is None:
                scenario_rec = rec  # fallback

            print(f"[RETRY] file={in_path.name} uid={uid} backend={backend} model={model_name}")

            if dry_run:
                upd = {"retry_attempts": 0, "retry_success": False, "api_error": "DRY_RUN"}
            else:
                upd = regenerate_one(
                    scenario_rec=scenario_rec,
                    backend=backend,
                    model_name=model_name,
                    attempts=attempts,
                    max_retries_per_call=max_retries_per_call,
                )

            if upd.get("retry_success"):
                n_fixed += 1

            new_rec = {**rec, **upd, "retried": True, "retry_mode": mode}
            f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")

    return n_total, n_bad, n_fixed


def main() -> None:
    ap = argparse.ArgumentParser(description="Retry empty/trunc generations in model_outputs using scenario_uid.")
    ap.add_argument("--scenarios", required=True, help="Scenario jsonl file (e.g. data/scenarios/views/en/parentbench_v0_en.jsonl)")
    ap.add_argument("--inputs", default="data/model_outputs", help="Input file or directory (default: data/model_outputs)")
    ap.add_argument("--pattern", default="*.jsonl", help="Glob pattern if inputs is a directory (default: *.jsonl)")
    ap.add_argument("--attempts", type=int, default=15, help="Max attempts per bad scenario (default: 15)")
    ap.add_argument("--max-retries-per-call", type=int, default=15, help="Internal retry per API call (default: 15)")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: alongside input files)")
    ap.add_argument("--suffix", default="_retried", help="Suffix for output filename (default: _retried)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retry-mode",choices=["empty", "trunc", "both"],default="both",help="Which records to retry: empty only, trunc only, or both (default: both)",)

    args = ap.parse_args()

    scenarios_path = Path(args.scenarios)
    if not scenarios_path.exists():
        raise FileNotFoundError(scenarios_path)

    scenarios_map = load_scenarios_map(scenarios_path)
    print(f"[INFO] Loaded scenarios: {len(scenarios_map)} from {scenarios_path}")

    inp = Path(args.inputs)
    files: List[Path] = []
    if inp.is_file():
        files = [inp]
    else:
        files = sorted(inp.glob(args.pattern))
    if not files:
        print(f"[WARN] No input files matched: {inp} pattern={args.pattern}")
        return

    total_bad = 0
    total_fixed = 0
    total_records = 0

    for fp in files:
        out_dir = Path(args.out_dir) if args.out_dir else fp.parent
        out_path = out_dir / f"{fp.stem}{args.suffix}.jsonl"

        n_total, n_bad, n_fixed = process_file(
            in_path=fp,
            scenarios_map=scenarios_map,
            attempts=args.attempts,
            max_retries_per_call=args.max_retries_per_call,
            out_path=out_path,
            dry_run=args.dry_run,
            mode=args.retry_mode,
        )

        total_records += n_total
        total_bad += n_bad
        total_fixed += n_fixed

        print(f"[DONE] {fp.name}: total={n_total} bad={n_bad} fixed={n_fixed} -> {out_path}")

    print("============================================================")
    print(f"[SUMMARY] files={len(files)} total_records={total_records} bad={total_bad} fixed={total_fixed}")
    if total_bad > 0:
        print(f"[SUMMARY] fix_rate={total_fixed/total_bad:.2%}")


if __name__ == "__main__":
    main()
