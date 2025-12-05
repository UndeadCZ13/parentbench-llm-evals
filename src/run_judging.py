# src/run_judging.py

from __future__ import annotations

import argparse
import datetime
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable

from model_caller_openai import call_model
from judges.judge_prompts import build_judge_prompt, RUBRIC_KEYS

# =================== 默认配置 ===================

DEFAULT_BACKEND = "openai"          # 也可以设为 "ollama"
DEFAULT_OLLAMA_MODEL_KEY = "qwen3_8b"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

DEFAULT_SCENARIO_FILE = Path("data/scenarios/parentbench_v0.jsonl")
DEFAULT_ANSWERS = "data/model_outputs/parentbench_v0_ollama_kimi-k2_20251205-162446.jsonl"
DEFAULT_REPEAT = 3

JUDGE_OUT_DIR = Path("data/judge_outputs")
JUDGE_OUT_DIR.mkdir(parents=True, exist_ok=True)


# =================== 工具函数 ===================

def load_answers(answer_file: Path) -> List[Dict[str, Any]]:
    """读取模型生成的回答记录（jsonl）。"""
    records: List[Dict[str, Any]] = []
    with answer_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] 跳过一行无法解析的 JSON: {e}")
                continue
            records.append(rec)
    print(f"[INFO] Loaded {len(records)} answer records from {answer_file}")
    return records


def try_get_scenario_id(rec: Dict[str, Any]) -> str:
    """兼容不同字段名，获取 scenario id。"""
    return str(
        rec.get("scenario_id")
        or rec.get("id")
        or rec.get("scenario")
        or rec.get("sid")
        or ""
    )


def try_get_scenario_text(rec: Dict[str, Any]) -> str:
    """兼容不同字段名，取出场景文本。"""
    return (
        rec.get("scenario_text")
        or rec.get("scenario")
        or rec.get("prompt")
        or rec.get("question")
        or ""
    )


def try_get_answer_text(rec: Dict[str, Any]) -> str:
    """兼容不同字段名，取出模型回答文本。"""
    return rec.get("answer") or rec.get("response") or rec.get("model_response") or ""


def sanitize_tag(s: str) -> str:
    return s.replace("/", "_").replace(":", "-").replace(" ", "_")


def safe_parse_json(text: Any) -> Dict[str, Any]:
    """从 LLM 输出中尽量解析出一个 JSON 对象。

    支持几种常见格式：
    - 直接就是 JSON
    - ```json ...``` 代码块
    - ``` ...``` 代码块
    - 前后有说明文字，中间有一段 {...}
    解析失败会抛出 ValueError。
    """
    if text is None:
        raise ValueError("safe_parse_json: 模型输出为 None")

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    if not text:
        raise ValueError("safe_parse_json: 模型输出为空字符串")

    # 1) 直接尝试整体解析
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) 提取 ```json ...``` 或 ``` ...``` 代码块
    fence_pattern = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    m = fence_pattern.search(text)
    if m:
        inner = m.group(1).strip()
        if inner:
            try:
                return json.loads(inner)
            except Exception:
                pass  # 再往下尝试

    # 3) 在整个文本中找到第一个 { 到 最后一个 } 之间的片段
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        snippet = text[first_brace : last_brace + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass

    # 4) 仍然失败
    raise ValueError("safe_parse_json: 未能在输出中找到有效 JSON")


def load_scenario_map(scenario_file: Path) -> Dict[str, str]:
    """从 scenario jsonl 构建 {scenario_id -> scenario_text} 映射。"""
    mapping: Dict[str, str] = {}
    if not scenario_file.exists():
        print(f"[WARN] scenario file not found: {scenario_file}")
        return mapping

    with scenario_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            sid = try_get_scenario_id(rec)
            prompt = try_get_scenario_text(rec)
            if sid and prompt:
                mapping[str(sid)] = str(prompt)

    print(f"[INFO] Loaded {len(mapping)} scenarios from {scenario_file}")
    return mapping


def aggregate_runs(
    raw_runs: List[Dict[str, Any]],
    rubric_keys: Iterable[str] = RUBRIC_KEYS,
) -> Dict[str, Any]:
    """对多次 judge 运行结果做聚合，返回:
       - 每个 rubric 的平均分 (key 同名)
       - 每个 rubric 的标准差 (key + '_std')
       - 选取一条代表性的 comment
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
            avg = sum(vals) / len(vals)
            # 简单标准差（总体）
            if len(vals) > 1:
                mean = avg
                var = sum((x - mean) ** 2 for x in vals) / len(vals)
                std = var ** 0.5
            else:
                std = 0.0
            avg_scores[key] = avg
            std_scores[key] = std
        else:
            avg_scores[key] = None
            std_scores[key] = None

    # comment：简单策略，取第一条非空 comment
    comment: Optional[str] = None
    for run in raw_runs:
        c = run.get("comment")
        if isinstance(c, str) and c.strip():
            comment = c.strip()
            break

    merged: Dict[str, Any] = {}
    merged.update(avg_scores)
    merged.update({f"{k}_std": v for k, v in std_scores.items()})
    merged["comment"] = comment
    return merged


# =================== 主逻辑 ===================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--answers",
        type=str,
        default=DEFAULT_ANSWERS,
        help="模型生成的回答记录文件（jsonl）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="评审输出保存路径（jsonl）。默认为 data/judge_outputs/ 下自动命名",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=DEFAULT_BACKEND,
        choices=["openai", "ollama", "local"],
        help="Judge 使用的 backend：openai / ollama / local (==ollama)",
    )
    parser.add_argument(
        "--ollama-model-key",
        type=str,
        default=DEFAULT_OLLAMA_MODEL_KEY,
        help="当 backend=ollama/local 时使用的 Ollama 模型 key（在 run_generation.OLLAMA_MODEL_REGISTRY 中定义）",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default=DEFAULT_OPENAI_MODEL,
        help="当 backend=openai 时使用的 OpenAI 模型名（默认 gpt-4o-mini）",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=DEFAULT_REPEAT,
        help="同一 (answer, judge) 组合重复评分次数，通常为 1 / 3 / 5（默认：3）",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=str(DEFAULT_SCENARIO_FILE),
        help="原始 scenario 文件（jsonl），用于兜底补充场景文本（默认 data/scenarios/parentbench_v0.jsonl）",
    )
    args = parser.parse_args()

    backend = args.backend.lower()
    if backend == "local":
        backend = "ollama"

    n_repeats = max(1, int(args.n_repeats))

    # ===== 解析 judge 模型名称 =====
    if backend == "ollama":
        # 复用 run_generation 中的 Ollama 模型 registry（如果存在）
        try:
            from run_generation import OLLAMA_MODEL_REGISTRY  # type: ignore
        except Exception:
            OLLAMA_MODEL_REGISTRY = {}  # fallback

        if args.ollama_model_key not in OLLAMA_MODEL_REGISTRY:
            raise ValueError(
                f"Ollama 模型 key '{args.ollama_model_key}' 不在 OLLAMA_MODEL_REGISTRY 中（或未定义）。\n"
                f"请在 run_generation.OLLAMA_MODEL_REGISTRY 中添加，或者修改 --ollama-model-key。当前可用：{list(OLLAMA_MODEL_REGISTRY.keys())}"
            )
        judge_model_name = OLLAMA_MODEL_REGISTRY[args.ollama_model_key]
    elif backend == "openai":
        judge_model_name = args.openai_model
    else:
        raise ValueError("backend 必须为 'openai', 'ollama', or 'local'")

    print(f"[INFO] Using judge backend={backend}, model={judge_model_name}, n_repeats={n_repeats}")

    answers_path = Path(args.answers)
    answer_records = load_answers(answers_path)

    # 载入 scenario 映射（用于补全场景文本）
    scenario_map = load_scenario_map(Path(args.scenarios))

    # 确定输出路径
    if args.output:
        out_path = Path(args.output)
    else:
        tag_backend = sanitize_tag(backend)
        tag_model = sanitize_tag(judge_model_name)
        out_name = f"{answers_path.stem}_judged_{tag_backend}_{tag_model}.jsonl"
        out_path = JUDGE_OUT_DIR / out_name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing judge outputs to {out_path}")

    # ========= 主循环：对每条 answer 进行多次评审 =========
    with out_path.open("w", encoding="utf-8") as f_out:
        for idx, rec in enumerate(answer_records, start=1):
            sid_str = try_get_scenario_id(rec)
            scenario_prompt = try_get_scenario_text(rec)
            if (not scenario_prompt) and sid_str:
                scenario_prompt = scenario_map.get(sid_str, "")

            if not scenario_prompt:
                print(f"[WARN] record {idx} 缺少 scenario_text，且在 scenario 文件中也未找到，跳过。record={rec}")
                continue

            model_response = try_get_answer_text(rec)
            if not model_response:
                print(f"[WARN] record {idx} 缺少 answer 文本，跳过。record={rec}")
                continue

            # 答案模型信息（来自生成阶段）
            answer_model_name = rec.get("model") or rec.get("answer_model")
            answer_backend = rec.get("backend") or rec.get("answer_backend")

            print(f"[{idx}/{len(answer_records)}] judging scenario {sid_str or '?'}")

            raw_runs: List[Dict[str, Any]] = []

            # 多次评审
            for r in range(n_repeats):
                judge_prompt = build_judge_prompt(
                    scenario_prompt=scenario_prompt,
                    model_response=model_response,
                    scenario_id=sid_str,
                )

                judge_output_text = call_model(
                    prompt=judge_prompt,
                    backend=backend,
                    model=judge_model_name,
                    temperature=0.0,   # 为了稳定性，建议 0
                    max_tokens=1024,
                )

                if judge_output_text is None:
                    judge_output_text = ""
                else:
                    judge_output_text = str(judge_output_text)

                try:
                    judge_json = safe_parse_json(judge_output_text)
                    raw_runs.append(judge_json)
                except ValueError as e:
                    print(f"[WARN] safe_parse_json 失败 (scenario_id={sid_str}, repeat={r+1}): {e}")
                    raw_runs.append({
                        "parse_error": str(e),
                        "raw_text": judge_output_text,
                    })

            # 聚合本条 answer 的多次评审结果
            merged_scores = aggregate_runs(raw_runs, RUBRIC_KEYS)

            record_out: Dict[str, Any] = {
                "scenario_id": sid_str,
                "scenario_text": scenario_prompt,
                "judge_backend": backend,
                "judge_model": judge_model_name,
                "n_repeats": n_repeats,
                "answer_model": answer_model_name,
                "answer_backend": answer_backend,
                "answer_text": model_response,
                # 展平聚合后的分数与 comment
                **{k: v for k, v in merged_scores.items() if k != "comment"},
                "comment": merged_scores.get("comment"),
                # 保留所有原始评分记录，方便未来做稳定性分析
                "raw_judge_runs": raw_runs,
                "judged_at": datetime.datetime.now().isoformat(),
            }

            f_out.write(json.dumps(record_out, ensure_ascii=False) + "\n")

    print("Judging completed.")
    print(f"→ Saved to {out_path}")


if __name__ == "__main__":
    main()
