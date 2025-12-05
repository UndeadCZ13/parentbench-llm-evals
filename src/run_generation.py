# src/run_generation.py

from __future__ import annotations

import argparse
import datetime
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from model_caller_openai import call_model

# ===== 默认配置 =====

DEFAULT_SCENARIO_FILE = Path("data/scenarios/parentbench_v0.jsonl")

MODEL_OUT_DIR = Path("data/model_outputs")
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

# 你本地的 ollama 模型映射（按你 `ollama list` 填的）
OLLAMA_MODEL_REGISTRY: Dict[str, str] = {
    "glm_4_6": "glm-4.6:cloud",
    "qwen3_8b": "qwen3:8b",
    "deepseek_v3":"deepseek-v3.1:671b-cloud",
    "deepseek_r1": "deepseek-r1:latest",
    "gpt_oss": "gpt-oss:20b-cloud",
    "ministral3_8b":"ministral-3:8b-cloud",
    "ministral3_14b":"ministral-3:14b-cloud",
    "kimi-k2":"kimi-k2-thinking:cloud",
    "minimax_m2":"minimax-m2:cloud",
}

DEFAULT_BACKEND = "ollama"          # 默认用 ollama
DEFAULT_OLLAMA_MODEL_KEY = "deepseek_v3"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

DEFAULT_SYSTEM_PROMPT = """You are an AI assistant specialized in parenting advice.
You must give safe, evidence-informed, empathetic, and practical suggestions.
If a question touches on medical issues, safety, or mental health,
encourage consulting appropriate professionals and avoid giving diagnoses.
"""


# ===== 小工具 =====

def sanitize_tag(s: str) -> str:
    return str(s).replace("/", "_").replace(":", "-").replace(" ", "_")


def load_scenarios(scenario_file: Path) -> List[Dict[str, Any]]:
    """读取 scenario jsonl，返回一组 dict。"""
    scenarios: List[Dict[str, Any]] = []
    with scenario_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] 跳过一行无法解析的 JSON: {e}")
                continue
            scenarios.append(rec)
    print(f"[INFO] Loaded {len(scenarios)} scenarios from {scenario_file}")
    return scenarios


def try_get_scenario_id(rec: Dict[str, Any]) -> str:
    """兼容不同字段名获取场景 id。"""
    return str(
        rec.get("scenario_id")
        or rec.get("id")
        or rec.get("sid")
        or rec.get("scenario")
        or ""
    )


def try_get_scenario_text(rec: Dict[str, Any]) -> str:
    """兼容不同字段名获取场景文本。"""
    return (
        rec.get("scenario_text")
        or rec.get("scenario")
        or rec.get("prompt")
        or rec.get("question")
        or ""
    )


def _parse_numeric_tail(s: str) -> int | None:
    """从 scenario_id 中抽取末尾数字，例如 'pb_v0_0003' -> 3。"""
    m = re.search(r"(\d+)$", str(s))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _filter_scenarios_by_id_and_range(
    scenarios: List[Dict[str, Any]],
    scenario_ids_arg: str | None,
    scenario_range_arg: str | None,
) -> List[Dict[str, Any]]:
    """根据 --scenario-ids / --scenario-range 过滤 scenario 列表。"""

    if not scenario_ids_arg and not scenario_range_arg:
        return scenarios

    id_full_set: set[str] = set()
    id_num_set: set[int] = set()

    # 解析 --scenario-ids
    if scenario_ids_arg:
        for part in scenario_ids_arg.split(","):
            part = part.strip()
            if not part:
                continue
            if part.isdigit():
                id_num_set.add(int(part))
            else:
                id_full_set.add(part)

    # 解析 --scenario-range（形如 "0-10", "1:20", "3,8"）
    range_start: int | None = None
    range_end: int | None = None
    if scenario_range_arg:
        s = scenario_range_arg.replace(":", "-").replace(",", "-")
        try:
            start_str, end_str = s.split("-", 1)
            range_start = int(start_str.strip())
            range_end = int(end_str.strip())
            if range_start > range_end:
                range_start, range_end = range_end, range_start
        except Exception:
            print(
                f"[WARN] 无法解析 --scenario-range='{scenario_range_arg}'，"
                "应为 'start-end' 形式，例如 0-10；已忽略该参数。"
            )
            range_start = range_end = None

    def should_keep(rec: Dict[str, Any]) -> bool:
        sid = try_get_scenario_id(rec)
        num = _parse_numeric_tail(sid)

        keep = True

        # 如果指定了 scenario_ids，则只保留这些
        if id_full_set or id_num_set:
            keep = False
            if sid in id_full_set:
                keep = True
            elif num is not None and num in id_num_set:
                keep = True

        # 在 ids 过滤基础上，再加 range 限制
        if keep and range_start is not None and range_end is not None:
            if num is None or not (range_start <= num <= range_end):
                keep = False

        return keep

    filtered = [rec for rec in scenarios if should_keep(rec)]
    print(
        f"[INFO] Filtered scenarios by id/range: {len(filtered)}/{len(scenarios)} kept."
    )
    return filtered


# ===== 主逻辑 =====

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        default=DEFAULT_BACKEND,
        choices=["openai", "ollama", "local"],
        help="生成时使用的 backend：openai / ollama / local(local==ollama)",
    )
    parser.add_argument(
        "--ollama-model-key",
        type=str,
        default=DEFAULT_OLLAMA_MODEL_KEY,
        help="当 backend=ollama/local 时使用的 Ollama 模型 key（在 OLLAMA_MODEL_REGISTRY 中定义）",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default=DEFAULT_OPENAI_MODEL,
        help="当 backend=openai 时使用的 OpenAI 模型名（默认 gpt-4o-mini）",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=str(DEFAULT_SCENARIO_FILE),
        help="scenario jsonl 文件路径（默认 data/scenarios/parentbench_v0.jsonl）",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=str,
        default=None,
        help="可选：从文件加载 system prompt；若不提供则使用内置 DEFAULT_SYSTEM_PROMPT",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 answer jsonl 文件路径；若不指定则自动命名到 data/model_outputs/ 下",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="可选：只生成前 N 条 scenario，方便调试",
    )
    # 新增：只生成指定编号 / 编号区间的 scenario
    parser.add_argument(
        "--scenario-ids",
        type=str,
        default=None,
        help=(
            "可选：只生成指定 scenario_id，逗号分隔。"
            "例如 'pb_v0_0003,pb_v0_0007' 或 '3,7'（按 scenario_id 末尾数字匹配）。"
        ),
    )
    parser.add_argument(
        "--scenario-range",
        type=str,
        default=None,
        help=(
            "可选：只生成编号在某个区间的 scenario，"
            "按 scenario_id 末尾数字解析。"
            "例如 '--scenario-range 0-10' 表示只生成编号 0~10 的场景。"
        ),
    )

    args = parser.parse_args()

    backend = args.backend.lower()
    if backend == "local":
        backend = "ollama"

    # ===== 解析模型名 =====
    if backend == "openai":
        model_name = args.openai_model
    elif backend == "ollama":
        if args.ollama_model_key not in OLLAMA_MODEL_REGISTRY:
            raise ValueError(
                f"Ollama 模型 key '{args.ollama_model_key}' 不在 OLLAMA_MODEL_REGISTRY 中。\n"
                f"当前可用：{list(OLLAMA_MODEL_REGISTRY.keys())}"
            )
        model_name = OLLAMA_MODEL_REGISTRY[args.ollama_model_key]
    else:
        raise ValueError("backend 必须为 'openai', 'ollama', or 'local'")

    print(f"[INFO] Using backend={backend}, model={model_name}")

    # ===== 读取 system prompt =====
    if args.system_prompt_file:
        sp_path = Path(args.system_prompt_file)
        system_prompt = sp_path.read_text(encoding="utf-8")
        print(f"[INFO] Loaded system prompt from {sp_path}")
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT
        print("[INFO] Using DEFAULT_SYSTEM_PROMPT")

    # ===== 读取 scenarios =====
    scenario_path = Path(args.scenarios)
    scenarios = load_scenarios(scenario_path)

    # 先按 id / 编号区间过滤（如果指定）
    scenarios = _filter_scenarios_by_id_and_range(
        scenarios,
        scenario_ids_arg=args.scenario_ids,
        scenario_range_arg=args.scenario_range,
    )

    # 再按 max_scenarios 截断
    if args.max_scenarios is not None:
        scenarios = scenarios[: args.max_scenarios]
        print(f"[INFO] Truncated to first {len(scenarios)} scenarios for this run.")

    # ===== 确定输出路径 =====
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tag_backend = sanitize_tag(backend)
        tag_model = sanitize_tag(model_name)
        out_name = f"{scenario_path.stem}_{tag_backend}_{tag_model}_{ts}.jsonl"
        out_path = MODEL_OUT_DIR / out_name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing model outputs to {out_path}")

    # ===== 主循环：对每个 scenario 生成回答 =====
    with out_path.open("w", encoding="utf-8") as f_out:
        for idx, sc in enumerate(scenarios, start=1):
            scenario_id = try_get_scenario_id(sc)
            scenario_text = try_get_scenario_text(sc)

            if not scenario_text:
                print(f"[WARN] scenario {idx} 缺少文本，跳过：{sc}")
                continue

            print(f"[{idx}/{len(scenarios)}] Generating for scenario_id={scenario_id or '?'}")

            # 统一用 scenario_text 作为 prompt
            answer_text = call_model(
                prompt=scenario_text,
                backend=backend,
                model=model_name,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=1024,
            )

            if answer_text is None:
                print(f"[WARN] backend/model 返回 None，写入空字符串。scenario_id={scenario_id}")
                answer_text = ""
            elif not isinstance(answer_text, str):
                answer_text = str(answer_text)

            record = {
                "scenario_id": scenario_id,
                "model": model_name,
                "backend": backend,
                "answer": answer_text,
                "generated_at": datetime.datetime.now().isoformat(),
            }

            # 如果你想把原始 scenario 文本也存进去，打开这行：
            # record["scenario_text"] = scenario_text

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print("  -> done.\n")

    print("All scenarios processed.")
    print(f"Final output file: {out_path}")


if __name__ == "__main__":
    main()
