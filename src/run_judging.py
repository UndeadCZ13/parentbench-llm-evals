# src/run_judging.py

from pathlib import Path
import json
import csv
import datetime

from model_caller_openai import call_model
from judges.judge_prompts import build_judge_prompt, RUBRIC_KEYS
import time


# ===== 需要你手动指定：要评哪个模型输出 =====
# 举例：你之前 Groq 批量生成的输出文件路径，可以复制真实文件名过来
INPUT_MODEL_OUTPUT_FILE = Path(
    "data/model_outputs/parentbench_v0_groq_qwen-qwen3-32b_20251116-165745.jsonl"
    # ↑ 这里请改成你自己 data/model_outputs 目录里实际存在的那个文件名
)

# ===== 评审结果输出到哪 =====
#OUTPUT_SCORES_CSV = Path("results/scores/parentbench_v0_judged_by_qwen.csv")
JUDGE_BACKEND = "local"  # 或 "groq"

if JUDGE_BACKEND == "groq":
    JUDGE_MODEL = "qwen/qwen3-32b"
elif JUDGE_BACKEND == "local":
    JUDGE_MODEL = "deepseek-r1"
else:
    raise ValueError(f"Unknown JUDGE_BACKEND: {JUDGE_BACKEND}")
# 根据 backend + judge 模型自动生成输出文件名
safe_model_name = (
    JUDGE_MODEL
    .replace("/", "-")
    .replace(":", "-")
    .replace(".", "-")
)

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

OUTPUT_SCORES_CSV = Path(
    f"results/scores/parentbench_v0_judged_{JUDGE_BACKEND}_{safe_model_name}_{timestamp}.csv"
)

# ===== 选择 judge 使用哪个 backend / 模型 =====
# 可以用 Groq (qwen/qwen3-32b) 或本地 DeepSeek
JUDGE_BACKEND = "groq"   # "groq" 或 "local"

if JUDGE_BACKEND == "groq":
    JUDGE_MODEL = "qwen/qwen3-32b"
elif JUDGE_BACKEND == "local":
    JUDGE_MODEL = "deepseek-r1"
else:
    raise ValueError(f"Unknown JUDGE_BACKEND: {JUDGE_BACKEND}")


def load_model_outputs(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model output JSONL not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def extract_json_from_text(raw_text: str):
    """
    从 LLM 的输出中提取第一个 JSON 对象。
    处理情况：
    - 前面有 <think>...</think> 或其他解释性文字
    - 只要里面有一段 { ... }，就抓第一对大括号的内容来解析
    """

    raw_text = raw_text.strip()
    if not raw_text:
        raise ValueError("Empty judge reply, cannot parse JSON.")

    # 找到第一个 '{' 和最后一个 '}'
    start = raw_text.find("{")
    end = raw_text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find a JSON object in judge reply: {raw_text[:200]}...")

    json_str = raw_text[start : end + 1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # 如果 JSON 里有一些奇怪的换行或空格，一般也不会影响，但这里再抛出去方便 debug
        raise ValueError(f"Failed to parse extracted JSON: {e}\nExtracted: {json_str[:200]}...")


def main():
    print(f"Loading model outputs from: {INPUT_MODEL_OUTPUT_FILE}")
    records = load_model_outputs(INPUT_MODEL_OUTPUT_FILE)
    print(f"Loaded {len(records)} model responses.\n")

    OUTPUT_SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)

    # 定义 CSV 列：场景 ID、被评模型信息 + 8 个 rubric 分数 + comment
    fieldnames = [
        "scenario_id",
        "backend_answer",
        "model_answer",
        "judge_backend",
        "judge_model",
        "generated_at",
    ] + RUBRIC_KEYS + ["comment"]

    with OUTPUT_SCORES_CSV.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for i, rec in enumerate(records, start=1):
            scenario_id = rec.get("scenario_id", f"sc_{i:04d}")
            answer_backend = rec.get("backend", "")
            answer_model = rec.get("model", "")
            scenario_prompt = rec["prompt"]
            model_response = rec["response"]

            print(f"[{i}/{len(records)}] Judging scenario {scenario_id} from model {answer_model}")

            judge_prompt = build_judge_prompt(
                scenario_prompt=scenario_prompt,
                model_response=model_response,
            )

            # 为了评审稳定性，temperature 建议设为 0
            judge_reply = call_model(
                judge_prompt,
                backend=JUDGE_BACKEND,
                model=JUDGE_MODEL,
                temperature=0.0,
            )

            # 期待 judge_reply 是一个 JSON 对象字符串
            judge_reply = call_model(
                judge_prompt,
                backend=JUDGE_BACKEND,
                model=JUDGE_MODEL,
                temperature=0.0,
            )

            judge_reply_stripped = judge_reply.strip()

            try:
                scores_obj = extract_json_from_text(judge_reply_stripped)
            except Exception as e:
                print(f"  JSON parse error for scenario {scenario_id}: {e}")
                print("  Raw judge reply:")
                print(judge_reply_stripped)
                raise


            row = {
                "scenario_id": scenario_id,
                "backend_answer": answer_backend,
                "model_answer": answer_model,
                "judge_backend": JUDGE_BACKEND,
                "judge_model": JUDGE_MODEL,
                "generated_at": datetime.datetime.now().isoformat(),
            }

            # 把 8 个 rubric 的得分抄进 row
            for key in RUBRIC_KEYS:
                row[key] = scores_obj.get(key, None)

            row["comment"] = scores_obj.get("comment", "")

            writer.writerow(row)
            print("  -> judged and saved.\n")
            # 为了减轻 Groq 限流压力，对每条多等一会
            if JUDGE_BACKEND == "groq":
                time.sleep(1.0)

    print("All responses judged.")
    print(f"Scores saved to: {OUTPUT_SCORES_CSV}")


if __name__ == "__main__":
    main()
