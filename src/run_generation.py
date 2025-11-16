# src/run_generation.py

from pathlib import Path
import json
import datetime

from model_caller_openai import call_model


SCENARIO_FILE = Path("data/scenarios/parentbench_v0.jsonl")

# ===== 在这里选择后端 =====
# backend = "groq"  使用 Groq 模型（例如 qwen/qwen3-32b）
# backend = "local" 使用本地 DeepSeek（Ollama）
BACKEND = "local"


def load_scenarios(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Scenario JSONL not found: {path}")
    scenarios = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scenarios.append(json.loads(line))
    return scenarios


def main():
    print(f"Loading scenarios from: {SCENARIO_FILE}")
    scenarios = load_scenarios(SCENARIO_FILE)
    print(f"Loaded {len(scenarios)} scenarios.\n")

    # 选择模型
    if BACKEND == "groq":
        model_name = "qwen/qwen3-32b"
    elif BACKEND == "local":
        model_name = "deepseek-r1"  # 或 deepseek-r1:1.5b
    else:
        raise ValueError(f"Unknown BACKEND: {BACKEND}")

    print(f"Using backend = {BACKEND}, model = {model_name}\n")

    out_dir = Path("data/model_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_model_name = (
        model_name
        .replace(".", "-")
        .replace(":", "-")
        .replace("/", "-")
    )

    out_path = out_dir / f"parentbench_v0_{BACKEND}_{safe_model_name}_{ts}.jsonl"
    print(f"All responses will be saved to:\n  {out_path}\n")

    # 打开输出文件，一条场景一行
    with out_path.open("w", encoding="utf-8") as f_out:
        for i, sc in enumerate(scenarios, start=1):
            scenario_id = sc.get("id", f"sc_{i:04d}")
            prompt = sc["prompt"]

            print(f"[{i}/{len(scenarios)}] Scenario {scenario_id}")
            print("Prompt:", prompt)
            print("Generating response...\n")

            if BACKEND == "groq":
                response_text = call_model(
                    prompt,
                    backend="groq",
                    model=model_name,
                    temperature=0.7,
                )
            else:  # local
                response_text = call_model(
                    prompt,
                    backend="local",
                    model=model_name,
                )

            record = {
                "scenario_id": scenario_id,
                "backend": BACKEND,
                "model": model_name,
                "prompt": prompt,
                "response": response_text,
                "description": sc.get("description", ""),
                "tags_raw": sc.get("tags_raw", ""),
                "generated_at": datetime.datetime.now().isoformat(),
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  -> done. (saved)\n")

    print("All scenarios processed.")
    print(f"Final output file: {out_path}")


if __name__ == "__main__":
    main()
