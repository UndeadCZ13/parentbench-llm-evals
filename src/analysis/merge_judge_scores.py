"""
merge_judge_scores.py

用途：
  - 将多个 judge 结果 CSV 合并为一个 all_judge_scores.csv
  - 自动检查 rubrics 列是否一致，避免混入旧版本文件
  - 在必要时对缺失的 rubrics 补列（填 NaN），并打印警告
  - 为每条记录增加 source_file 列，方便追溯

使用方式：
  1. 修改 INPUT_FILES 列表，填入你想合并的 judge CSV 文件路径
  2. 在项目根目录运行：
       python -m src.analysis.merge_judge_scores
     或：
       cd src && python analysis/merge_judge_scores.py
"""

from pathlib import Path
import pandas as pd

# ======== 需要你手动填写的部分：要合并的 judge 文件列表 ========
INPUT_FILES = [
    # 示例（请改成你自己的文件路径）：
    "results/scores/parentbench_v0_openai_gpt-4o-mini_20251203-220722_judged_openai_gpt-4o-mini.csv",
    "results/scores/parentbench_v0_local_deepseek-r1_20251204-171641_judged_openai_gpt-4o-mini.csv",
    "results/scores/parentbench_v0_groq_qwen-qwen3-32b_20251116-170212_judged_openai_gpt-4o-mini.csv",
    "results/scores/parentbench_v0_ollama_gpt-oss-20b-cloud_20251205-143035_judged_openai_gpt-4o-mini.csv",
    "results/scores/parentbench_v0_ollama_deepseek-v3.1-671b-cloud_20251205-145354_judged_openai_gpt-4o-mini.csv",
    "results/scores/parentbench_v0_ollama_kimi-k2_20251205-162446_judged_openai_gpt-4o-mini.csv",
    "results/scores/parentbench_v0_ollama_ministral3_8b_20251205-173601_judged_openai_gpt-4o-mini.csv",
    "results/scores/parentbench_v0_ollama_ministral3_14b_20251205-173601_judged_openai_gpt-4o-mini.csv",
    "results/scores/parentbench_v0_ollama_minimax_m2_20251205-173601_judged_openai_gpt-4o-mini.csv",
]

# 输出文件路径
OUTPUT_DIR = Path("results/merged")
OUTPUT_FILE = OUTPUT_DIR / "all_judge_scores.csv"

# ======== 如果你有统一的 rubric 列名，可以在这里写死 ========
# 如果你已经在 judges/judge_prompts.py 里定义了 RUBRIC_KEYS，也可以 import
# from judges.judge_prompts import RUBRIC_KEYS
RUBRIC_KEYS = [
    "accuracy",
    "safety",
    "helpfulness",
    "empathy",
    "completeness",
    "bias_avoidance",
    "limitation_awareness",
    "communication",
]

# 其他“元信息”列（如果你有别的，也可以加进来）
META_COLUMNS_CANDIDATES = [
    "scenario_id",
    "backend_answer",    # 生成回答用的 backend（openai/local/groq等）
    "model_answer",      # 生成回答的模型名
    "judge_backend",
    "judge_model",
    "generated_at",
    "comment",
]


def main():
    if not INPUT_FILES:
        raise ValueError("请先在 INPUT_FILES 中填入至少一个 judge 结果 CSV 文件路径。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    all_rubric_sets = []

    print("即将合并以下文件：")
    for fp in INPUT_FILES:
        print("  -", fp)
    print()

    for fp in INPUT_FILES:
        path = Path(fp)
        if not path.exists():
            raise FileNotFoundError(f"找不到文件：{path}")

        df = pd.read_csv(path)

        # 记录来源文件名
        df["source_file"] = path.name

        # 检查有哪些 rubrics 列
        rubric_cols_in_file = [col for col in df.columns if col in RUBRIC_KEYS]
        all_rubric_sets.append(set(rubric_cols_in_file))

        # 提醒可能缺失／多出的列
        missing_rubrics = [r for r in RUBRIC_KEYS if r not in rubric_cols_in_file]
        extra_rubrics = [col for col in rubric_cols_in_file if col not in RUBRIC_KEYS]

        print(f"文件 {path.name}:")
        print(f"  发现的 rubric 列：{rubric_cols_in_file}")

        if missing_rubrics:
            print(f"  ⚠ 缺失的 rubric 列：{missing_rubrics}（将在合并时补充为空值）")
        if extra_rubrics:
            print(f"  ⚠ 额外的 rubric 列（未在 RUBRIC_KEYS 中）：{extra_rubrics}")

        # 对缺失的 rubrics 补列
        for r in RUBRIC_KEYS:
            if r not in df.columns:
                df[r] = pd.NA

        all_dfs.append(df)
        print()

    # 检查所有文件的 rubric 集合是否一致（仅作提示，不强制报错）
    unique_rubric_sets = {tuple(sorted(s)) for s in all_rubric_sets}
    if len(unique_rubric_sets) > 1:
        print("🔎 注意：不同文件的 rubrics 列集合不完全一致，已通过补列方式对齐。")
    else:
        print("✅ 所有文件的 rubrics 列集合一致。")

    # 合并
    merged = pd.concat(all_dfs, ignore_index=True)

    # 统一列顺序（便于后续分析）
    # 先把常见的 meta 列放前面，再是 rubrics，剩下的放后面
    meta_cols_present = [c for c in META_COLUMNS_CANDIDATES if c in merged.columns]
    cols_order = meta_cols_present + RUBRIC_KEYS

    # 把剩余的列（不在 meta + rubrics 里的）追加到最后
    remaining_cols = [c for c in merged.columns if c not in cols_order]
    cols_order += remaining_cols

    merged = merged[cols_order]

    # 保存
    merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print("\n✅ 合并完成。")
    print(f"  共合并 {len(INPUT_FILES)} 个文件，得到 {len(merged)} 条记录。")
    print(f"  输出文件：{OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
