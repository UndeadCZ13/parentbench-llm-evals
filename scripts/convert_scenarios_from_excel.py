# scripts/convert_scenarios_from_excel.py

from pathlib import Path
import pandas as pd
import json


def main():
    # 1. 输入/输出路径
    in_path = Path("data/scenarios/v0_ParentBench_Scenarios.xlsx")
    out_path = Path("data/scenarios/parentbench_v0.jsonl")

    if not in_path.exists():
        raise FileNotFoundError(
            f"Input Excel not found: {in_path}. "
            "Please put your 'v0 ParentBench Scenarios.xlsx' under data/scenarios/ and "
            "rename it to 'v0_ParentBench_Scenarios.xlsx'."
        )

    print(f"Reading Excel from: {in_path}")
    df = pd.read_excel(in_path)

    # 确认列名
    expected_cols = {"Question", "Description", "Tags"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in Excel: {missing}")

    records = []
    for idx, row in df.iterrows():
        q = str(row["Question"]).strip()
        desc = str(row["Description"]).strip() if not pd.isna(row["Description"]) else ""
        tags_raw = str(row["Tags"]).strip() if not pd.isna(row["Tags"]) else ""

        # 简单生成一个 id，例如 pb_v0_0001
        scenario_id = f"pb_v0_{idx:04d}"

        record = {
            "id": scenario_id,
            "prompt": q,
            "description": desc,
            # 暂时不强行解析 tags，保持原始字符串，后面需要时再细化
            "tags_raw": tags_raw,
        }
        records.append(record)

    # 写成 JSONL（每行一个 JSON 对象）
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Converted {len(records)} scenarios to JSONL:")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
