"""
analyze_scores.py

用途：
  - 读取 merge 后的 all_judge_scores.csv
  - 以 (answer_model, judge_model) 为单位，做 rubric 统计：
      * mean / std / min / max / count
      * 额外计算一列 overall_mean（多个 rubrics 的平均值）
  - 生成基础可视化（箱线图）：
      * 对于每个 judge_model，画每个 rubric 在不同回答模型上的分布
  - 生成模型间相关性矩阵（同一 judge 下，不同回答模型的 overall_mean Spearman 相关）
  - 生成雷达图：
      * 为每个回答模型单独画一张雷达图
      * 为多个模型画在同一张雷达图上便于对比（总体 + 按 judge 分组）

输入：
  results/merged/all_judge_scores.csv

输出（示例）：
  results/analysis/model_judge_stats.csv
  results/analysis/model_corr_matrix_judge-<judge>.csv
  results/analysis/model_correlations_by_judge.csv
  results/analysis/box_<rubric>_by_model_judge-<judge>.png
  results/analysis/radar_single_<model>.png
  results/analysis/radar_multi_overall.png
  results/analysis/radar_multi_judge-<judge>.png
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ====== 路径配置 ======
# 如果你的 all_judge_scores.csv 放在别的位置，在这里改路径即可
DATA_FILE = Path("results/merged/all_judge_scores.csv")
OUT_DIR = Path("results/analysis")


# ====== Rubric 列名 ======
RUBRICS: List[str] = [
    "accuracy",
    "safety",
    "helpfulness",
    "empathy",
    "completeness",
    "bias_avoidance",
    "limitation_awareness",
    "communication",
]


def ensure_overall_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    如果没有 overall_mean 列，则用 RUBRICS 的简单平均新建一列。
    """
    if "overall_mean" not in df.columns:
        missing = [r for r in RUBRICS if r not in df.columns]
        if missing:
            raise ValueError(f"数据中缺少以下 rubric 列，无法计算 overall_mean: {missing}")
        df["overall_mean"] = df[RUBRICS].mean(axis=1)
    return df


def get_model_col(df: pd.DataFrame) -> str:
    """
    尝试在 df 中找到回答模型列名：优先 answer_model，其次 model_answer。
    """
    if "answer_model" in df.columns:
        return "answer_model"
    if "model_answer" in df.columns:
        return "model_answer"
    raise ValueError("DataFrame 中未找到回答模型列（answer_model 或 model_answer）。")


def plot_boxplots(df: pd.DataFrame) -> None:
    """
    对每个 judge_model，画各个 rubric（含 overall_mean）在不同回答模型上的箱线图。
    每个图：X 轴模型，Y 轴某一 rubric 的分数。
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    judge_models = df["judge_model"].dropna().unique()
    model_col = get_model_col(df)
    models = df[model_col].dropna().unique()

    # 确认 rubrics 都存在
    for r in RUBRICS:
        if r not in df.columns:
            raise ValueError(f"数据中缺少 rubric 列：{r}")

    cols_to_plot = RUBRICS + ["overall_mean"]

    for judge in judge_models:
        sub = df[df["judge_model"] == judge]

        for rub in cols_to_plot:
            plt.figure(figsize=(8, 6))

            data = [sub[sub[model_col] == m][rub].dropna() for m in models]
            data_nonempty = []
            labels_nonempty = []
            for m, arr in zip(models, data):
                if len(arr) > 0:
                    data_nonempty.append(arr)
                    labels_nonempty.append(m)

            if not data_nonempty:
                plt.close()
                continue

            plt.boxplot(data_nonempty, labels=labels_nonempty, showmeans=True, showfliers=False)
            plt.title(f"Distribution of {rub} by model (judge={judge})")
            plt.ylabel(rub)
            plt.xlabel(model_col)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            fig_path = OUT_DIR / f"box_{rub}_by_model_judge-{judge}.png"
            plt.savefig(fig_path, dpi=200)
            plt.close()
            print(f"[Box] Saved: {fig_path}")


def compute_model_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    以 (answer_model, judge_model) 为单位，做 rubric 统计：
    mean / std / min / max / count。
    """
    model_col = get_model_col(df)
    group_cols = [model_col, "judge_model"]
    agg_cols = RUBRICS + ["overall_mean"]

    grouped = (
        df.groupby(group_cols)[agg_cols]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )

    # 展开多重列索引：变成诸如 accuracy_mean, accuracy_std, ...
    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns.values
    ]
    return grouped


def save_model_stats(df: pd.DataFrame) -> None:
    stats = compute_model_stats(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_file = OUT_DIR / "model_judge_stats.csv"
    stats.to_csv(stats_file, index=False, encoding="utf-8")
    print(f"[Stats] 已保存模型×judge 的统计结果到：{stats_file}")


def compute_and_save_correlations(df: pd.DataFrame) -> None:
    """
    对每个 judge，下计算不同回答模型之间基于 overall_mean 的 Spearman 相关。
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_col = get_model_col(df)

    corr_records = []
    judge_models = df["judge_model"].dropna().unique()

    for judge in judge_models:
        sub = df[df["judge_model"] == judge]

        pivot = sub.pivot_table(
            index="scenario_id",
            columns=model_col,
            values="overall_mean",
        )

        if pivot.shape[1] < 2:
            print(f"[Corr] judge={judge} 只有一个模型，有效列不足，跳过相关性计算。")
            continue

        corr = pivot.corr(method="spearman")

        # 展成记录形式
        for m1 in corr.columns:
            for m2 in corr.columns:
                if m1 >= m2:
                    continue
                corr_records.append(
                    {
                        "judge_model": judge,
                        "model_A": m1,
                        "model_B": m2,
                        "spearman_corr": corr.loc[m1, m2],
                    }
                )

        corr_file = OUT_DIR / f"model_corr_matrix_judge-{judge}.csv"
        corr.to_csv(corr_file, encoding="utf-8")
        print(f"[Corr] 已保存 judge={judge} 的相关矩阵到：{corr_file}")

    if corr_records:
        corr_df = pd.DataFrame(corr_records)
        pair_file = OUT_DIR / "model_correlations_by_judge.csv"
        corr_df.to_csv(pair_file, index=False, encoding="utf-8")
        print(f"[Corr] 已保存模型间相关记录到：{pair_file}")
    else:
        print("[Corr] 模型数量不足以计算模型间相关性。")


# ===================== 雷达图部分 =====================

def _compute_radar_ylim(df: pd.DataFrame, rubric_cols: list[str]) -> float:
    """
    根据数据自动选择雷达图的 y 轴上限：
    - 如果最大值 <= 10，则用 10
    - 否则用向上取整到 10 的整数倍（例如 100）
    """
    max_val = float(df[rubric_cols].max().max())
    if max_val <= 10:
        return 10.0
    # 向上取整到最近的 10 倍
    return float(int((max_val + 9) // 10 * 10))


def plot_single_model_radar(
    df: pd.DataFrame,
    model_name: str,
    rubric_cols: list[str],
    save_path: Path | None = None,
) -> None:
    """
    为单个回答模型画一张雷达图（聚合所有 judge）。
    """
    model_col = get_model_col(df)
    sub = df[df[model_col] == model_name]
    if sub.empty:
        print(f"[Radar-Single] No data for model={model_name}, skip.")
        return

    means = sub[rubric_cols].mean().values
    N = len(rubric_cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    means = np.concatenate((means, [means[0]]))
    angles += angles[:1]

    ylim = _compute_radar_ylim(df, rubric_cols)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(angles, means, linewidth=2)
    ax.fill(angles, means, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(rubric_cols, fontsize=10)
    ax.set_ylim(0, ylim)
    ax.set_title(f"Radar – {model_name} (all judges)", fontsize=14)

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[Radar-Single] Saved: {save_path}")
    else:
        plt.show()


def plot_multi_model_radar(
    df: pd.DataFrame,
    rubric_cols: list[str],
    title: str,
    save_path: Path | None = None,
    judge_filter: str | None = None,
) -> None:
    """
    将多个模型的雷达曲线画在同一张图上，便于对比。
    - 如果 judge_filter 为某个 judge_model 名称，则只使用该 judge 的评分；
      否则使用所有 judge 的数据（整体平均）。
    """
    model_col = get_model_col(df)

    if judge_filter is not None:
        sub = df[df["judge_model"] == judge_filter]
        if sub.empty:
            print(f"[Radar-Multi] judge_filter={judge_filter} 没有数据，跳过。")
            return
    else:
        sub = df

    models = sub[model_col].dropna().unique()
    if len(models) == 0:
        print("[Radar-Multi] 没有发现任何回答模型，跳过雷达图绘制。")
        return

    # 计算每个模型在各 rubric 上的平均分
    model_means: dict[str, np.ndarray] = {}
    for m in models:
        m_sub = sub[sub[model_col] == m]
        if m_sub.empty:
            continue
        model_means[m] = m_sub[rubric_cols].mean().values

    if not model_means:
        print("[Radar-Multi] 所有模型在指定条件下都没有有效数据。")
        return

    N = len(rubric_cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_loop = angles + angles[:1]

    ylim = _compute_radar_ylim(sub, rubric_cols)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for m, vals in model_means.items():
        vals_loop = np.concatenate((vals, [vals[0]]))
        ax.plot(angles_loop, vals_loop, linewidth=2, label=m)
        ax.fill(angles_loop, vals_loop, alpha=0.15)

    ax.set_xticks(angles)
    ax.set_xticklabels(rubric_cols, fontsize=10)
    ax.set_ylim(0, ylim)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    # 假设所有 rubric 都是 0-100 分
    radar_min = 70
    radar_max = 100
    ax.set_ylim(radar_min, radar_max)
    ax.set_yticks([70, 75, 80, 85, 90, 95, 100])
    ax.set_yticklabels([str(t) for t in [70, 75, 80, 85, 90, 95, 100]])

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[Radar-Multi] Saved: {save_path}")
    else:
        plt.show()
def plot_deviation_radar(
    df: pd.DataFrame,
    rubrics: list[str],
    model_col: str = "answer_model",
    out_path: str = "results/analysis/radar_deviation_overall.png",
) -> None:
    """
    绘制“相对于所有模型平均值的偏差”雷达图。

    df:   行中包含各个 rubric 的平均分（例如 merge 后的总体表）
    rubrics: rubric 列名列表，例如 ["accuracy", "safety", ...]
    model_col: 哪一列表示模型名（通常是 "answer_model"）
    """

    # 1. 先按模型聚合（如果 df 里还有 judge 维度，这里会自动对 judge 取平均）
    df_agg = df.groupby(model_col)[rubrics].mean()

    if df_agg.empty:
        print("[WARN] plot_deviation_radar: df_agg 为空，跳过绘图。")
        return

    # 2. 计算每个 rubric 的“所有模型平均值”
    mean_vec = df_agg.mean(axis=0)  # Series，index=rubrics

    # 3. 计算每个模型相对于平均值的“偏差”
    df_diff = df_agg[rubrics] - mean_vec[rubrics]

    # 4. 雷达图角度设置
    labels = rubrics
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    # 5. 计算全局最大偏差，作为坐标轴上下界
    max_abs = float(np.abs(df_diff.values).max())
    if max_abs == 0:
        max_abs = 1.0  # 防止全 0
    # 为了美观，稍微放大一点，比如向上取整到 2 的倍数
    radius = np.ceil(max_abs / 2.0) * 2.0

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # 6. 每个模型画一条线（中心是 0）
    for model_name, row in df_diff.iterrows():
        values = row.values.tolist()
        values += values[:1]  # 闭合
        ax.plot(
            angles,
            values,
            label=model_name,
            linewidth=2.0,
            marker="o",
            markersize=4,
        )
        ax.fill(angles, values, alpha=0.08)

    # 7. 轴与标签设置
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)

    ax.set_ylim(-radius, radius)
    yticks = np.linspace(-radius, radius, num=5)  # 比如 [-R, -R/2, 0, R/2, R]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:.1f}" for t in yticks], fontsize=9)
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")  # 0 轴

    ax.set_title("Radar – Deviation from Model Average (all judges)", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[INFO] Saved deviation radar plot to: {out_path}")


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"找不到数据文件：{DATA_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"读取数据：{DATA_FILE}")
    df = pd.read_csv(DATA_FILE)

    # 确认 rubrics 列都存在
    for r in RUBRICS:
        if r not in df.columns:
            raise ValueError(f"数据中缺少 rubric 列：{r}")

    df = ensure_overall_mean(df)

    # 保存统计结果
    save_model_stats(df)

    # 生成箱线图
    plot_boxplots(df)

    # 相关性分析
    compute_and_save_correlations(df)

    # ===== 雷达图：单模型 & 多模型 =====
    model_col = get_model_col(df)
    models = df[model_col].dropna().unique().tolist()
    print(f"发现回答模型：{models}")

    # 单模型雷达图
    for m in models:
        safe_m = m.replace("/", "_").replace(":", "-").replace(" ", "_")
        single_path = OUT_DIR / f"radar_single_{safe_m}.png"
        plot_single_model_radar(df, m, RUBRICS, save_path=single_path)

    # 多模型合并雷达图（所有 judge）
    multi_overall_path = OUT_DIR / "radar_multi_overall.png"
    plot_multi_model_radar(
        df,
        RUBRICS,
        title="Radar – All Models (all judges)",
        save_path=multi_overall_path,
        judge_filter=None,
    )
    plot_deviation_radar(
        df=df,
        rubrics=RUBRICS,
        model_col="answer_model",
        out_path="results/analysis/radar_deviation_overall.png",
    )


    # 按 judge 分组的多模型雷达图
    judge_models = df["judge_model"].dropna().unique().tolist()
    for j in judge_models:
        safe_j = j.replace("/", "_").replace(":", "-").replace(" ", "_")
        judge_path = OUT_DIR / f"radar_multi_judge-{safe_j}.png"
        title = f"Radar – All Models (judge={j})"
        plot_multi_model_radar(
            df,
            RUBRICS,
            title=title,
            save_path=judge_path,
            judge_filter=j,
        )

    print("\n分析完成。所有结果已输出到:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
