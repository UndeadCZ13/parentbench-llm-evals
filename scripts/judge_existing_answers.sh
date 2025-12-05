#!/bin/bash
set -euo pipefail

########################################
# 用户配置区
########################################

# 场景文件（通常不用改）
SCENARIO_FILE="data/scenarios/parentbench_v0.jsonl"

# 需要被 judge 的 answer 文件名（只写文件名，不带目录，多个用 ; 分隔）
# 例如：ANSWER_FILES="parentbench_v0_local_deepseek-r1_20251204-171641.jsonl;parentbench_v0_ollama_glm-4-6_20251205-133105.jsonl"
ANSWER_FILES="parentbench_v0_local_deepseek-r1_20251204-171641.jsonl"

# Judge 模型（单个），格式：backend:model_id
# - 对 openai：model_id == --openai-model
# - 对 ollama/local：model_id == --ollama-model-key
JUDGE_MODEL_SPEC="ollama:deepseek_v3"

# 每个 (answer, judge) 组合重复评分次数
N_REPEATS=3

########################################
# 以下一般不需要改
########################################

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "== 项目根目录: $PROJECT_ROOT"
echo "== 场景文件:   $SCENARIO_FILE"
echo "== Answer 文件: $ANSWER_FILES"
echo "== Judge 模型: $JUDGE_MODEL_SPEC"
echo "== n_repeats:  $N_REPEATS"
echo

mkdir -p data/judge_outputs results/scores

SCENARIO_STEM="$(basename "$SCENARIO_FILE" .jsonl)"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

sanitize() {
  echo "$1" | tr '/' '-' | tr ' ' '_' | tr ':' '-'
}

########################################
# 解析 Judge 模型
########################################

JUDGE_SPEC_TRIM="${JUDGE_MODEL_SPEC//[[:space:]]/}"
IFS=':' read -r JUDGE_BACKEND_RAW JUDGE_MODEL_ID <<< "$JUDGE_SPEC_TRIM" || {
  echo "[ERROR] 无法解析 JUDGE_MODEL_SPEC='$JUDGE_MODEL_SPEC'，应为 backend:model 格式"
  exit 1
}
JUDGE_BACKEND="$(echo "$JUDGE_BACKEND_RAW" | tr '[:upper:]' '[:lower:]')"
JUDGE_MODEL_TAG="$(sanitize "$JUDGE_MODEL_ID")"

echo "== 解析 Judge 模型:"
echo "   backend = $JUDGE_BACKEND"
echo "   model   = $JUDGE_MODEL_ID"
echo

########################################
# 解析 Answer 文件列表
########################################

ANSWER_FILES_STR="${ANSWER_FILES:-}"
if [[ -z "$ANSWER_FILES_STR" ]]; then
  echo "[ERROR] ANSWER_FILES 为空，请在脚本顶部配置至少一个文件名。"
  exit 1
fi

ANSWER_LIST=()
IFS=';' read -ra ANSWER_LIST <<< "$ANSWER_FILES_STR"

if [[ ${#ANSWER_LIST[@]} -eq 0 ]]; then
  echo "[ERROR] 从 ANSWER_FILES 中未解析到任何文件，请检查格式。当前值: '$ANSWER_FILES_STR'"
  exit 1
fi

########################################
# 循环处理每一个 answer 文件
########################################

for FNAME in "${ANSWER_LIST[@]}"; do
  FNAME_TRIM="${FNAME//[[:space:]]/}"
  [[ -z "$FNAME_TRIM" ]] && continue

  ANSWERS_PATH="data/model_outputs/$FNAME_TRIM"
  if [[ ! -f "$ANSWERS_PATH" ]]; then
    echo "[WARN] 找不到 answer 文件: $ANSWERS_PATH，跳过。"
    continue
  fi

  ANSWERS_STEM="$(basename "$ANSWERS_PATH" .jsonl)"

  echo "==============================="
  echo ">>> 处理 Answer 文件: $ANSWERS_PATH"
  echo "==============================="

  # ---------- Step 1: 评分 ----------
  JUDGE_OUTPUT="data/judge_outputs/${ANSWERS_STEM}_judged_${JUDGE_BACKEND}_${JUDGE_MODEL_TAG}_${TIMESTAMP}.jsonl"

  JUDGE_CMD=(python src/run_judging.py
             --answers "$ANSWERS_PATH"
             --backend "$JUDGE_BACKEND"
             --n_repeats "$N_REPEATS"
             --scenarios "$SCENARIO_FILE"
             --output "$JUDGE_OUTPUT")

  if [[ "$JUDGE_BACKEND" == "openai" ]]; then
    JUDGE_CMD+=(--openai-model "$JUDGE_MODEL_ID")
  elif [[ "$JUDGE_BACKEND" == "ollama" || "$JUDGE_BACKEND" == "local" ]]; then
    JUDGE_CMD+=(--ollama-model-key "$JUDGE_MODEL_ID")
  else
    echo "[ERROR] 不支持的 judge backend: $JUDGE_BACKEND"
    exit 1
  fi

  echo "== [1] 评分 -> $JUDGE_OUTPUT"
  echo "   命令: ${JUDGE_CMD[*]}"
  "${JUDGE_CMD[@]}"
  echo

  # ---------- Step 2: 导出得分 CSV ----------
  SCORES_OUT="results/scores/$(basename "$JUDGE_OUTPUT" .jsonl).csv"

  echo "== [2] 导出得分 CSV -> $SCORES_OUT"
  python -m src.analysis.export_scores \
    --input "$JUDGE_OUTPUT" \
    --output "$SCORES_OUT"
  echo

done

echo "======================"
echo "所有指定 answer 文件已完成打分并导出。"
echo "  - Judge 输出： data/judge_outputs/*_judged_${JUDGE_BACKEND}_${JUDGE_MODEL_TAG}_${TIMESTAMP}.jsonl"
echo "  - 得分 CSV：   results/scores/*_judged_${JUDGE_BACKEND}_${JUDGE_MODEL_TAG}_${TIMESTAMP}.csv"
echo "======================"
