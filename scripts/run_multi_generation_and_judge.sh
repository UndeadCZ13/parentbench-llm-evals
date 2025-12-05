
#!/bin/bash
set -euo pipefail

########################################
# 用户配置区（根据需要修改）
########################################

# 场景文件
SCENARIO_FILE="data/scenarios/parentbench_v0.jsonl"

# 需要生成回答的模型列表（多个用 ; 分隔）
# 约定格式： backend:model_key
#   - 对 openai：model_key 就是 --openai-model
#   - 对 ollama/local：model_key 是 --ollama-model-key（需要在 OLLAMA_MODEL_REGISTRY 中有定义）
ANSWER_MODELS="ollama:ministral3_8b;ollama:ministral3_14b;ollama:minimax_m2;ollama:qwen3_8b"

# 评分用的 Judge 模型（单个）
# 同样使用 backend:model 格式
JUDGE_MODEL_SPEC="openai:gpt-4o-mini"

# 每个 (answer, judge) 组合的重复评分次数
N_REPEATS=3

# 可选：system prompt 文件（留空则使用 run_generation.py 中的默认）
SYSTEM_PROMPT_FILE=""

########################################
# 以下一般不需要改
########################################

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "== 项目根目录: $PROJECT_ROOT"
echo "== 场景文件:   $SCENARIO_FILE"
echo "== Answer 模型列表: $ANSWER_MODELS"
echo "== Judge 模型: $JUDGE_MODEL_SPEC"
echo "== n_repeats:  $N_REPEATS"
echo

mkdir -p data/model_outputs data/judge_outputs results/scores

SCENARIO_STEM="$(basename "$SCENARIO_FILE" .jsonl)"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

# 小工具：bash 版 sanitize
sanitize() {
  # 替换 / 为 -, 空格为 _，冒号为 -
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

echo "== 解析 Judge:"
echo "   backend = $JUDGE_BACKEND"
echo "   model   = $JUDGE_MODEL_ID"
echo

########################################
# 解析 Answer 模型列表
########################################

# 防止 ANSWER_MODELS 未设置或为空导致的 unbound 变量问题
ANSWER_MODELS_STR="${ANSWER_MODELS:-}"
if [[ -z "$ANSWER_MODELS_STR" ]]; then
  echo "[ERROR] ANSWER_MODELS 为空，请在脚本顶部配置至少一个模型，例如："
  echo "        ANSWER_MODELS=\"ollama:gpt_oss;ollama:deepseek_r1\""
  exit 1
fi

# 初始化数组，确保在 set -u 下已定义
MODEL_SPECS=()
IFS=';' read -ra MODEL_SPECS <<< "$ANSWER_MODELS_STR"

if [[ ${#MODEL_SPECS[@]} -eq 0 ]]; then
  echo "[ERROR] 从 ANSWER_MODELS 中未解析到任何模型，请检查格式。当前值: '$ANSWER_MODELS_STR'"
  exit 1
fi

########################################
# 循环处理每一个 Answer 模型
########################################

for SPEC in "${MODEL_SPECS[@]}"; do
  SPEC_TRIM="${SPEC//[[:space:]]/}"
  [[ -z "$SPEC_TRIM" ]] && continue

  IFS=':' read -r BACKEND_RAW MODEL_ID <<< "$SPEC_TRIM" || {
    echo "[WARN] 跳过无法解析的模型配置: '$SPEC'"
    continue
  }

  BACKEND="$(echo "$BACKEND_RAW" | tr '[:upper:]' '[:lower:]')"
  MODEL_TAG="$(sanitize "$MODEL_ID")"

  echo "==============================="
  echo ">>> 处理 Answer 模型: backend=$BACKEND, model_id=$MODEL_ID"
  echo "==============================="

  # ---------- Step 1: 生成回答 (run_generation) ----------
  ANSWERS_FILE="data/model_outputs/${SCENARIO_STEM}_${BACKEND}_${MODEL_TAG}_${TIMESTAMP}.jsonl"

  GEN_CMD=(python src/run_generation.py
           --scenarios "$SCENARIO_FILE"
           --backend "$BACKEND"
           --output "$ANSWERS_FILE")

  if [[ -n "$SYSTEM_PROMPT_FILE" ]]; then
    GEN_CMD+=(--system-prompt-file "$SYSTEM_PROMPT_FILE")
  fi

  if [[ "$BACKEND" == "openai" ]]; then
    GEN_CMD+=(--openai-model "$MODEL_ID")
  elif [[ "$BACKEND" == "ollama" || "$BACKEND" == "local" ]]; then
    GEN_CMD+=(--ollama-model-key "$MODEL_ID")
  else
    echo "[ERROR] 不支持的 answer backend: $BACKEND"
    exit 1
  fi

  echo "== [1] 生成回答 -> $ANSWERS_FILE"
  echo "   命令: ${GEN_CMD[*]}"
  "${GEN_CMD[@]}"
  echo

  # ---------- Step 2: 评分 (run_judging) ----------
  JUDGE_OUTPUT="data/judge_outputs/$(basename "$ANSWERS_FILE" .jsonl)_judged_${JUDGE_BACKEND}_${JUDGE_MODEL_TAG}.jsonl"

  JUDGE_CMD=(python src/run_judging.py
             --answers "$ANSWERS_FILE"
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

  echo "== [2] 评分 -> $JUDGE_OUTPUT"
  echo "   命令: ${JUDGE_CMD[*]}"
  "${JUDGE_CMD[@]}"
  echo

  # ---------- Step 3: 导出得分 CSV (export_scores) ----------
  SCORES_OUT="results/scores/$(basename "$JUDGE_OUTPUT" .jsonl).csv"

  echo "== [3] 导出得分 CSV -> $SCORES_OUT"
  python -m src.analysis.export_scores \
  --input "$JUDGE_OUTPUT" \
  --output "$SCORES_OUT"
  echo


done

echo "======================"
echo "全部模型处理完成。"
echo "生成的文件包括："
echo "  - 回答：      data/model_outputs/*_${TIMESTAMP}.jsonl"
echo "  - 打分结果：  data/judge_outputs/*_${TIMESTAMP}_judged_*.jsonl"
echo "  - 得分表：    results/scores/*_${TIMESTAMP}_judged_*.csv"
echo "======================"
