# ParentBench-LLM-Evals 工程说明文档（最新版）

文档版本：2026-03-03  
适用对象：项目协作者、外部评审、潜在接入方、研究报告读者  
口径说明：本说明以仓库当前代码与当前产物为准，历史 `.docx` 仅作为补充参考。

---

## 1. 项目定位与目标

ParentBench-LLM-Evals 是一个面向亲子沟通场景（Parenting Advice）的 LLM 评测工程系统，目标不是只做“平均分排行榜”，而是同时回答三类问题：

1. 模型能力：在 8 个育儿建议维度上表现如何。  
2. 工程稳定性：是否存在空输出、截断、API 异常、重试依赖。  
3. 可部署性：不同模型在中英文和不同 Judge 视角下是否稳定、可解释、可复现。

项目整体流程：

`Scenario -> Generation -> Quality Control -> Judging -> Aggregation -> Analysis -> Auto Report`

---

## 2. 当前版本事实快照（截至 2026-03-03）

### 2.1 代码规模（本仓库）

- `src/` 文件数：25  
- `scripts/` 文件数：15  
- `src/analysis/` 文件数：15

### 2.2 数据与实验规模（当前已有产物）

- 英文场景：100 条（`data/scenarios/views/en/parentbench_v0_en.jsonl`）  
- 中文场景：100 条（`data/scenarios/views/zh/parentbench_v0_zh.jsonl`）  
- 生成结果文件：30 个（15 模型 x 2 语言），每个文件 100 条  
- Judge 结果文件：60 个（30 生成文件 x 2 Judge）
  - 56 个文件为 100 条
  - 2 个文件为 92 条
  - 2 个文件为 82 条
- 合并评分表：`results/merged/all_judge_scores.csv`，解析后 5948 条记录（`pandas` 行数）
- Style-2D 明细：3000 条（`style_2d_rows.csv`）

### 2.3 当前 Judge 与模型覆盖（合并表口径）

- 被评估 answer_model：15 个  
- Judge 模型：2 个  
  - `gpt-5.2`
  - `deepseek-v3.1:671b-cloud`
- 语言：`en` / `zh`

---

## 3. 工程架构总览

### 3.1 分层设计

1. **核心执行层（src）**：统一模型调用、生成、评分、分析。  
2. **流程编排层（scripts）**：一键跑全链路、质量修复、自动报告。  
3. **数据层（data）**：场景、生成输出、Judge 输出。  
4. **结果层（results）**：评分表、分析图、报告文档。

### 3.2 核心目录

- `src/config.py`：路径、模型池、默认参数、标签清洗。  
- `src/model_caller.py`：统一调用入口（OpenAI/Ollama）。  
- `src/run_generation.py`：批量生成与截断重试。  
- `src/run_judging.py`：Judge 打分与多次重复聚合。  
- `src/analysis/*.py`：统计、可视化、结构分析、跨语言分析。  
- `scripts/run_multi_generation_and_judge.sh`：多模型 + 多 Judge 一键主脚本。  
- `scripts/run_full_en_zh.sh`：EN+ZH 双语全流程编排。  
- `scripts/generate_llm_report_with_vision.py`：自动长报告生成（Vision + Writer）。

---

## 4. 数据模型与文件约定

### 4.1 Scenario 结构（JSONL）

核心字段：

- `scenario_uid`
- `language`
- `scenario_text`
- `source`
- `metadata`

语言视图按文件分离（而非混合）：

- `data/scenarios/views/en/*.jsonl`
- `data/scenarios/views/zh/*.jsonl`

### 4.2 Generation 结果结构（JSONL）

关键字段：

- `answer_raw`、`answer`
- `model`、`backend`
- `finish_reason`、`api_error`
- `reasoning_stripped`
- `suspected_truncation`
- `retry_used`

说明：系统保留原始输出和清洗后输出，便于故障追踪。

### 4.3 Judge 结果结构（JSONL/CSV）

8 个 rubric（0-100 整数评分，聚合后为均值/方差）：

- `accuracy`
- `safety`
- `helpfulness`
- `empathy`
- `completeness`
- `bias_avoidance`
- `limitation_awareness`
- `communication`

并包含：

- `raw_judge_runs`（每次评分原始解析结果）
- `*_std`（重复评分标准差）
- `comment`（Judge 简评）

---

## 5. 核心流程详解

### 5.1 Generation（`src/run_generation.py`）

关键能力：

1. 按后端解析模型参数（OpenAI/Ollama/Groq 键名约定）。  
2. `final_only` 提示词策略，禁止输出推理过程。  
3. 输出清洗（去 `<think>`、Reasoning 文本）。  
4. 截断检测与 token 扩容重试。  
5. 逐条写回 JSONL，保留完整元信息。

### 5.2 Quality Control（`scripts/check_generation_completeness.py` + `scripts/retry_bad_generations.py`）

提供两类闭环：

1. 统计诊断：按模型/语言统计空输出、截断率、API 错误率。  
2. 定向修复：仅对空输出、仅对截断、或两者一起重跑；不覆盖原文件。

### 5.3 Judging（`src/run_judging.py`）

要点：

1. Judge Prompt 有中英文版本，要求严格 JSON 输出。  
2. 支持 `n_repeats` 多次评分，自动计算均值和标准差。  
3. 对模型输出容错解析（代码块/包裹文本中抽取 JSON）。  
4. 输出可直接进入聚合层。

### 5.4 Aggregation（`src/analysis/export_scores.py` / `collect_and_merge.py` / `merge_judge_scores.py`）

1. 单个 Judge JSONL -> CSV。  
2. 多文件拼接为统一 merged 表。  
3. 统一列顺序，便于后续分析脚本复用。

### 5.5 Analysis（`src/analysis/`）

主要分析族：

1. **基础统计与可视化**：`analyze_scores.py`
   - boxplot/radar/winrate
   - judge agreement
   - PCA embedding
   - EN/ZH 跨语言差异热图和场景级漂移图
2. **能力分组分析**：`basic_grouped_capability_report.py`
   - Constructive Answering vs Responsible Alignment
3. **分组有效性验证**：`validate_rubric_split_anysize_ranked.py`
   - 固定锚点 `accuracy` 枚举分组，按相关结构 signal 排序
4. **场景价值挖掘**：`scenario_value_mining.py`
   - Constructive-weak / Responsible-weak / Polarizing / Balanced-hard / Imbalanced
5. **风格二维分析（R,D）**：
   - `run_style_judging.py`（0-1 连续打分）
   - `extract_style_2d.py`（映射四类风格概率）
   - `analyze_style_2d.py`（散点/箱线/堆叠/主风格热图）
6. **因子与结构稳定性**：
   - `rubric_factor_analysis.py`
   - `model_factor_visualization_and_stability.py`
   - `model_overall_competence_and_structure.py`

### 5.6 Auto Report（`scripts/generate_llm_report_with_vision.py`）

两阶段：

1. Vision 模型读图（批量图像证据解读）。  
2. Writer 模型融合文本证据与图像证据生成长报告。

输出：

- `results/analysis/report/*.md`
- 可再用 `scripts/markdown_to_docx.py` 转为 `.docx`

---

## 6. 运行方式（推荐）

### 6.1 单步执行（最小闭环）

1. 生成：

```bash
python -m src.run_generation \
  --scenarios data/scenarios/views/en/parentbench_v0_en.jsonl \
  --backend openai \
  --openai-model gpt-5.2 \
  --output data/model_outputs/en_openai_gpt-5.2.jsonl
```

2. 评分：

```bash
python -m src.run_judging \
  --answers data/model_outputs/en_openai_gpt-5.2.jsonl \
  --backend openai \
  --openai-model gpt-5.2 \
  --n-repeats 3 \
  --output data/judge_outputs/en_openai_gpt-5.2_judged_openai_gpt-5.2.jsonl
```

3. 聚合与分析：

```bash
python -m src.analysis.collect_and_merge \
  --judge-dir data/judge_outputs \
  --scores-dir results/scores \
  --merged-output results/merged/all_judge_scores.csv

python -m src.analysis.analyze_scores \
  --input results/merged/all_judge_scores.csv \
  --language all
```

### 6.2 一键全流程

```bash
bash scripts/run_full_en_zh.sh
```

---

## 7. 对外可交付产物

面向外部说明时，建议以以下目录为“交付面”：

1. `results/merged/all_judge_scores.csv`：主数据表。  
2. `results/analysis/all/`：核心图表总集（含跨语言分析）。  
3. `results/analysis/style_2d_tables/`：风格二维明细与模型汇总。  
4. `results/analysis/report/`：自动生成报告（md/docx）。  
5. `docs/ParentBench-LLM-Evals_工程说明文档_2026-03-03.md`：本说明文档。

---

## 8. 与历史说明文档的关系

已核对三份历史文档：

1. `双维度 Parenting Style 结构分析报告 2.11.docx`  
2. `ParentBench-LLM-Evals 工程说明文档 1.7.docx`  
3. `Rubric 分类与能力结构分析进展说明 2.4.docx`

保留内容：总体流程、风格二维定义、rubric 分组思路。  
更新内容：以当前代码真实入口、脚本行为、目录结构和实际产物规模为准。

---

## 9. 已知限制与技术债（对外必须如实说明）

1. **Groq 调用实现不完整**  
   - `src/config.py` 与脚本层声明了 Groq 支持；  
   - 但 `src/model_caller.py` 当前仅实现 OpenAI/Ollama 调用分支。  
   - 结论：Groq 在脚本层“可配置”，但核心调用层仍需补全，才能保证端到端可用性。

2. **部分编排脚本存在参数冗余/历史残留**  
   - `scripts/judge_existing_answers.sh` 存在重复 `--backend` 传参与 Groq 分支不一致写法。  
   - 建议优先使用 `scripts/run_multi_generation_and_judge.sh` 作为主流程入口。

3. **历史兼容代码并存**  
   - `src/model_caller_openai.py` 与主调用器并行存在，属于历史兼容资产。  
   - 外部接入时应以 `src/model_caller.py` 为主入口。

4. **记录数与物理行数不一致是正常现象**  
   - 合并 CSV 中 `comment` 字段可含换行，导致文件物理行数高于记录条数。  
   - 对外统计应以数据框解析行数为准。

---

## 10. 对外一句话摘要（可直接引用）

ParentBench-LLM-Evals 是一个可复现、可诊断、可扩展的亲子沟通场景 LLM 评测工程系统，覆盖多模型双语生成、结构化多 Judge 评分、工程故障修复、能力与风格双视角分析，以及面向决策的自动化报告生成。
