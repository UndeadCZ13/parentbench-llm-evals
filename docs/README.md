# ParentBench-LLM-Evals

ParentBench-LLM-Evals is a **research- and engineering-oriented evaluation system for Large Language Models (LLMs)**, built to benchmark parenting-advice quality in realistic parent-child scenarios.

Compared with benchmark pipelines that only report average scores, this project is designed to be:

* **Multi-model and cross-vendor** under one unified workflow
* **Bilingual and cross-language analyzable** (`en` vs `zh`)
* Explicit about **engineering failures** (empty outputs, truncation, API errors, retries)
* **Diagnosable and repairable**, not only “run once and score”
* Able to produce **long-form, decision-oriented reports** (`.md` + `.docx`)

For the Chinese README, see [README_CN.md](../README_CN.md).

---

## 1. End-to-End Pipeline Overview

```
Scenario Construction
   ↓
Generation (multi-model, retry/truncation-aware)
   ↓
Quality Control (completeness scan + targeted regeneration)
   ↓
Judging (single/multi-judge structured scoring)
   ↓
Aggregation (CSV export + merged score table)
   ↓
Analysis (language-specific + cross-language diagnostics)
   ↓
Auto Report (Vision evidence -> Writer synthesis)
```

---

## 2. Repository Structure

```
parentbench-llm-evals/
├── data/
│   ├── scenarios/                  # scenario definitions & language views
│   ├── model_outputs/              # generation outputs (jsonl)
│   └── judge_outputs/              # judge outputs (jsonl)
│
├── results/
│   ├── scores/                     # per-judge CSV exports
│   ├── merged/                     # merged score tables
│   ├── analysis/                   # plots, diagnostics, reports
│   └── human_annotation_pack/      # optional human annotation exports
│
├── scripts/                        # orchestration & utility scripts
├── src/                            # core pipeline implementation
├── docs/                           # project documents
└── README.md
```

---

## 3. Quick Start (Minimal Reproducible Run)

### 3.1 Environment

Use Python 3.10+ (recommended).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `requirements.txt` is not portable in your environment, install a minimal runtime set:

```bash
pip install openai groq requests python-dotenv pandas numpy matplotlib seaborn scikit-learn tqdm openpyxl python-docx
```

### 3.2 Configure `.env`

Create and fill `.env` at project root:

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_API_KEY=
OLLAMA_ALWAYS_SEND_KEY=0

GROQ_API_KEY=
GROQ_BASE_URL=
```

Quick environment/model check:

```bash
python -m src.list_supported_models
```

### 3.3 Generate -> Judge -> Merge -> Analyze

1) Generate answers (example: OpenAI `gpt-5.2`, English scenarios):

```bash
python -m src.run_generation \
  --scenarios data/scenarios/views/en/parentbench_v0_en.jsonl \
  --backend openai \
  --openai-model gpt-5.2 \
  --output data/model_outputs/en_openai_gpt-5.2.jsonl
```

2) Judge generated answers (example: OpenAI judge `gpt-5.2`, 3 repeats):

```bash
python -m src.run_judging \
  --answers data/model_outputs/en_openai_gpt-5.2.jsonl \
  --backend openai \
  --openai-model gpt-5.2 \
  --n-repeats 3 \
  --max-tokens 1024 \
  --output data/judge_outputs/en_openai_gpt-5.2_judged_openai_gpt-5.2.jsonl
```

3) Export + merge judge outputs:

```bash
python -m src.analysis.collect_and_merge \
  --judge-dir data/judge_outputs \
  --scores-dir results/scores \
  --merged-output results/merged/all_judge_scores.csv
```

4) Run baseline analysis:

```bash
python -m src.analysis.analyze_scores \
  --input results/merged/all_judge_scores.csv \
  --out-dir results/analysis \
  --language all
```

---

## 4. Scenario Construction & Language Views

### 4.1 Scenario Data Layout

Current views are split by language:

* `data/scenarios/views/en/parentbench_v0_en.jsonl`
* `data/scenarios/views/zh/parentbench_v0_zh.jsonl`

Each record is keyed by stable `scenario_uid`.

### 4.2 Build / Convert / Translate

Useful scripts:

```bash
python scripts/convert_scenarios_from_excel.py
python scripts/translate_scenarios_to_zh.py --backend openai --model gpt-4o-mini
python scripts/generate_zh_scenarios.py --theme sleep --age-group toddler --difficulty moderate --n 20
python scripts/build_scenario_views.py
```

---

## 5. Generation Pipeline

Generation entrypoint: `src/run_generation.py`

Key features:

* language-aligned prompt construction
* reasoning-strip cleanup (`answer_raw` vs `answer`)
* truncation suspicion detection
* token-expanded retry on truncation
* per-item metadata logging (`finish_reason`, `api_error`, `retry_used`)

Example (Ollama registry key):

```bash
python -m src.run_generation \
  --scenarios data/scenarios/views/zh/parentbench_v0_zh.jsonl \
  --backend ollama \
  --ollama-model-key deepseek_v3 \
  --max-items 20 \
  --output data/model_outputs/zh_ollama_deepseek_v3.jsonl
```

---

## 6. Quality Control & Repair

### 6.1 Completeness Scan

```bash
python scripts/check_generation_completeness.py \
  --dir data/model_outputs \
  --out results/analysis/generation_completeness.csv
```

### 6.2 Targeted Regeneration

Retry only problematic samples (`empty`, `trunc`, or `both`) without overwriting original files:

```bash
python scripts/retry_bad_generations.py \
  --scenarios data/scenarios/views/zh/parentbench_v0_zh.jsonl \
  --inputs data/model_outputs \
  --pattern "zh_*.jsonl" \
  --retry-mode both \
  --attempts 15 \
  --suffix _retried
```

---

## 7. Judging Pipeline

Judging entrypoint: `src/run_judging.py`

Highlights:

* strict JSON rubric scoring
* `n_repeats` support for judge variance estimation
* robust JSON parsing from imperfect model outputs
* structured output including `raw_judge_runs`, `api_errors`, and aggregated rubric stats

Batch-judge existing answer files:

```bash
bash scripts/judge_existing_answers.sh "data/model_outputs/en_openai_gpt-5.2.jsonl"
```

---

## 8. One-Click Multi-Model Runs

### 8.1 Main Orchestrator (Generation + Judging)

```bash
LANGUAGE_MODE=en \
MAX_ITEMS=20 \
GEN_OPENAI_MODELS="gpt_5_2,gpt_4o_mini" \
GEN_OLLAMA_MODELS="deepseek_v3,qwen3_8b" \
GEN_GROQ_MODELS=none \
JUDGE_SPECS="openai:gpt_5_2,ollama:deepseek_v3" \
N_REPEATS=3 \
bash scripts/run_multi_generation_and_judge.sh
```

### 8.2 Full EN + ZH Run

```bash
SKIP_EXISTING=1 \
N_REPEATS=3 \
bash scripts/run_full_en_zh.sh
```

---

## 9. Analysis Modules

Baseline analysis:

```bash
python -m src.analysis.analyze_scores \
  --input results/merged/all_judge_scores.csv \
  --language all
```

Grouped capability report:

```bash
python -m src.analysis.basic_grouped_capability_report \
  --csv results/merged/all_judge_scores.csv \
  --out results/analysis/basic_capability_report
```

Rubric split validation:

```bash
python -m src.analysis.validate_rubric_split_anysize_ranked \
  --csv results/merged/all_judge_scores.csv \
  --out results/analysis/rubric_split_anysize_ranked
```

Scenario value mining:

```bash
python -m src.analysis.scenario_value_mining \
  --csv results/merged/all_judge_scores.csv \
  --out results/analysis/scenario_value_mining
```

Model overall vs structure:

```bash
python -m src.analysis.model_overall_competence_and_structure \
  --csv results/merged/all_judge_scores.csv \
  --out results/analysis/model_overall_competence
```

---

## 10. 2D Parenting Style Pipeline (R,D)

This project includes a style-only (not quality) interpretation layer:

* `R` = responsiveness in `[0,1]`
* `D` = demandingness in `[0,1]`
* derived style probabilities:
  - authoritative = `R * D`
  - permissive = `R * (1-D)`
  - authoritarian = `(1-R) * D`
  - neglectful = `(1-R) * (1-D)`

Run the full style pipeline:

```bash
python -m src.run_style_judging \
  --model_outputs_dir data/model_outputs \
  --output_dir data/judge_outputs/style_2d \
  --judge_model gpt-5.2 \
  --judge_backend openai

python -m src.analysis.extract_style_2d \
  --input-dir data/judge_outputs/style_2d \
  --language all \
  --out-csv results/analysis/style_2d_tables/style_2d_rows.csv \
  --out-model-csv results/analysis/style_2d_tables/style_2d_model_summary.csv

python -m src.analysis.analyze_style_2d \
  --rows-csv results/analysis/style_2d_tables/style_2d_rows.csv \
  --out-dir results/analysis \
  --language all
```

---

## 11. Automated Long-Form Report

Report generator (`scripts/generate_llm_report_with_vision.py`) uses:

1. **Vision stage**: image-level evidence extraction
2. **Writer stage**: long-form synthesis from tables + text + image evidence

```bash
python scripts/generate_llm_report_with_vision.py \
  --project-root . \
  --analysis-dir results/analysis \
  --include-merged \
  --vision-model gpt-4o-mini \
  --report-model gpt-5-nano \
  --max-images 200
```

Markdown-to-DOCX conversion:

```bash
python scripts/markdown_to_docx.py \
  --md results/analysis/report/<report>.md \
  --out results/analysis/report/<report>.docx
```

---

## 12. Output Artifacts Cheat Sheet

* Generations: `data/model_outputs/*.jsonl`
* Judge outputs: `data/judge_outputs/*.jsonl`
* Per-file score CSVs: `results/scores/*.csv`
* Merged score table: `results/merged/all_judge_scores.csv`
* Baseline analysis figures: `results/analysis/{en|zh|all}/...`
* Style-2D tables: `results/analysis/style_2d_tables/*.csv`
* Auto reports: `results/analysis/report/*`

---

## 13. Known Caveats (Current Codebase)

* Groq is listed in registries and shell orchestration, but core runtime support should be verified in your current branch before production runs.
* `scripts/judge_existing_answers.sh` contains legacy branches; `scripts/run_multi_generation_and_judge.sh` is the recommended main orchestrator.
* CSV physical line count may exceed logical row count because `comment` fields can contain embedded newlines. Use dataframe row count as ground truth.

---

## 14. Recommended Operating Practices

* Run long jobs inside `tmux`.
* Keep `SKIP_EXISTING=1` for resumable pipelines.
* Use `MAX_ITEMS` for smoke tests before full runs.
* Keep all generated artifacts; do not overwrite raw outputs.
* Separate “capability conclusions” from “engineering failure diagnostics” in reporting.

---

## License

Research and internal evaluation use only.

