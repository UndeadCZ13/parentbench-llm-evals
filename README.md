# ParentBench-LLM-Evals

ParentBench-LLM-Evals is a **research- and engineering-oriented evaluation pipeline for Large Language Models (LLMs)**, designed to systematically benchmark how different models provide parenting advice in realistic parent–child communication scenarios.

Unlike conventional benchmarks that focus only on average generation quality, this project emphasizes:

* **Multi-model, cross-vendor, cross-scale evaluation** under a unified framework
* **Multilingual and cross-language (EN vs ZH) robustness analysis**
* Explicit modeling of **engineering failures** (empty outputs, truncation, retries)
* **Diagnosable and repairable** evaluation workflows
* Automatic generation of **deployment-oriented long-form analysis reports** (Markdown + DOCX)

For a Chinese version, see [README_CN.md]
---

## 1. End-to-End Pipeline Overview

The complete evaluation pipeline is structured as:

```
Scenario Construction
   ↓
Generation (multi-model + retry & truncation handling)
   ↓
Quality Control (completeness checks & targeted re-generation)
   ↓
Judging (single / multi-judge structured scoring)
   ↓
Aggregation (score merging & statistics)
   ↓
Analysis (language-specific & cross-language)
   ↓
Auto Report (Vision → Writer, long-form report)
```

---

## 2. Repository Structure

```
parentbench-llm-evals/
├── data/                     # Raw and intermediate data
│   ├── scenarios/            # Scenario definitions and language views
│   ├── model_outputs/        # Model generations (including retried)
│   └── judge_outputs/        # Judge scoring outputs
│
├── results/                  # All evaluation artifacts
│   ├── scores/               # Aggregated CSV scores
│   ├── merged/               # Multi-judge merged results
│   └── analysis/             # Visualizations, statistics, reports
│
├── scripts/                  # High-level orchestration scripts
├── src/                      # Core Python implementation
└── README.md
```

---

## 3. Scenario Construction & Language Views

### 3.1 Scenario Design

* Each scenario represents a concrete parent–child interaction case
* Stable `scenario_uid` for reproducibility
* Stored in JSONL format with metadata

### 3.2 Language View Abstraction

Scenarios are stored **per language**, not mixed:

* `en` – English
* `zh` – Chinese
* `en_zh` – bilingual view (optional)

This guarantees:

* Language-aligned generation
* Controlled evaluation
* Clean cross-language comparison

Relevant scripts:

```bash
python scripts/build_scenario_views.py
python scripts/translate_scenarios_to_zh.py
python scripts/generate_zh_scenarios.py
```

---

## 4. Generation Pipeline

### 4.1 Supported Model Backends

* OpenAI
* Ollama (local or cloud)
* Groq

All models are accessed through a unified abstraction (`model_caller.py`) supporting:

* Model alias resolution
* Batch execution
* Exponential backoff retries
* Truncation detection and recovery

### 4.2 Single-Model Generation Example

```bash
python src/run_generation.py \
  --model gpt-5.2 \
  --language en \
  --scenario_file data/scenarios/views/en/scenarios.jsonl
```

Common parameters:

| Parameter       | Description           |
| --------------- | --------------------- |
| `--model`       | Model name or alias   |
| `--language`    | `en` / `zh`           |
| `--max_tokens`  | Max generation tokens |
| `--retry_limit` | API retry limit       |

---

## 5. Judging & Scoring

### 5.1 Judge Models

* Supports single or multiple judges
* Judges output **structured JSON scores** following a unified rubric

### 5.2 Run Judging

```bash
python src/run_judging.py \
  --judge_model gpt-5.2 \
  --language en
```

Judging existing generations only:

```bash
bash scripts/judge_existing_answers.sh
```

---

## 6. Quality Control & Repair

### 6.1 Generation Completeness Check

```bash
python scripts/check_generation_completeness.py
```

Automatically detects:

* Empty outputs
* Suspected truncation
* API error rates

### 6.2 Targeted Re-generation

```bash
python scripts/retry_bad_generations.py \
  --only_empty \
  --max_retries 15
```

Design principles:

* Original outputs are **never overwritten**
* Explicit separation of **engineering failure vs model limitation**

---

## 7. Score Aggregation & Analysis

### 7.1 Merge Scores

```bash
python src/analysis/collect_and_merge.py
python src/analysis/merge_judge_scores.py
```

### 7.2 Analysis & Visualization

```bash
python src/analysis/analyze_scores.py \
  --language all
```

Supported analyses:

* Radar charts, boxplots, PCA
* Win-rate analysis
* Judge agreement
* Cross-language EN vs ZH comparison

---

## 8. Automated Report Generation (Auto Report Pipeline)

### 8.1 Design

The report system uses a **two-stage architecture**:

1. **Vision Stage**: A multimodal model interprets evaluation figures
2. **Writer Stage**: A text model synthesizes tables, figures, and evidence into a long-form report

This avoids mixing vision and long-context generation in a single model call, improving stability.

### 8.2 Generate Reports

```bash
python scripts/generate_llm_report_with_vision.py \
  --language all
```

Outputs:

* `results/analysis/report/*.md`
* `results/analysis/report/*.docx`

Report guarantees:

* ≥ 4000 Chinese characters (or equivalent length)
* Coverage of all evaluated models
* Model taxonomy and comparison tables
* Deployment-oriented conclusions

---

## 9. One-Click Full Pipelines

### Full EN + ZH Pipeline

```bash
bash scripts/run_full_en_zh.sh
```

### Multi-Model Generation + Judging

```bash
bash scripts/run_multi_generation_and_judge.sh
```

---

## 10. Environment & Reproducibility

### 10.1 Environment Variables

```bash
cp .env.example .env
# Fill in OpenAI / Groq / Ollama API keys
```

### 10.2 Execution Characteristics

* Designed for long-running jobs (tmux-friendly)
* Interruptible and resumable
* All intermediate artifacts are traceable

---

## 11. Intended Use Cases

* Comparative evaluation of LLMs
* Multilingual robustness research
* Engineering stability diagnostics
* Decision-oriented model selection

---

## License

Research and internal evaluation use only.
