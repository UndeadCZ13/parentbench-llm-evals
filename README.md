# ParentBench – LLM Evaluation Pipeline for Parenting Scenarios (v0)

ParentBench is an evaluation framework for benchmarking Large Language Models (LLMs) on **parenting advice** tasks.  
This repo implements an end-to-end protocol for:

1. Scenario authoring & conversion (Excel → JSONL)
2. Multi-backend answer generation (OpenAI + Ollama local / Cloud)
3. LLM-as-judge scoring with 8 parenting rubrics (0–100)
4. Robust JSON extraction & aggregation across repeated judgments
5. Score export, merging, and analysis (CSV + plots)

This is the full evaluation pipeline for **ParentBench v0**.

---

## 1. Goals & Evaluation Protocol

ParentBench evaluates how well different LLMs support caregivers in realistic parenting situations.

Typical questions we care about:

- Can models give **accurate & safe** guidance in everyday and high-risk parenting scenarios?
- Do they show **empathy, cultural sensitivity, and limitation awareness**?
- How stable are scores when we re-judge the same answer multiple times?
- How do LLM judges compare with human raters on the same scenarios? :contentReference[oaicite:1]{index=1}  

**Core protocol (ParentBench v0)**

1. Curate parenting scenarios in Excel.
2. Convert scenarios to JSONL and freeze as a versioned benchmark (e.g., `parentbench_v0.jsonl`).
3. Use multiple LLMs to generate answers for each scenario (generation stage).
4. Use an LLM-as-judge to score each `(scenario, answer)` **multiple times** (e.g., 3 repeats).
5. Aggregate rubric scores per scenario (mean + std).
6. Export scores for comparison, merge across runs, and run statistical analysis.

---

## 2. Repository Structure

Overall layout:

```bash
parentbench-llm-evals/
│
├── configs/                     # (reserved for future config files)
│
├── data/
│   ├── scenarios/               # Scenario JSONL (and original Excel if needed)
│   ├── model_outputs/           # LLM-generated answers (generation stage outputs)
│   └── judge_outputs/           # LLM-as-judge raw JSONL outputs
│
├── docs/                        # (optional project docs)
├── notebooks/                   # Jupyter exploration / ad-hoc analysis
│
├── results/
│   ├── scores/                  # Per-run score CSVs (export_scores.py)
│   ├── merged/                  # Merged score tables across runs/judges
│   └── analysis/                # Plots, summary CSVs (analyze_scores.py)
│
├── scripts/
│   ├── convert_scenarios_from_excel.py
│   │   # One-off script: Excel → data/scenarios/*.jsonl
│   └── run_full_judge_and_analyze.sh
│       # One-click pipeline: judging → export → merge → analyze
│
└── src/
    ├── __init__.py
    │
    ├── run_generation.py        # Main script: generate answers for all scenarios
    ├── run_judging.py           # Main script: LLM-as-judge, multi-repeat scoring
    ├── model_caller_openai.py   # Unified backend dispatcher (OpenAI / Ollama)
    │
    ├── judges/
    │   └── judge_prompts.py     # Rubrics, judge prompt builder
    │
    └── analysis/
        ├── export_scores.py     # JSONL → per-scenario score CSV
        ├── merge_judge_scores.py# Merge multiple score CSVs
        ├── analyze_scores.py    # Stats & plots for model comparison
        └── scoring_utils.py     # Score aggregation (mean/std/comments)
```
## 3. Installation
**3.1. Clone & environment**
git clone https://github.com/UndeadCZ13/parentbench-llm-evals.git
cd parentbench-llm-evals

conda create -n parentbench-evals python=3.12
conda activate parentbench-evals
**3.2. Install dependencies**
pip install -r requirements.txt
Typical dependencies:
Core: python-dotenv, requests, pandas, tqdm, openpyxl
Backends: openai (for OpenAI / compatible APIs)
Analysis: matplotlib, seaborn, jupyter
Optional: tiktoken, etc.
You also need:
Ollama (for local models) installed and running if you use local inference
API keys and access for any cloud backends you plan to use
## 4. Environment & Backends
All secrets are loaded via python-dotenv from a .env file in the project root.
Create .env like:
OpenAI / compatible API
OPENAI_API_KEY=your_openai_key_here
optional base URL if using an OpenAI-compatible endpoint
OPENAI_BASE_URL=https://api.openai.com/v1

Ollama local / cloud
Local (default):
OLLAMA_BASE_URL=http://localhost:11434
Cloud:
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_API_KEY=your_ollama_cloud_key_here
**4.1. Unified backend interface**
src/model_caller_openai.py exposes a single entry point:
from model_caller_openai import call_model

answer = call_model(
    prompt="Why is the sky blue?",
    backend="openai",           # or "ollama" / "local"
    model="gpt-4o-mini",        # OpenAI model name, or Ollama model key
)
backend="openai"
Uses OPENAI_API_KEY + optional OPENAI_BASE_URL
backend="ollama" / "local"
Uses OLLAMA_BASE_URL
If https://ollama.com, automatically attaches Authorization: Bearer $OLLAMA_API_KEY
Designed to work for both local and Cloud models
The caller is written so that adding a new model or swapping base URLs is simple and centralized.
## 5. Step 0 – Preparing Scenarios (Excel → JSONL)
If you start from an Excel file of parenting scenarios, use:
python scripts/convert_scenarios_from_excel.py
This script converts your curated Excel into a JSONL benchmark file such as:
data/scenarios/parentbench_v0.jsonl
Each line is a scenario record with a unique scenario_id and text fields (e.g., prompt, tags, etc.).
This JSONL is the single source of truth for the benchmark scenarios.
## 6. Step 1 – Answer Generation (run_generation.py)
Main script: src/run_generation.py
**6.1. Key options**
Internally it uses:
DEFAULT_SCENARIO_FILE = data/scenarios/parentbench_v0.jsonl
DEFAULT_BACKEND (e.g., "ollama")
DEFAULT_OLLAMA_MODEL_KEY (e.g., "glm_4_6")
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
You can override via CLI:
Example: use Ollama Cloud model registered as "glm_4_6"
python src/run_generation.py \
  --scenarios data/scenarios/parentbench_v0.jsonl \
  --backend ollama \
  --ollama-model-key glm_4_6

Example: use OpenAI GPT-4o mini
python src/run_generation.py \
  --scenarios data/scenarios/parentbench_v0.jsonl \
  --backend openai \
  --openai-model gpt-4o-mini
Internally:
The script resolves the concrete model name from the backend:
For ollama, it looks up OLLAMA_MODEL_REGISTRY[ollama_model_key].
For openai, it uses the --openai-model string directly.
For each scenario:
Builds a prompt (optionally with a system prompt, default DEFAULT_SYSTEM_PROMPT).
Calls call_model(...).
Writes a JSONL record to data/model_outputs/ with:
scenario_id, model_backend, model_name, answer_text, timestamps, etc.
Output example
data/model_outputs/parentbench_v0_ollama_glm-4-6_20251205-133105.jsonl
Each line = one scenario + one model answer.
## 7. Step 2 – LLM-as-Judge (run_judging.py)
Main script: src/run_judging.py
This stage reads the generated answers and scores them with an LLM judge using the ParentBench rubrics.

**7.1. Rubrics & prompts**
Rubrics and prompt templates live in:
src/judges/judge_prompts.py
Key constant:
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
The judge prompt builds a structured instruction that asks the LLM judge to output a strict JSON object with integer scores from 0–100 for each rubric, plus optional comments.
**7.2. Multi-repeat judging**
run_judging.py supports repeated scoring of the same answer to measure stability:
--n_repeats: number of independent judge runs per (scenario, answer)
(e.g., 3 – recommended default)
CLI examples:
Use OpenAI judge (e.g., gpt-4o-mini), 3 repeats
python src/run_judging.py \
  --answers data/model_outputs/parentbench_v0_ollama_glm-4-6_20251205-133105.jsonl \
  --backend openai \
  --openai-model gpt-4o-mini \
  --n_repeats 3 \
  --scenarios data/scenarios/parentbench_v0.jsonl

Use Ollama judge (local / Cloud), 3 repeats
python src/run_judging.py \
  --answers data/model_outputs/parentbench_v0_ollama_glm-4-6_20251205-133105.jsonl \
  --backend ollama \
  --ollama-model-key glm_4_6 \
  --n_repeats 3 \
  --scenarios data/scenarios/parentbench_v0.jsonl
For each answer, the script:
Builds n_repeats judge prompts using build_judge_prompt(...).
Calls call_model(...) for each repeat.
Parses and stores each raw judge JSON as an element of raw_judge_runs.
Aggregates them via scoring_utils.aggregate_runs into:
Per-rubric mean and std
A representative comment (typically first non-empty)
Writes a record to data/judge_outputs/*.jsonl containing:
Scenario metadata
Answer model info
Judge backend/model info
n_repeats
Aggregated rubric scores
Full raw_judge_runs
Output example
data/judge_outputs/parentbench_v0_ollama_glm-4-6_20251205-133105_judged_openai_gpt-4o-mini.jsonl
## 8. Step 3 – Export Scores to CSV (export_scores.py)
Script: src/analysis/export_scores.py
This converts judge JSONL files into a flat CSV table for analysis, one row per (scenario, answer_model, judge_model).

Usage:

python src/analysis/export_scores.py \
  --input data/judge_outputs/parentbench_v0_ollama_glm-4-6_20251205-133105_judged_openai_gpt-4o-mini.jsonl
If you omit --output, it writes to:
results/scores/<judge_output_stem>.csv
e.g. results/scores/parentbench_v0_ollama_glm-4-6_20251205-133105_judged_openai_gpt-4o-mini.csv
Each row includes:
scenario_id, scenario_text
answer_model, answer_backend
judge_model, judge_backend
n_repeats
Per-rubric mean score (accuracy, safety, …)
Per-rubric std (accuracy_std, safety_std, …)
A representative comment
The CSV is generated by re-using scoring_utils.aggregate_runs on raw_judge_runs to ensure consistency.
## 9. Step 4 – Merge Multiple Runs (merge_judge_scores.py)
Script: src/analysis/merge_judge_scores.py
Typical use cases:

Compare multiple judge models (e.g., GPT-4o vs GLM-4).
Compare multiple independent scoring runs (e.g., different seeds / dates).
Combine scores across answer models.
In the script you specify a list of input CSVs, e.g.:
INPUT_FILES = [
    "results/scores/pb_v0_glm46_answer_judge_gpt4omini_run1.csv",
    "results/scores/pb_v0_glm46_answer_judge_gpt4omini_run2.csv",
    "results/scores/pb_v0_glm46_answer_judge_glm4_run1.csv",
]

OUTPUT_FILE = Path("results/merged/all_judge_scores.csv")
Then run:
python src/analysis/merge_judge_scores.py
The script concatenates all input files, checks rubric columns, and writes a single merged CSV to results/merged/.
##  10. Step 5 – Analysis & Visualization (analyze_scores.py)
Script: src/analysis/analyze_scores.py
Defaults:

Input: results/merged/all_judge_scores.csv
Output directory: results/analysis/
What it does (configurable via code):
Ensures all rubrics are present.
Optionally creates an overall_mean as an unweighted average of the core rubrics.
Aggregates statistics by model and judge:
mean / std / min / max / count
Generates plots such as:
Per-rubric boxplots (model comparisons)
Radar charts per model / per judge
Correlation matrices between models’ overall scores
Example output file already in repo:
results/analysis/radar_deepseek-r1.png
## 11. One-Click Pipeline Script
To avoid manually calling each step, you can use:
./scripts/run_full_judge_and_analyze.sh
This script (see its header comments) allows you to configure:
ANSWERS_FILE — which model output JSONL to score
SCENARIO_FILE — scenario benchmark (usually parentbench_v0.jsonl)
JUDGE_BACKEND — "openai" or "ollama"
OPENAI_MODEL / OLLAMA_MODEL_KEY
N_REPEATS — number of judge repeats
MERGED_CSV — where the final merged table should be written
It then runs in order:
python src/run_judging.py ...
python src/analysis/export_scores.py ...
(optionally) merges scores
python src/analysis/analyze_scores.py
You can re-run the whole evaluation for a new answer model or judge by only changing a few variables at the top of this script.
## 12. Adding a New Model
To add a new model to the pipeline:
Register the backend in model_caller_openai.py
If it is OpenAI-compatible, you can re-use call_openai_chat by pointing OPENAI_BASE_URL to the vendor’s endpoint and using the vendor’s key as OPENAI_API_KEY.
If it is an Ollama (local or Cloud) model, add it to the OLLAMA_MODEL_REGISTRY and reference it via --ollama-model-key.
Expose it in run_generation.py
Add it to comments / presets.
Run generation with appropriate --backend and --... flags.
Optionally make it available as a judge model
Run run_judging.py with this backend/model as the judge.
Scores and analysis will automatically pick it up through the CSVs.
The pipeline is intentionally simple and backend-agnostic so you can plug in new models with minimal changes.
## 13. ParentBench Rubrics (0–100)
The current v0 rubrics use 0–100 integer scores (higher is better) for each dimension:
Accuracy & Evidence Basis
Safety & Harm Avoidance
Helpfulness & Actionability
Empathy & Supportive Tone
Completeness & Depth
Bias & Stereotype Avoidance
Limitation Awareness & Referral
Communication & Context Handling
For each (scenario, answer) pair:
The judge LLM outputs a structured JSON with all rubric scores plus free-text comments.
The pipeline repeats judging multiple times and aggregates scores (mean/std), exposing both per-rubric and overall metrics for downstream analysis.