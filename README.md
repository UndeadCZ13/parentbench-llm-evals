# ParentBench – LLM Evaluation Pipeline for Parenting Scenarios

ParentBench is an evaluation framework designed to benchmark Large Language Models (LLMs) on **parenting advice tasks**.  
It provides an end-to-end pipeline for:

1. **Scenario ingestion** (Excel → JSONL)
2. **Multi-model answer generation** (Groq Cloud, local DeepSeek)
3. **LLM-as-judge scoring** using 8 pedagogical rubrics
4. **Robust JSON extraction** from judge outputs
5. **Result storage** for downstream analysis (CSV)

This repository hosts the full evaluation protocol implementation for ParentBench v0.

---

## Project Objectives

ParentBench evaluates how well different LLMs perform in real-world parenting situations from the parent’s perspective.  
The core research questions include:

- How accurately and safely can LLMs provide parenting guidance?
- Do different models excel or fail in different parenting domains?
- How consistent is LLM-as-judge scoring across multiple runs?
- How do LLM judges compare with human expert evaluations?

---

## Project Structure
```
parentbench-llm-evals/
│
├── data/
│   ├── scenarios/              # Excel and generated scenario JSONL
│   ├── model_outputs/          # LLM-generated answers
│   └── ...
│
├── results/
│   └── scores/                 # CSV scoring results
│
├── src/
│   ├── run_generation.py       # Multi-model generation pipeline
│   ├── run_judging.py          # LLM-as-judge scoring pipeline
│   ├── model_caller_openai.py  # Groq / local / OpenAI backends
│   ├── judges/
│   │   └── judge_prompts.py    # Rubrics prompt builder
│   └── utils/                  # JSON extractors, helper functions
│
└── README.md
```

## Installation

### 1. Clone this repository
```
git clone https://github.com/UndeadCZ13/parentbench-llm-evals.git
cd parentbench-llm-evals
```
### 2. Create & activate the Conda environment
```
conda create -n parentbench-evals python=3.12
conda activate parentbench-evals
```
### 3. Install dependencies
```
Core:python-dotenv,requests,pandas,openpyxl
LLM Backends:groq,openai 
Local LLM via subprocess (no Python package needed for ollama)
JSON / parsing utilities
tiktoken
Notebook / analysis
jupyter
matplotlib
seaborn
tqdm
You may also need:
Ollama for local DeepSeek models
Access keys for Groq/OpenAI if using cloud backends
```
## Environment Variables
```
Create a .env file in the project root:
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=optional_if_available
```
## Running LLM Answer Generation
```
Edit backend and model in src/run_generation.py:
##Choose backend:
backend = "groq"      # or "local"
```
## Example models:
```
model = "qwen/qwen3-32b"      # Groq Qwen
model = "deepseek-r1"       # Local DeepSeek
Run:
python src/run_generation.py
This will:
Read the scenario JSONL
Generate LLM responses
Save output to data/model_outputs/...jsonl
```
## Running LLM-as-Judge
```
Select judge backend and model in src/run_judging.py:
JUDGE_BACKEND = "local"     # or "groq"
JUDGE_MODEL = "deepseek-r1" # or "qwen/qwen3-32b"
Run:
python src/run_judging.py
This produces a scored CSV file in:
results/scores/parentbench_v0_judged_{backend}_{model}_{timestamp}.csv
```
## Adding a New Model
```
To add a new backend/model:
Add a new function in model_caller_openai.py
Register it in call_model()
Add to the model list in run_generation.py
(Optional) Add corresponding judge model
The interface is designed to be easily extendable.
```
## Rubrics (0–10)
```
Accuracy & Evidence-Basis
Safety & Harm Avoidance
Helpfulness & Actionability
Empathy & Tone
Completeness
Bias & Stereotype Avoidance
Limitation Awareness & Referral
Communication & Context Gathering
Each score is generated autonomously by a judge LLM.
```