ParentBench – LLM Evaluation Pipeline for Parenting Scenarios
ParentBench is an evaluation framework designed to benchmark Large Language Models (LLMs) on parenting advice scenarios.
It implements a complete, reproducible pipeline including:
Scenario ingestion (Excel → JSONL)
Multi-model answer generation (Groq Cloud, local DeepSeek)
LLM-as-judge scoring using 8 pedagogical rubrics
Robust JSON extraction from judge model outputs
Result storage for downstream analysis
🌱 Project Objectives
ParentBench evaluates how well LLMs respond to real-world parenting questions from the parent’s perspective.
Core research questions:
How accurate and safe are LLMs when giving parenting guidance?
Which domains of parenting are handled well or poorly by different models?
How consistent are LLM-as-judge scores across repeated runs?
How do LLM judges compare with human expert ratings?
📂 Project Structure
parentbench-llm-evals/
│
├── data/
│   ├── scenarios/              # Excel and generated scenario JSONL
│   ├── model_outputs/          # LLM-generated answers
│   └── ...
│
├── results/
│   └── scores/                 # CSV scoring results from judges
│
├── src/
│   ├── run_generation.py       # Multi-model LLM answer generation
│   ├── run_judging.py          # LLM-as-judge scoring pipeline
│   ├── model_caller_openai.py  # Backends: Groq, local DeepSeek, OpenAI (future)
│   ├── judges/
│   │   └── judge_prompts.py    # Rubric prompt construction
│   └── utils/                  # JSON extractors, helpers
│
└── README.md
🛠️ Installation
1. Clone repository
git clone https://github.com/USERNAME/parentbench-llm-evals.git
cd parentbench-llm-evals
2. Create Conda environment
conda create -n parentbench-evals python=3.12
conda activate parentbench-evals
3. Install dependencies
pip install -r requirements.txt
Optional:
Install Ollama for local DeepSeek inference
Add Groq/OpenAI API keys
🔑 Environment Variables
Create a .env file in the project root:
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_if_available
🤖 Running LLM Answer Generation
1. Edit model + backend in src/run_generation.py:
Examples:
backend = "groq"
model = "qwen/qwen3-32b"
or
backend = "local"
model = "deepseek-r1"
2. Run the generation script:
python src/run_generation.py
Outputs will appear in:
data/model_outputs/
🧠 Running LLM-as-Judge
1. Configure judge backend in run_judging.py:
JUDGE_BACKEND = "local"
JUDGE_MODEL = "deepseek-r1"
or
JUDGE_BACKEND = "groq"
JUDGE_MODEL = "qwen/qwen3-32b"
2. Run:
python src/run_judging.py
Scoring results (CSV) are saved to:
results/scores/parentbench_v0_judged_{backend}_{model}_{timestamp}.csv
📊 Analysis
You may create a Jupyter notebook in:
notebooks/analysis.ipynb
Recommended analyses:
Score distribution per rubric
Run-to-run consistency of judges
Comparison between judge models
Scenario-level error analysis
Model ranking
🧩 Rubrics (0–10)
Accuracy & Evidence-Basis
Safety & Harm Avoidance
Helpfulness & Actionability
Empathy & Tone
Completeness
Bias & Stereotype Avoidance
Limitation Awareness & Referral
Communication & Context Gathering
🧭 Roadmap / TODO
Engineering
Add OpenAI / Anthropic backends
Parallelize evaluation
Add resume-checkpoint mechanism
Package the pipeline as a Python module
Research
Incorporate human expert ratings
Compute inter-annotator agreement
Extend parenting scenario dataset (ParentBench v1/v2)
Benchmark more LLMs
Prepare publication-ready benchmark