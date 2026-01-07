# ParentBench-LLM-Evals

ParentBench-LLM-Evals 是一个**面向研究与工程诊断的多语言大语言模型（LLM）评测流水线**，用于系统化评估不同模型在「亲子沟通（Parenting Advice）」场景中的生成能力、稳定性与语言鲁棒性。

本工程不仅关注模型最终得分，更强调：

* 多模型、跨厂商、跨规模的**统一评测**
* 中英文及跨语言（EN vs ZH）**能力差异分析**
* 生成失败、截断、重试等**工程问题的显式建模**
* 可诊断、可修复、可复现的完整评测流程
* 自动生成**可部署导向的长篇分析报告（Markdown / DOCX）**

---

## 一、整体 Pipeline 概览

完整评测流程如下：

```
Scenario 构建
   ↓
Generation（多模型生成 + Retry / 截断检测）
   ↓
Quality Control（完整性统计 & 定向重跑）
   ↓
Judging（单 / 多 Judge 结构化评分）
   ↓
Aggregation（分数合并、统计）
   ↓
Analysis（语言内 / 跨语言分析 + 可视化）
   ↓
Auto Report（Vision → Writer 两阶段自动报告）
```

---

## 二、工程目录结构

```
parentbench-llm-evals/
├── data/                     # 原始与中间数据
│   ├── scenarios/            # Scenario 定义与语言视图
│   ├── model_outputs/        # 各模型生成结果
│   └── judge_outputs/        # Judge 评分结果
│
├── results/                  # 所有分析与报告产物
│   ├── scores/               # 聚合后的分数 CSV
│   ├── merged/               # 多 Judge 合并结果
│   └── analysis/             # 可视化、统计、报告
│
├── scripts/                  # 高层脚本（一键流水线）
├── src/                      # 核心 Python 实现
└── README.md
```

---

## 三、Scenario 构建与语言视图

### 1. Scenario 设计

* 每个 scenario 描述一个具体的亲子互动情境
* 使用稳定的 `scenario_uid`
* JSONL 格式存储，便于追溯与版本管理

### 2. Language View 机制

不同语言 **完全独立存储**，避免混淆：

* `en`：英文
* `zh`：中文
* `en_zh`：中英对照（可选）

相关脚本：

```bash
python scripts/build_scenario_views.py
python scripts/translate_scenarios_to_zh.py
python scripts/generate_zh_scenarios.py
```

---

## 四、Generation（模型生成）

### 1. 支持的模型后端

* OpenAI
* Ollama（本地 / 云端）
* Groq

统一通过 `model_caller.py` 封装，支持：

* 模型简称解析
* 批量调用
* Retry + 指数退避
* 截断检测与补救生成

### 2. 单模型生成

```bash
python src/run_generation.py \
  --model gpt-5.2 \
  --language en \
  --scenario_file data/scenarios/views/en/scenarios.jsonl
```

常用参数：

| 参数              | 说明         |
| --------------- | ---------- |
| `--model`       | 模型名称或简称    |
| `--language`    | en / zh    |
| `--max_tokens`  | 最大生成 token |
| `--retry_limit` | API 失败重试次数 |

---

## 五、Judging（评测与打分）

### 1. Judge 模型

* 支持单 Judge 或多 Judge 并行
* 输出结构化 JSON 分数

### 2. 执行评测

```bash
python src/run_judging.py \
  --judge_model gpt-5.2 \
  --language en
```

或直接对已有生成结果评测：

```bash
bash scripts/judge_existing_answers.sh
```

---

## 六、质量控制（Quality Control）

### 1. 生成完整性检查

```bash
python scripts/check_generation_completeness.py
```

自动统计：

* 空输出
* 疑似截断输出
* API 错误率

### 2. 定向重跑失败样本

```bash
python scripts/retry_bad_generations.py \
  --only_empty \
  --max_retries 15
```

特点：

* **不覆盖原始生成结果**
* 明确区分工程失败 vs 模型能力限制

---

## 七、分数聚合与分析

### 1. 分数合并

```bash
python src/analysis/collect_and_merge.py
python src/analysis/merge_judge_scores.py
```

### 2. 分析与可视化

```bash
python src/analysis/analyze_scores.py \
  --language all
```

支持：

* Radar / Boxplot / PCA
* Win-rate 分析
* Judge Agreement
* 跨语言 EN vs ZH 差异分析

---

## 八、自动化分析报告（Auto Report Pipeline）

### 1. 设计

采用 **两阶段报告生成**：

1. **Vision Model**：读取分析图像，生成证据性解读
2. **Writer Model**：整合表格 + 文本 + 图像，输出长报告

### 2. 生成报告

```bash
python scripts/generate_llm_report_with_vision.py \
  --language all
```

输出：

* `results/analysis/report/*.md`
* `results/analysis/report/*.docx`

报告特性：

* ≥ 4000 中文字
* 覆盖所有模型
* 模型分组（Taxonomy）
* 重点模型对比表

---

## 九、一键运行完整流水线

### 中英文全流程

```bash
bash scripts/run_full_en_zh.sh
```

### 多模型生成 + 评测

```bash
bash scripts/run_multi_generation_and_judge.sh
```

---

## 十、环境与复现

### 1. 环境变量

```bash
cp .env.example .env
# 填写 OpenAI / Groq / Ollama 相关 Key
```

### 2. 运行特性

* 支持 tmux / 长时间运行
* 可中断、可续跑
* 所有中间产物可追溯

---

## 十一、适用场景

* 多模型能力横向对比
* 中英文 / 跨语言鲁棒性研究
* LLM 工程稳定性诊断
* 决策导向的模型选型分析

---

## License

Research & internal evaluation use only.
