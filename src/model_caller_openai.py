# src/model_caller_openai.py
"""
通用模型调用封装：
- backend="groq"  : 使用 Groq 的 DeepSeek 模型（推荐）
- backend="local" : 使用本地 Ollama 的 DeepSeek 模型
"""

from dotenv import load_dotenv
import os
import subprocess
import textwrap
import requests


import time

load_dotenv()

# src/model_caller_openai.py 片段

import time
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def call_groq_chat(
    prompt: str,
    model: str = "qwen/qwen3-32b",
    temperature: float = 0.7,
    max_retries: int = 8,      # 比之前多给几次机会
) -> str:
    """
    通过 Groq API 调用模型（例如 qwen/qwen3-32b），带指数退避重试。
    用于“生成阶段”和“评测阶段”的统一后端。
    """

    api_key = os.getenv("GROQ_API_KEY")
    if api_key is None:
        raise RuntimeError("GROQ_API_KEY not found in environment/.env")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}

    system_prompt = (
        "You are a careful, evidence-informed assistant that gives parenting advice "
        "or evaluates parenting advice. Your behavior must be: safe, developmentally "
        "appropriate, empathetic, and practically useful. If you are unsure, you say so "
        "and suggest seeking professional or local human support instead of guessing."
    )

    data = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }

    for attempt in range(max_retries):
        resp = requests.post(url, headers=headers, json=data, timeout=60)

        # 成功
        if resp.status_code == 200:
            payload = resp.json()
            return payload["choices"][0]["message"]["content"]

        # 限流：429
        if resp.status_code == 429:
            # 简单打印 + 指数退避
            wait_seconds = 2 + attempt * 2  # 2, 4, 6, ... 秒
            print(
                f"[Groq] Rate limit hit (attempt {attempt+1}/{max_retries}), "
                f"waiting {wait_seconds}s then retry..."
            )
            time.sleep(wait_seconds)
            continue

        # 其它错误直接抛出
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text}")

    # 重试多次仍失败
    raise RuntimeError("Groq API failed after max retries due to repeated rate limits or errors.")

# ============ 本地 DeepSeek（Ollama）===========

def call_local_deepseek(
    prompt: str,
    model: str = "deepseek-r1",
    **kwargs,
) -> str:
    """
    使用 Ollama 在本地调用 DeepSeek 模型。
    额外传进来的参数（如 temperature）会被忽略。
    """
    ...


    # 可以在 prompt 前面加一个简单的角色设定
    full_prompt = textwrap.dedent(f"""
    You are a careful, evidence-informed assistant that gives parenting advice.
    Your answers should be safe, developmentally appropriate, empathetic,
    and practically useful. If you are unsure, say so and suggest seeking
    professional or local human support instead of guessing.

    Here is the parent's question:
    {prompt}
    """)

    # 通过子进程调用 `ollama run`
    result = subprocess.run(
        ["ollama", "run", model],
        input=full_prompt,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Ollama command failed with code {result.returncode}:\n{result.stderr}"
        )

    # deepseek-r1 模型可能会输出带 <think>...</think> 的推理过程
    # 这里先简单返回完整输出，你之后可以再写清洗逻辑
    return result.stdout.strip()


# ============ 统一入口函数 ============

def call_model(
    prompt: str,
    backend: str = "groq",
    **kwargs,
) -> str:
    """
    统一模型调用接口：
      backend = "groq"  -> 调用 Groq DeepSeek
      backend = "local" -> 调用本地 Ollama DeepSeek
    其他参数会传递给对应的函数（例如 model, temperature 等）
    """

    backend = backend.lower()
    if backend == "groq":
        return call_groq_chat(prompt, **kwargs)
    elif backend == "local":
        return call_local_deepseek(prompt, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'groq' or 'local'.")
