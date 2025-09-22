# mvp/utils.py

import json
import time
import requests
from openai import OpenAI
import config
import random

# --- OpenAI 客户端初始化 ---
# 使用config.py中的配置初始化一个全局客户端
try:
    client = OpenAI(
        api_key=config.OPENKEY_API_KEY,
        base_url=config.OPENKEY_BASE_URL
    )
except Exception as e:
    print(f"初始化OpenAI客户端时出错: {e}")
    client = None

# --- 辅助函数 ---

def call_llm(prompt: str, model: str, is_json: bool = False, system_prompt: str = None):
    """
    A general-purpose LLM call wrapper with exponential backoff and jitter.

    Args:
        prompt (str): The main prompt for the user role.
        model (str): The name of the model to use.
        is_json (bool): Whether to expect a JSON formatted output.
        system_prompt (str, optional): The prompt for the system role. Defaults to None.

    Returns:
        str or None: The LLM's response content, or None if it fails.
    """
    if not client:
        print("错误: OpenAI客户端未初始化。")
        return None

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response_format = {"type": "json_object"} if is_json else None
    
    # --- New Retry Logic ---
    for attempt in range(config.MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
                response_format=response_format
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # If this is the last attempt, print the final error and return None
            if attempt == config.MAX_RETRIES - 1:
                print(f"对模型 {model} 的LLM调用最终失败。错误: {e}")
                break

            # Calculate delay with exponential backoff and jitter
            backoff_time = config.INITIAL_BACKOFF * (2 ** attempt)
            jitter = random.uniform(0, 1)
            delay = backoff_time + jitter
            
            print(f"对模型 {model} 的LLM调用失败，将在 {delay:.2f} 秒后重试... (尝试 {attempt + 1}/{config.MAX_RETRIES}) 错误: {e}")
            time.sleep(delay)
            
    return None # Return None if all retries fail

def execute_search(query: str) -> str:
    """
    使用Serper.dev API执行搜索查询。

    Args:
        query (str): 搜索查询字符串。

    Returns:
        str: 拼接好的搜索结果摘要，或在出错时返回错误信息。
    """
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": 3}) # 获取前3个结果
    headers = {
        'X-API-KEY': config.SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        results = response.json()
        snippets = [res.get("snippet", "") for res in results.get("organic", [])]
        return "\\n".join(filter(None, snippets))
    except requests.exceptions.RequestException as e:
        print(f"\\n[错误] Serper API调用失败: {e}")
        return "SEARCH_API_ERROR"
    except json.JSONDecodeError:
        print(f"\\n[错误] 解码来自Serper的JSON响应失败。")
        return "SEARCH_API_ERROR"