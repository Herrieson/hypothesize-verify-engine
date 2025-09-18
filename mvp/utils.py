# mvp/utils.py

import json
import time
import requests
from openai import OpenAI
import config

# --- OpenAI 客户端初始化 ---
# 使用config.py中的配置初始化一个全局客户端
try:
    client = OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL
    )
except Exception as e:
    print(f"初始化OpenAI客户端时出错: {e}")
    client = None

# --- 辅助函数 ---

def call_llm(prompt: str, model: str, is_json: bool = False, system_prompt: str = None):
    """
    一个通用的LLM调用封装函数，包含基本的重试逻辑。

    Args:
        prompt (str): 发送给用户角色的主提示。
        model (str): 要使用的模型名称。
        is_json (bool): 是否期望返回JSON格式的输出。
        system_prompt (str, optional): 系统角色的提示。 Defaults to None.

    Returns:
        str or None: LLM的响应内容，如果失败则返回None。
    """
    if not client:
        print("错误: OpenAI客户端未初始化。")
        return None

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response_format = {"type": "json_object"} if is_json else None

    for attempt in range(2): # 重试一次
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
                response_format=response_format
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"对模型 {model} 的LLM调用失败，正在重试... 错误: {e}")
            time.sleep(1)
    return None

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