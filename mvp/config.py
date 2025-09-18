# mvp/config.py

import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# --- 核心API配置 ---
# 检查并加载必要的环境变量
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")

if not all([OPENROUTER_API_KEY, SERPER_API_KEY]):
    raise ValueError("错误: 缺少必要的环境变量: OPENROUTER_API_KEY, SERPER_API_KEY")

# --- 模型选择 ---
# 您可以根据性能和成本需求更换这些模型
# 用于生成初始假设图谱的模型 (来自实验 1)
HYPOTHESIZER_MODEL = "google/gemini-2.5-flash"
# 用于从三元组生成搜索查询的模型 (来自实验 2)
QUERY_GENERATOR_MODEL = "google/gemini-2.5-flash-lite"
# 用于判断证据是否支持三元组的验证器模型 (来自实验 3)
VERIFIER_MODEL = "google/gemini-2.5-flash"
# 用于基于已验证的证据生成最终答案的规划/回答模型 (来自实验 4)
ANSWERER_MODEL = "google/gemini-2.5-pro"

# --- API调用参数 ---
LLM_TEMPERATURE = 0.1 # 较低的温度以获得更确定的输出