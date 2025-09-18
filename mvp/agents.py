# mvp/agents.py

import json
import config
import utils

# --- Prompt 模板 ---

HYPOTHESIZER_PROMPT = """
你是一个知识图谱构建专家。请根据以下问题，仅利用你的内部知识，生成一个回答该问题所需要的“假设性知识图谱”。
这个图谱应该能逻辑清晰地推导出问题的答案。

图谱应以JSON格式的三元组（主体, 关系, 客体）的列表形式输出，并包含在一个名为 "triples" 的键中。

问题: "{question}"

请直接输出JSON格式的对象。
"""

QUERY_GENERATOR_PROMPT = """
你是一位专业的搜索引擎用户。你的首要任务是为一个给定的知识三元组生成有效的Google搜索查询，以便对其进行验证。
请生成最多3个多样化且简洁的查询。

知识三元组: {triple}

你的最终输出必须是一个JSON对象，其中包含一个名为 "queries" 的键，其值为一个字符串列表。
示例: {{"queries": ["维多利亚女王何时与阿尔伯特亲王结婚", "维多利亚女王 阿尔伯特亲王 婚礼日期"]}}
"""

VERIFIER_PROMPT = """
你是一个高度精确的语言注释员。你的任务是判断“证据摘要”与“知识三元组”之间的关系。
你的决策必须*仅仅*基于所提供的摘要。你的回答必须是单个词: Supports, Refutes, 或 Neutral。

- Supports: 摘要直接陈述或强烈暗示该三元组为真。
- Refutes: 摘要直接陈述或强烈暗示该三元组为假。
- Neutral: 摘要不相关、模棱两可或信息不足。

知识三元组: {triple}
证据摘要:
---
{snippets}
---

你的单字回答:
"""

ANSWERER_PROMPT = """
你是一位智能且客观的问答机器人。你的任务是综合所有已验证的信息来回答用户的原始问题。
你的最终答案应该简洁，并直接回应原始问题。

原始问题: "{question}"

已验证的外部证据:
---
{verified_evidence}
---

基于以上所有信息，请为原始问题提供一个最终的、基于证据的答案。
"""

# --- 智能体定义 ---

def generate_graph(question: str) -> list:
    """
    (假设器) 根据问题生成一个假设性知识图谱。
    """
    raw_output = utils.call_llm(
        prompt=HYPOTHESIZER_PROMPT.format(question=question),
        model=config.HYPOTHESIZER_MODEL,
        is_json=True
    )
    if not raw_output:
        return []
    try:
        data = json.loads(raw_output)
        return data.get("triples", [])
    except (json.JSONDecodeError, AttributeError):
        print(f"[警告] 无法从假设器输出中解析三元组: {raw_output}")
        return []

def generate_queries(triple: list) -> list:
    """
    (查询生成器) 为一个知识三元组生成搜索查询。
    """
    raw_output = utils.call_llm(
        prompt=QUERY_GENERATOR_PROMPT.format(triple=str(tuple(triple))),
        model=config.QUERY_GENERATOR_MODEL,
        is_json=True
    )
    if not raw_output:
        return []
    try:
        data = json.loads(raw_output)
        return data.get("queries", [])
    except (json.JSONDecodeError, AttributeError):
        print(f"[警告] 无法从查询生成器输出中解析查询: {raw_output}")
        return []

def verify(triple: list, snippets: str) -> str:
    """
    (验证器) 判断证据摘要是否支持知识三元组。
    """
    if not snippets or snippets == "SEARCH_API_ERROR":
        return "Neutral"

    response = utils.call_llm(
        prompt=VERIFIER_PROMPT.format(triple=str(tuple(triple)), snippets=snippets),
        model=config.VERIFIER_MODEL
    )
    return response if response in ["Supports", "Refutes", "Neutral"] else "Neutral"

def generate_answer(question: str, verified_evidence: list) -> str:
    """
    (回答器) 基于已验证的证据生成最终答案。
    """
    if not verified_evidence:
        return "抱歉，经过验证，我没有足够的信息来回答这个问题。"

    # 将已验证的事实格式化为清晰的上下文
    evidence_str = ""
    for fact in verified_evidence:
        evidence_str += f"- 事实: {fact['triple']}\\n"
        evidence_str += f"  证据: {fact['evidence'].replace('\\n', ' ')}\\n\\n"

    response = utils.call_llm(
        prompt=ANSWERER_PROMPT.format(question=question, verified_evidence=evidence_str),
        model=config.ANSWERER_MODEL
    )
    return response if response else "在生成最终答案时出错。"