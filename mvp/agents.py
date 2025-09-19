# mvp/agents.py

import json
import config
import utils

# --- Prompt 模板 (Optimized) ---

HYPOTHESIZER_PROMPT = """
你是一位严谨的知识图谱构建专家。你的任务是分析一个问题，并分解成一个“假设性知识图谱”，这个图谱中的三元组（主体, 关系, 客体）必须是回答该问题所需的核心事实。

请遵循以下步骤：
1.  **思考**: 首先，在<thinking>标签中，逐步分析问题。确定需要知道哪些实体（人、地点、事物）的哪些属性或它们之间的关系才能推导出最终答案。
2.  **构建**: 基于你的思考，生成一个JSON对象。这个对象必须包含两个键：
    - "reasoning": 一个字符串，包含你在<thinking>标签中的完整分析过程。
    - "triples": 一个JSON格式的三元组列表 `[主体, 关系, 客体]`。

**重要规则**:
- 三元组应尽可能原子化和具体。
- 仅输出一个最终的、格式正确的JSON对象。

问题: "{question}"
"""

QUERY_GENERATOR_PROMPT = """
你是一位专业的搜索引擎查询构造师。你的任务是为一个知识三元组生成最多3个高效、多样化的Google搜索查询，用于事实核查。

**指导原则**:
- **多样性**: 创建不同角度的查询。例如，可以分别从主体和客体的角度提问。
- **简洁性**: 查询应简短精炼，直击要点。
- **避免是非题**: 不要生成简单的“是不是”或“对不对”的问题。

知识三元组: {triple}

请以JSON对象格式输出，其中包含一个名为 "queries" 的键，其值为一个字符串列表。
示例: {{"queries": ["维多利亚女王与阿尔伯特亲王的婚姻年份", "阿尔伯特亲王 妻子", "维多利亚女王 丈夫"]}}
"""

VERIFIER_PROMPT = """
你是一个极其精确和专注的事实核查员。你的唯一任务是判断提供的“证据摘要”是否明确支持或反驳给定的“知识三元组”。

**你的决策必须严格遵守以下规则**:
1.  **只依赖证据**: 你的判断必须*完全*基于所提供的“证据摘要”。绝对不能使用任何你的内部知识。
2.  **精确匹配**:
    - **Supports**: 摘要中必须有非常明确的陈述，直接证实该三元组。强烈的暗示也可以接受，但不能是猜测。
    - **Refutes**: 摘要中必须有非常明确的陈述，直接否定该三元组。
    - **Neutral**: 如果摘要与三元组无关、信息不充分、模棱两可，或者无法直接判断真伪，则必须回答Neutral。
3.  **单字输出**: 你的回答必须是单个词: `Supports`, `Refutes`, 或 `Neutral`。

知识三元组: {triple}

证据摘要:
---
{snippets}
---

你的单字回答:
"""

ANSWERER_PROMPT = """
你是一位智能且客观的问答机器人。你的任务是根据下面提供的“已验证的外部证据”，为用户的“原始问题”生成一个简洁、直接的最终答案。

**核心指令**:
- **忠于证据**: 你的答案必须*只能*从已验证的证据中推导出来。不要添加任何外部信息或个人知识。
- **直接回答**: 直接回答原始问题。
- **综合信息**: 如果有多条证据，请将它们综合起来形成一个连贯的答案。
- **承认不足**: 如果提供的证据不足以回答问题，请明确说明这一点，例如：“根据已验证的信息，无法确定...”。

原始问题: "{question}"

已验证的外部证据:
---
{verified_evidence}
---

基于以上所有信息，请为原始问题提供一个最终的、基于证据的答案。
"""

EVALUATOR_PROMPT = """
You are an expert evaluator for a question-answering system. Your task is to determine if the "Generated Answer" correctly and completely answers the "Original Question" based on the "Ideal Answer".

**Instructions**:
1.  Compare the semantic meaning of the "Generated Answer" and the "Ideal Answer".
2.  The generated answer does not need to be a verbatim match, but it must contain the same core information.
3.  Provide a step-by-step reasoning for your decision in a "reasoning" field.
4.  Output your final decision as either "Correct" or "Incorrect" in a "decision" field.

**JSON Output Format**:
{
  "reasoning": "Your step-by-step analysis here...",
  "decision": "Correct" or "Incorrect"
}

**Evaluation Task**:
- **Original Question**: "{question}"
- **Ideal Answer**: "{ideal_answer}"
- **Generated Answer**: "{generated_answer}"
"""

# --- 智能体定义 (无需修改函数逻辑) ---

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
        # 可选：打印或记录 data.get("reasoning") 用于调试
        # print(f"[Hypothesizer Reasoning]: {data.get('reasoning')}")
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
        evidence_str += f"- 事实: {fact['triple']}\n"
        evidence_str += f"  证据: {fact['evidence'].replace('\\n', ' ')}\n\n"

    response = utils.call_llm(
        prompt=ANSWERER_PROMPT.format(question=question, verified_evidence=evidence_str),
        model=config.ANSWERER_MODEL
    )
    return response if response else "在生成最终答案时出错。"

def evaluate_answer(question: str, ideal_answer: str, generated_answer: str) -> dict:
    """
    (Evaluator/Judge) Uses an LLM to determine if the generated answer matches the ideal answer.
    """
    raw_output = utils.call_llm(
        prompt=EVALUATOR_PROMPT.format(
            question=question,
            ideal_answer=ideal_answer,
            generated_answer=generated_answer
        ),
        model="gpt-4o-mini", # Use a reliable model for evaluation
        is_json=True
    )
    if not raw_output:
        return {"decision": "Incorrect", "reasoning": "Failed to get a response from the evaluator model."}
    try:
        return json.loads(raw_output)
    except (json.JSONDecodeError, AttributeError):
        return {"decision": "Incorrect", "reasoning": f"Failed to parse JSON from evaluator: {raw_output}"}