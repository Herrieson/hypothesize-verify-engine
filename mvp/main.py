# mvp/main.py

import time
import agents
import utils
from rich.console import Console
from rich.panel import Panel

# 初始化一个漂亮的打印控制台
console = Console()

def run_pipeline(question: str):
    """
    执行完整的“假设-验证”问答流水线。
    """
    console.print(Panel(f"[bold yellow]🤔 正在处理问题: [/bold yellow]{question}", title="[bold green]开始[/bold green]", border_style="green"))

    # --- 步骤 1: 假设 (Hypothesize) ---
    start_time = time.time()
    console.print("\\n[bold cyan]步骤 1: 生成假设性知识图谱...[/bold cyan]")
    hypothesis_graph = agents.generate_graph(question)
    console.print(f"   - [green]完成[/green] ({time.time() - start_time:.2f}秒)")

    if not hypothesis_graph:
        console.print("[bold red]无法生成假设图谱。流程终止。[/bold red]")
        return

    console.print("   - [bold]生成的假设:[/bold]")
    for triple in hypothesis_graph:
        console.print(f"     - {triple}")

    # --- 步骤 2: 验证 (Verify) ---
    console.print("\\n[bold cyan]步骤 2: 验证每个假设三元组...[/bold cyan]")
    verified_facts = []
    for triple in hypothesis_graph:
        start_time_triple = time.time()
        console.print(f"   - [bold]正在验证:[/bold] {triple}")

        queries = agents.generate_queries(triple)
        if not queries:
            console.print("     - [yellow]警告: 未能生成搜索查询。[/yellow]")
            continue

        console.print(f"     - 生成的查询: {queries}")

        is_verified = False
        evidence_for_triple = ""
        for query in queries:
            snippets = utils.execute_search(query)
            verification_result = agents.verify(triple, snippets)
            console.print(f"       - 查询 '{query}' -> 结果: [bold magenta]{verification_result}[/bold magenta]")

            if verification_result == "Supports":
                is_verified = True
                evidence_for_triple = snippets
                verified_facts.append({"triple": triple, "evidence": evidence_for_triple})
                break # 找到支持证据，跳出查询循环

        if is_verified:
            console.print(f"   - [green]✔ 已验证[/green] ({time.time() - start_time_triple:.2f}秒)")
        else:
            console.print(f"   - [red]✖ 未验证[/red] ({time.time() - start_time_triple:.2f}秒)")

    # --- 步骤 3: 回答 (Answer) ---
    start_time_answer = time.time()
    console.print("\\n[bold cyan]步骤 3: 基于已验证的证据生成最终答案...[/bold cyan]")
    final_answer = agents.generate_answer(question, verified_facts)
    console.print(f"   - [green]完成[/green] ({time.time() - start_time_answer:.2f}秒)")

    console.print(Panel(final_answer, title="[bold green]最终答案[/bold green]", border_style="green"))


if __name__ == "__main__":
    # 使用实验代码中的一个问题作为示例
    test_question = "斯科特·德瑞克森和艾德·伍德的国籍相同吗？"
    run_pipeline(test_question)