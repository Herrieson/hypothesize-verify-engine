# evaluation.py (Revised with sample limiting)
import json
import time
from datetime import datetime
import agents
from main import run_pipeline
from rich.console import Console
from rich.table import Table
import numpy as np

console = Console()

LOG_FILE = "./data/evaluation_log.md"

def load_jsonl_test_suite(file_path: str):
    """Loads a test suite from a JSON Lines (.jsonl) file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    except FileNotFoundError:
        console.print(f"[bold red]Error: Test suite file not found at '{file_path}'[/bold red]")
        return []
    except json.JSONDecodeError as e:
        console.print(f"[bold red]Error decoding JSON on a line in {file_path}: {e}[/bold red]")
        return []

def log_failure(test_case, result, reasoning):
    """Appends the details of a failed test case to the log file."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"### ❌ Failure: {test_case['question']}\n")
        f.write(f"- **Ideal Answer:** `{test_case.get('answer', 'N/A')}`\n")
        f.write(f"- **Generated Answer:** `{result.get('final_answer', 'N/A')}`\n")
        f.write(f"- **Judge's Reasoning:** *{reasoning}*\n")
        f.write("- **Verified Facts Fed to Answerer:**\n")
        verified_facts = result.get('verified_facts', [])
        if verified_facts:
            for fact in verified_facts:
                f.write(f"  - `{fact.get('triple', 'N/A')}`\n")
        else:
            f.write("  - (No facts were verified)\n")
        f.write("\n---\n")

# The new parameter `max_samples` is added here
def run_evaluation_framework(max_samples: int = None):
    """Main function to run the full evaluation suite."""
    console.print("[bold cyan]Starting Quantitative Evaluation Framework...[/bold cyan]", justify="center")
    
    test_suite_path = "./data/hotpotqa_test_set.jsonl"
    test_suite = load_jsonl_test_suite(test_suite_path)
    if not test_suite:
        return
    
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n## Test Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n---\n")

    if max_samples and max_samples > 0:
        console.print(f"[yellow]Running on the first {min(max_samples, len(test_suite))} of {len(test_suite)} samples.[/yellow]")
        test_suite = test_suite[:max_samples]

    results = []
    
    for i, test_case in enumerate(test_suite):
        console.print(f"\n--- Running Test Case {i+1}/{len(test_suite)} ---")
        question = test_case.get("question", "No question found in test case")
        console.print(f"[yellow]Question:[/yellow] {question}")
        
        pipeline_result = run_pipeline(question, verbose=False)
        ideal_answer = test_case.get("answer", "")
        
        # --- NEW: Use the LLM to evaluate the answer ---
        console.print("   - [purple]Asking LLM Judge for evaluation...[/purple]")
        evaluation_result = agents.evaluate_answer(question, ideal_answer, pipeline_result['final_answer'])
        decision = evaluation_result.get("decision", "Incorrect")
        reasoning = evaluation_result.get("reasoning", "N/A")
        success = 1 if decision == "Correct" else 0
        
        if not success:
            log_failure(test_case, pipeline_result, reasoning)
            
        results.append({
            "question": question,
            "success": success,
            "latency": pipeline_result['latency']
        })
        console.print(f"[green]Finished.[/green] Judge's Decision: {'✔ Correct' if success else '✖ Incorrect'}, Latency: {pipeline_result['latency']:.2f}s")

    if not results:
        console.print("[bold red]No results to display.[/bold red]")
        return
        
    table = Table(title="[bold]Evaluation Summary[/bold]")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Average", style="magenta")
    table.add_column("Std. Dev.", style="green")

    avg_success_rate = np.mean([r['success'] for r in results]) * 100
    avg_latency = np.mean([r['latency'] for r in results])
    std_latency = np.std([r['latency'] for r in results])

    table.add_row("Task Success Rate", f"{avg_success_rate:.2f}%", "---")
    table.add_row("Latency (seconds)", f"{avg_latency:.3f}s", f"{std_latency:.3f}s")
    
    console.print("\n")
    console.print(table)


if __name__ == "__main__":
    run_evaluation_framework(max_samples=10)