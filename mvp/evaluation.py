# evaluation.py (Improved Version)
import json
import time
from datetime import datetime
import agents
from main import run_pipeline
from rich.console import Console
from rich.table import Table
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress

console = Console()

def load_jsonl_test_suite(file_path: str):
    """Loads a test suite from a JSON Lines (.jsonl) file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        console.print(f"[bold red]Error: Test suite file not found at '{file_path}'[/bold red]")
        return []
    except json.JSONDecodeError as e:
        console.print(f"[bold red]Error decoding JSON in {file_path}: {e}[/bold red]")
        return []

def log_result(log_file, test_case, result, reasoning, success):
    """Appends the details of a single test case to the log file."""
    status_icon = "✅" if success else "❌"
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"### {status_icon} Test Case: {test_case['question']}\n")
        f.write(f"- **Ideal Answer:** `{test_case.get('answer', 'N/A')}`\n")
        f.write(f"- **Generated Answer:** `{result.get('final_answer', 'N/A')}`\n")
        f.write(f"- **Judge's Decision:** {'Correct' if success else 'Incorrect'}\n")
        f.write(f"- **Judge's Reasoning:** *{reasoning}*\n")
        f.write("- **Verified Facts Fed to Answerer:**\n")
        verified_facts = result.get('verified_facts', [])
        if verified_facts:
            for fact in verified_facts:
                f.write(f"  - `{fact.get('triple', 'N/A')}`\n")
        else:
            f.write("  - (No facts were verified)\n")
        f.write(f"- **Latency:** {result.get('latency', 0):.2f}s\n")
        f.write("\n---\n")

def run_single_test(test_case):
    """Executes the pipeline and evaluation for a single test case."""
    question = test_case.get("question")
    ideal_answer = test_case.get("answer")

    try:
        # Assumes run_pipeline will be modified to return a dictionary
        pipeline_result = run_pipeline(question, verbose=False)

        evaluation_result = agents.evaluate_answer(question, ideal_answer, pipeline_result['final_answer'])
        decision = evaluation_result.get("decision", "Incorrect")
        reasoning = evaluation_result.get("reasoning", "N/A")
        success = 1 if decision == "Correct" else 0

        return {
            "success": success,
            "latency": pipeline_result['latency'],
            "test_case": test_case,
            "pipeline_result": pipeline_result,
            "reasoning": reasoning
        }
    except Exception as e:
        # Handle unexpected errors during pipeline execution
        return {
            "success": 0,
            "latency": 0,
            "test_case": test_case,
            "pipeline_result": {"final_answer": "PIPELINE_ERROR", "verified_facts": []},
            "reasoning": f"An unexpected error occurred: {e}"
        }

def run_evaluation_framework(
    test_suite_path: str = "./data/hotpotqa_test_set.jsonl",
    log_file_path: str = "./data/evaluation_log.md",
    max_samples: int = None,
    max_workers: int = 5 # Number of parallel tests
):
    """Main function to run the full evaluation suite concurrently."""
    console.print("[bold cyan]Starting Quantitative Evaluation Framework...[/bold cyan]", justify="center")

    test_suite = load_jsonl_test_suite(test_suite_path)
    if not test_suite:
        return

    if max_samples and 0 < max_samples < len(test_suite):
        console.print(f"[yellow]Running on the first {max_samples} of {len(test_suite)} samples.[/yellow]")
        test_suite = test_suite[:max_samples]

    # Initialize log file for this run
    run_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(f"\n## Test Run: {run_timestamp}\n---\n")

    results = []
    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Running tests...", total=len(test_suite))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_case = {executor.submit(run_single_test, case): case for case in test_suite}

            for future in as_completed(future_to_case):
                result = future.result()
                results.append(result)
                log_result(log_file_path, result['test_case'], result['pipeline_result'], result['reasoning'], result['success'])
                progress.update(task, advance=1)

    if not results:
        console.print("[bold red]No results to display.[/bold red]")
        return

    # --- Display Summary Table ---
    table = Table(title=f"[bold]Evaluation Summary ({run_timestamp})[/bold]")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    avg_success_rate = np.mean([r['success'] for r in results]) * 100
    avg_latency = np.mean([r['latency'] for r in results])
    std_latency = np.std([r['latency'] for r in results])

    table.add_row("Total Samples", str(len(results)))
    table.add_row("Task Success Rate", f"{avg_success_rate:.2f}%")
    table.add_row("Average Latency", f"{avg_latency:.3f}s (±{std_latency:.3f}s)")

    # Log summary to file
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write("### Run Summary\n")
        f.write(f"- **Total Samples:** {len(results)}\n")
        f.write(f"- **Task Success Rate:** {avg_success_rate:.2f}%\n")
        f.write(f"- **Average Latency:** {avg_latency:.3f}s (±{std_latency:.3f}s)\n\n")

    console.print("\n")
    console.print(table)

if __name__ == "__main__":
    run_evaluation_framework(max_samples=10, max_workers=5)