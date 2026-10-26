"""Cost analytics: modeled prices and cost-per-correct-decision.

Prices are *modeled* (the benchmark runs on local CPU at $0 actual
spend): per-token prices proportional to hosted prices for this model
class, large = 3x small, prompt and completion tokens billed alike.
The headline metric is cost per correct decision = total cost of a
strategy over the suite / number of tasks it answered correctly.
"""

from __future__ import annotations

PRICE_PER_TOKEN = {  # USD per token
    "small": 0.20 / 1_000_000,  # Qwen2.5-0.5B class
    "large": 0.60 / 1_000_000,  # Qwen2.5-1.5B class (3x)
}


def call_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens + completion_tokens) * PRICE_PER_TOKEN[model]


def summarize(results: list[dict]) -> dict:
    """Aggregate per-task result dicts into strategy-level metrics."""
    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    total_cost = sum(r["cost_usd"] for r in results)
    total_tokens = sum(r["total_tokens"] for r in results)
    return {
        "tasks": n,
        "correct": correct,
        "accuracy": (correct / n) if n else 0.0,
        "total_cost_usd": total_cost,
        "cost_per_correct_decision_usd": (total_cost / correct) if correct else None,
        "total_tokens": total_tokens,
        "mean_tokens_per_task": (total_tokens / n) if n else 0.0,
        "escalations": sum(r.get("escalations", 0) for r in results),
        "borrows": sum(r.get("borrows", 0) for r in results),
        "early_exits": sum(1 for r in results if r.get("early_exit")),
        "budget_exhausted": sum(1 for r in results if r.get("budget_exhausted")),
    }
