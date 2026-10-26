"""Analytics and answer-parsing tests."""

from budgetpilot.analytics import PRICE_PER_TOKEN, call_cost, summarize
from budgetpilot.runner import parse_answer


def test_call_cost_ratio():
    c_small = call_cost("small", 900, 100)
    c_large = call_cost("large", 900, 100)
    assert abs(c_large - 3 * c_small) < 1e-15
    assert c_small == 1000 * PRICE_PER_TOKEN["small"]


def test_summarize_cost_per_correct():
    rows = [
        {"correct": True, "cost_usd": 0.002, "total_tokens": 100, "escalations": 1, "borrows": 0, "early_exit": True, "budget_exhausted": False},
        {"correct": False, "cost_usd": 0.004, "total_tokens": 200, "escalations": 0, "borrows": 2, "early_exit": False, "budget_exhausted": True},
        {"correct": True, "cost_usd": 0.003, "total_tokens": 150, "escalations": 0, "borrows": 0, "early_exit": False, "budget_exhausted": False},
    ]
    s = summarize(rows)
    assert s["tasks"] == 3 and s["correct"] == 2
    assert abs(s["cost_per_correct_decision_usd"] - 0.009 / 2) < 1e-12
    assert s["escalations"] == 1 and s["borrows"] == 2
    assert s["early_exits"] == 1 and s["budget_exhausted"] == 1


def test_summarize_no_correct():
    rows = [{"correct": False, "cost_usd": 0.001, "total_tokens": 10}]
    s = summarize(rows)
    assert s["cost_per_correct_decision_usd"] is None


def test_parse_numeric():
    assert parse_answer("42", "chain") == "42"
    assert parse_answer("The result is 1,234.", "chain") == "1234"
    assert parse_answer("step value: -17", "refine") == "-17"
    assert parse_answer("no digits here", "chain") is None
    assert parse_answer("", "chain") is None


def test_parse_entity():
    assert parse_answer("Aldermere", "hop") == "Aldermere"
    assert parse_answer("Aldermere.\n(extra words)", "hop") == "Aldermere"
    assert parse_answer("   ", "hop") is None
