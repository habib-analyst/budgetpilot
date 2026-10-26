"""Task-suite tests: determinism and independent gold re-derivation."""

import re

from budgetpilot.tasks import build_suite


def test_suite_deterministic():
    a = build_suite(42)
    b = build_suite(42)
    assert len(a) == len(b) == 72
    for ta, tb in zip(a, b):
        assert ta.task_id == tb.task_id
        assert ta.prompt == tb.prompt
        assert ta.gold == tb.gold
        assert [s.gold for s in ta.steps] == [s.gold for s in tb.steps]


def test_seed_changes_suite():
    a = build_suite(1)
    b = build_suite(2)
    assert any(ta.gold != tb.gold for ta, tb in zip(a, b))


def test_chain_golds_recompute_from_instructions():
    """Re-apply each step's operation (parsed from its instruction) to the
    running value and check it reproduces every step gold and the task gold."""
    for task in build_suite(42):
        if task.family != "chain":
            continue
        value = int(task.start_state)
        for step in task.steps:
            text = step.instruction
            if "add" in text:
                n = int(re.search(r"add (\d+)", text).group(1))
                value += n
            elif "subtract" in text:
                n = int(re.search(r"subtract (\d+)", text).group(1))
                value -= n
            elif "multiply" in text:
                n = int(re.search(r"multiply by (\d+)", text).group(1))
                value *= n
            elif "increase by" in text:
                p = int(re.search(r"increase by (\d+)%", text).group(1))
                assert (value * (100 + p)) % 100 == 0
                value = value * (100 + p) // 100
            elif "decrease by" in text:
                p = int(re.search(r"decrease by (\d+)%", text).group(1))
                assert (value * (100 - p)) % 100 == 0
                value = value * (100 - p) // 100
            else:  # pragma: no cover
                raise AssertionError(f"unparsed op in {text!r}")
            assert str(value) == step.gold
        assert str(value) == task.gold


def test_hop_golds_resolvable_from_facts():
    """Independently resolve each hop task from the printed fact table."""
    for task in build_suite(42):
        if task.family != "hop":
            continue
        lives = dict(re.findall(r"(\w+) lives in (\w+)\.", task.prompt))
        in_country = dict(re.findall(r"(\w+) is a city in (\w+)\.", task.prompt))
        capital = dict(re.findall(r"The capital of (\w+) is (\w+)\.", task.prompt))
        person = task.start_state
        city = lives[person]
        assert city == task.steps[0].gold
        if len(task.steps) >= 2:
            country = in_country[city]
            assert country == task.steps[1].gold
        if len(task.steps) >= 3:
            assert capital[country] == task.steps[2].gold
        assert task.gold == task.steps[-1].gold


def test_refine_golds():
    for task in build_suite(42):
        if task.family != "refine":
            continue
        m = re.search(r"What is (\d+) x (\d+)\?", task.prompt)
        if m:
            assert int(task.gold) == int(m.group(1)) * int(m.group(2))
            continue
        m = re.search(r"shop has (\d+) items\. (\d+)% of them are red\. (\d+)%", task.prompt)
        if m:
            base, p1, p2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
            assert int(task.gold) == base * p1 // 100 * p2 // 100
            continue
        m = re.search(r"What is ([\d \+]+)\?", task.prompt)
        assert m, task.prompt
        assert int(task.gold) == sum(int(x) for x in m.group(1).split("+"))
