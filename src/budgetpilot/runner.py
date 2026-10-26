"""Agent runner: executes multi-step tasks under three strategies.

- ``always_small`` / ``always_large``: every step (and every refinement
  round) on one fixed model with a fixed per-step completion cap.
  No budget controller, no escalation, no early exit.
- ``budgetpilot``: a per-task completion-token budget is allocated up
  front from a complexity estimate, sliced across steps, and managed
  while the task runs: steps start on the small model, low-confidence
  steps escalate to the large model (paid from the remaining pool),
  steps that overrun their slice borrow from future slices
  (reallocation), refinement rounds exit early once the answer is
  confident and stable, and the hard pool cap ends the task when the
  budget is exhausted.

The budget unit is completion tokens; cost accounting (analytics.py)
bills prompt + completion tokens at modeled prices.
"""

from __future__ import annotations

import re
import time

from . import budget as B
from .analytics import call_cost
from .tasks import Task

FIXED_CAP = 96  # per-step completion cap for the fixed strategies
MAX_ROUNDS = 3  # refinement rounds for the refine family

_NUM_RE = re.compile(r"-?\d[\d,]*")


def parse_answer(text: str, family: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    if family in ("chain", "refine"):
        nums = _NUM_RE.findall(text)
        if not nums:
            return None
        return nums[-1].replace(",", "")
    # hop: first line, stripped of trailing punctuation
    first = text.splitlines()[0].strip().rstrip(".").strip()
    return first or None


def _norm(ans: str | None, family: str) -> str:
    if ans is None:
        return ""
    if family in ("chain", "refine"):
        try:
            return str(int(ans))
        except ValueError:
            return ans
    return ans.casefold()


class _Ledger:
    """Accumulates cost/token/call statistics for one task run."""

    def __init__(self) -> None:
        self.cost = 0.0
        self.tokens = 0
        self.completion = 0
        self.calls = 0
        self.escalations = 0
        self.borrows = 0

    def add(self, model_name: str, res) -> None:
        self.cost += call_cost(model_name, res.prompt_tokens, res.completion_tokens)
        self.tokens += res.prompt_tokens + res.completion_tokens
        self.completion += res.completion_tokens
        self.calls += 1


def _run_steps_fixed(task: Task, model, model_name: str, led: _Ledger) -> str:
    state = task.start_state
    final = state
    for step in task.steps:
        prompt = task.prompt + "\n\n" + step.instruction.format(prev=state)
        res = model.generate(prompt, FIXED_CAP)
        led.add(model_name, res)
        ans = parse_answer(res.text, task.family)
        if ans is not None:
            state = ans
            final = ans
    return final


def _run_refine_fixed(task: Task, model, model_name: str, led: _Ledger) -> str:
    step = task.steps[0]
    prev = None
    final = ""
    for r in range(MAX_ROUNDS):
        prompt = task.prompt + "\n\n" + step.instruction
        if r > 0 and prev is not None:
            prompt += (
                f"\nA previous attempt gave {prev}. Recompute carefully "
                "from scratch before answering."
            )
        res = model.generate(prompt, FIXED_CAP)
        led.add(model_name, res)
        ans = parse_answer(res.text, task.family)
        if ans is not None:
            prev = ans
            final = ans
    return final


def _run_steps_budgetpilot(task: Task, models: dict, led: _Ledger) -> tuple[str, dict]:
    acct = B.BudgetAccount.allocate(task.features)
    state = task.start_state
    final = state
    exhausted = False
    for i, step in enumerate(task.steps):
        prompt = task.prompt + "\n\n" + step.instruction.format(prev=state)
        cap = min(acct.slice_for(i), acct.remaining)
        if cap <= 0:
            exhausted = True
            break
        res = models["small"].generate(prompt, cap)
        led.add("small", res)
        acct.charge(i, res.completion_tokens)
        ans = parse_answer(res.text, task.family)
        conf = res.confidence
        # overrun: unparseable -> borrow from future slices and retry once
        if ans is None:
            moved = acct.borrow_for(i)
            if moved > 0:
                led.borrows += 1
                cap2 = min(acct.slice_for(i), acct.remaining)
                if cap2 > 0:
                    res = models["small"].generate(prompt, cap2)
                    led.add("small", res)
                    acct.charge(i, res.completion_tokens)
                    ans = parse_answer(res.text, task.family)
                    conf = res.confidence
        # low confidence -> escalate this step to the large model
        if ans is not None and conf < B.ESCALATE_BELOW and acct.remaining >= B.MIN_SLICE:
            cap3 = min(acct.remaining, max(B.MIN_SLICE, acct.slices[i]))
            res_l = models["large"].generate(prompt, cap3)
            led.add("large", res_l)
            led.escalations += 1
            acct.charge(i, res_l.completion_tokens)
            ans_l = parse_answer(res_l.text, task.family)
            if ans_l is not None:
                ans = ans_l
                conf = res_l.confidence
        if acct.remaining <= 0 and i < len(task.steps) - 1:
            exhausted = True
        if ans is not None:
            state = ans
            final = ans
        acct.return_surplus(i)
    return final, {"early_exit": False, "budget_exhausted": exhausted}


def _run_refine_budgetpilot(task: Task, models: dict, led: _Ledger) -> tuple[str, dict]:
    feats = dict(task.features)
    feats["n_steps"] = MAX_ROUNDS  # one slice per refinement round
    acct = B.BudgetAccount.allocate(feats)
    step = task.steps[0]
    prev = None
    prev_conf = 0.0
    final = ""
    early = False
    exhausted = False
    for r in range(MAX_ROUNDS):
        cap = min(acct.slice_for(r), acct.remaining)
        if cap <= 0:
            exhausted = True
            break
        model_name = "large" if (r > 0 and prev_conf < B.ESCALATE_BELOW) else "small"
        if model_name == "large":
            led.escalations += 1
        prompt = task.prompt + "\n\n" + step.instruction
        if r > 0 and prev is not None:
            prompt += (
                f"\nA previous attempt gave {prev}. Recompute carefully "
                "from scratch before answering."
            )
        res = models[model_name].generate(prompt, cap)
        led.add(model_name, res)
        acct.charge(r, res.completion_tokens)
        ans = parse_answer(res.text, task.family)
        if ans is None:
            moved = acct.borrow_for(r)
            if moved > 0 and r < MAX_ROUNDS - 1:
                led.borrows += 1
            continue
        final = ans
        # early exit: confident, and (first round or stable vs previous)
        if res.confidence >= B.EXIT_ABOVE and (r == 0 or _norm(ans, task.family) == _norm(prev, task.family)):
            early = r < MAX_ROUNDS - 1
            prev, prev_conf = ans, res.confidence
            break
        prev, prev_conf = ans, res.confidence
        acct.return_surplus(r)
    return final, {"early_exit": early, "budget_exhausted": exhausted}


def run_task(task: Task, strategy: str, models: dict) -> dict:
    led = _Ledger()
    t0 = time.time()
    extra = {"early_exit": False, "budget_exhausted": False}
    if strategy == "budgetpilot":
        if task.family == "refine":
            final, extra = _run_refine_budgetpilot(task, models, led)
        else:
            final, extra = _run_steps_budgetpilot(task, models, led)
    elif strategy in ("always_small", "always_large"):
        name = "small" if strategy == "always_small" else "large"
        if task.family == "refine":
            final = _run_refine_fixed(task, models[name], name, led)
        else:
            final = _run_steps_fixed(task, models[name], name, led)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return {
        "task_id": task.task_id,
        "family": task.family,
        "strategy": strategy,
        "answer": final,
        "correct": _norm(final, task.family) == _norm(task.gold, task.family),
        "cost_usd": led.cost,
        "total_tokens": led.tokens,
        "completion_tokens": led.completion,
        "calls": led.calls,
        "escalations": led.escalations,
        "borrows": led.borrows,
        "early_exit": extra["early_exit"],
        "budget_exhausted": extra["budget_exhausted"],
        "wall_s": round(time.time() - t0, 2),
    }
