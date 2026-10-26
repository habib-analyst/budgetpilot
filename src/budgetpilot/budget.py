"""Per-task token budgeting for multi-step agent runs.

The controller has four mechanisms, all operating on a *task-level* budget
(unlike a per-request cap):

1. **Allocation** — before the first model call, a cheap a-priori
   complexity estimate (features of the task spec, no model call) is
   turned into a total token budget for the whole task.
2. **Slicing** — the total is divided into per-step slices proportional
   to per-step difficulty weights.
3. **Reallocation** — when a step overruns its slice, the shortfall is
   borrowed from the slices of the steps that remain (they shrink
   proportionally); when a step underruns, the surplus flows back into
   the remaining pool.
4. **Escalation & early exit** — the runner consults this module's
   thresholds: a low-confidence step result may be retried on the large
   model (paid from the task pool), and a high-confidence stable result
   on a refinement task ends the task early, returning the unspent
   budget.

All accounting is in tokens; prices live in :mod:`budgetpilot.analytics`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- policy constants (fixed a priori, not tuned on the test suite) ------
BASE_TOKENS = 40  # fixed overhead allowance per task
PER_STEP_TOKENS = 46  # allowance per step at difficulty weight 1.0
DIGIT_TOKENS = 6  # extra per digit of the largest operand
MUL_TOKENS = 26  # extra per multiplication step
PCT_TOKENS = 18  # extra per percentage step
FACT_TOKENS = 1  # per fact in a lookup table (reading cost)

ESCALATE_BELOW = 0.86  # step confidence under this -> escalate if pool allows
EXIT_ABOVE = 0.93  # refinement round accepted early at/above this confidence
MIN_SLICE = 16  # a step never gets fewer tokens than this
BORROW_CAP_FRAC = 0.5  # a step may borrow at most this fraction of its slice extra


def step_weight(features: dict, step_index: int, n_steps: int) -> float:
    """Difficulty weight of one step, from task features (heuristic, fixed)."""
    w = 1.0
    w += 0.35 * max(0, features.get("max_digits", 1) - 2)
    if features.get("n_mul", 0):
        w += 0.5 * features["n_mul"] / max(1, n_steps)
    if features.get("n_pct", 0):
        w += 0.35 * features["n_pct"] / max(1, n_steps)
    return w


def estimate_budget(features: dict) -> int:
    """A-priori total token budget for a task, from its spec features."""
    n = max(1, features.get("n_steps", 1))
    b = BASE_TOKENS + PER_STEP_TOKENS * n
    b += DIGIT_TOKENS * max(0, features.get("max_digits", 1) - 1)
    b += MUL_TOKENS * features.get("n_mul", 0)
    b += PCT_TOKENS * features.get("n_pct", 0)
    b += FACT_TOKENS * features.get("n_facts", 0)
    return int(b)


@dataclass
class BudgetAccount:
    """A task-level token account with per-step slices and borrowing."""

    total: int
    slices: list[int]
    spent: list[int] = field(default_factory=list)
    borrowed: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.spent = [0] * len(self.slices)
        self.borrowed = [0] * len(self.slices)

    @classmethod
    def allocate(cls, features: dict) -> "BudgetAccount":
        total = estimate_budget(features)
        n = max(1, features.get("n_steps", 1))
        weights = [step_weight(features, i, n) for i in range(n)]
        wsum = sum(weights)
        slices = [max(MIN_SLICE, int(total * w / wsum)) for w in weights]
        # correct rounding drift on the last slice
        drift = total - sum(slices)
        slices[-1] = max(MIN_SLICE, slices[-1] + drift)
        return cls(total=total, slices=slices)

    # -- accounting ---------------------------------------------------------
    @property
    def spent_total(self) -> int:
        return sum(self.spent)

    @property
    def remaining(self) -> int:
        return self.total - self.spent_total

    def slice_for(self, step: int) -> int:
        """Current token allowance for a step (after prior reallocations)."""
        base = self.slices[step] - self.spent[step]
        return max(0, base)

    def charge(self, step: int, tokens: int) -> None:
        self.spent[step] += tokens

    def borrow_for(self, step: int) -> int:
        """Move tokens from future slices into this step's slice.

        Returns the amount actually moved (0 if nothing left to borrow).
        Borrowing is capped at BORROW_CAP_FRAC of the step's original
        slice, and future slices never drop below MIN_SLICE.
        """
        cap = int(self.slices[step] * BORROW_CAP_FRAC)
        want = cap - self.borrowed[step]
        if want <= 0:
            return 0
        moved = 0
        for j in range(step + 1, len(self.slices)):
            spare = self.slices[j] - self.spent[j] - MIN_SLICE
            if spare <= 0:
                continue
            take = min(spare, want - moved)
            self.slices[j] -= take
            self.slices[step] += take
            moved += take
            if moved >= want:
                break
        self.borrowed[step] += moved
        return moved

    def return_surplus(self, step: int) -> None:
        """Unspent slice of a finished step flows back to later slices."""
        surplus = self.slices[step] - self.spent[step]
        if surplus <= 0:
            return
        later = list(range(step + 1, len(self.slices)))
        if not later:
            return
        share = surplus // len(later)
        for j in later:
            self.slices[j] += share
        self.slices[step] -= share * len(later)
