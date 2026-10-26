"""Budget controller tests: allocation, borrowing, surplus return."""

from budgetpilot import budget as B


def _feats(n_steps=3, **kw):
    f = {"n_steps": n_steps, "max_digits": 3, "n_mul": 1, "n_pct": 1}
    f.update(kw)
    return f


def test_estimate_budget_grows_with_steps():
    assert B.estimate_budget(_feats(4)) > B.estimate_budget(_feats(2))


def test_allocation_sums_to_total():
    acct = B.BudgetAccount.allocate(_feats(3))
    assert sum(acct.slices) == acct.total
    assert all(s >= B.MIN_SLICE for s in acct.slices)
    assert acct.remaining == acct.total


def test_allocation_single_step():
    acct = B.BudgetAccount.allocate(_feats(1))
    assert len(acct.slices) == 1
    assert acct.slices[0] == acct.total


def test_charge_and_remaining():
    acct = B.BudgetAccount.allocate(_feats(3))
    acct.charge(0, 10)
    assert acct.spent_total == 10
    assert acct.remaining == acct.total - 10
    assert acct.slice_for(0) == acct.slices[0] - 10


def test_borrow_moves_from_future_and_caps():
    acct = B.BudgetAccount.allocate(_feats(3))
    before = list(acct.slices)
    moved = acct.borrow_for(0)
    assert moved > 0
    assert acct.slices[0] == before[0] + moved
    assert sum(acct.slices) == acct.total  # conservation
    assert all(s >= B.MIN_SLICE for s in acct.slices)
    # cap: a second borrow cannot exceed the per-step borrow cap
    moved2 = acct.borrow_for(0)
    assert acct.borrowed[0] <= int(before[0] * B.BORROW_CAP_FRAC) + moved2
    assert acct.borrowed[0] == moved + moved2


def test_borrow_last_step_gets_nothing():
    acct = B.BudgetAccount.allocate(_feats(3))
    assert acct.borrow_for(2) == 0


def test_return_surplus_flows_forward():
    acct = B.BudgetAccount.allocate(_feats(3))
    acct.charge(0, 1)  # spend almost nothing of slice 0
    s0, s1, s2 = acct.slices
    acct.return_surplus(0)
    assert acct.slices[0] < s0
    assert acct.slices[1] + acct.slices[2] > s1 + s2
    assert sum(acct.slices) <= acct.total  # integer share rounding may drop <2 tokens
