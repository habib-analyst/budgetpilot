# budgetpilot

**Per-task token budgeting for multi-step LLM agents** — a budget that
lives for exactly one task: allocated up front from a complexity
estimate, spent across steps, reallocated when a step overruns, with
confidence-based escalation to a larger model and early exit when the
answer is confident enough. Measured the way that actually matters:
**cost per correct decision**.

## Why this exists

Routers budget *requests*. Agents run *tasks* — sequences of dependent
steps where an early cheap mistake compounds and a late expensive step
may be wasted. budgetpilot moves the budget unit from the request to
the task:

1. **Allocation** — before the first model call, a cheap a-priori
   complexity estimate of the task spec becomes a total completion-token
   budget, sliced across steps by difficulty.
2. **Escalation** — steps start on the small model; a low-confidence
   step is retried on the large model, paid from the task's pool.
3. **Reallocation** — a step that overruns its slice borrows from
   future steps' slices; a step that underruns returns its surplus.
4. **Early exit** — refinement rounds stop as soon as the answer is
   confident and stable.

This is the run-time control layer that neither of my earlier projects
has: `routewise` routes single independent requests with per-request
budgets, and `costlens` does post-hoc cost attribution. Full
differentiation and the relation to published prior art (Budget-Aware
Agentic Routing, BudgetMLAgent, BAVT) are in [PAPER.md](PAPER.md).

## Measured results (seed 42, `results.json`)

Models: Qwen2.5-0.5B-Instruct (small) and Qwen2.5-1.5B-Instruct
(large), both local on CPU (bf16, greedy). Costs are modeled at
$0.20 / 1M tokens (small) and $0.60 / 1M (large), prompt + completion
billed; actual spend was $0 (local CPU).

| Strategy | Correct | Accuracy | Total cost (modeled) | **Cost / correct decision** | Tokens | Calls |
|---|---|---|---|---|---|
| always-small | 10/72 | 13.9% | $0.007373 | $0.000737 | 36,867 | 207 |
| always-large | 30/72 | 41.7% | $0.021785 | **$0.000726** | 36,308 | 207 |
| budgetpilot | 20/72 | 27.8% | $0.021829 | $0.001091 | 58,918 | 325 |

**The honest headline: on this suite, budgetpilot loses on its own
metric.** It doubled the small model's accuracy (10 → 20 correct; it
rescued 14 tasks the small model failed) but spent as much as
always-large while answering 10 fewer tasks correctly, giving it the
worst cost per correct decision. The mechanism-level diagnosis is in
[PAPER.md](PAPER.md): the logprob confidence signal fired on 144 steps
across 70 of 72 tasks, turning the controller into a pay-both cascade;
prompt re-sends (56,011 of budgetpilot's 58,918 tokens) dominate cost
at this task scale; and two mechanisms — slice borrowing and early
exit — never triggered at all on this suite. The per-family breakdown
and the design lessons are in PAPER.md.

## Quickstart

```bash
pip install -e .
# download the two models (any HF mirror / huggingface-cli works)
python -m budgetpilot.bench \
  --small-path /path/to/Qwen2.5-0.5B-Instruct \
  --large-path /path/to/Qwen2.5-1.5B-Instruct \
  --seed 42 --out results.json
pytest  # model-free tests: budget accounting, task golds, analytics
```

## Layout

| Module | Role |
|---|---|
| `budgetpilot/tasks.py` | Seeded generators for the 3 multi-step task families; golds computed by construction |
| `budgetpilot/budget.py` | Complexity estimate, per-task allocation, slice borrowing / surplus return, thresholds |
| `budgetpilot/runner.py` | Agent runner: fixed strategies vs the budgetpilot controller |
| `budgetpilot/models.py` | Local HF wrapper: greedy generation, token counts, answer confidence from logprobs |
| `budgetpilot/analytics.py` | Modeled prices, cost-per-correct-decision aggregation |
| `budgetpilot/bench.py` | Benchmark CLI → `results.json` |

## Honest limits

Synthetic short tasks; a crude logprob confidence signal; two small
same-family models; one seed. Absolute accuracies are low and should
not be extrapolated to frontier models. Details in
[PAPER.md](PAPER.md#honest-limits).

MIT license. Author: Habib Ur Rehman.
