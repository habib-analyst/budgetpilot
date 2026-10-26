# budgetpilot: per-task token budgeting for multi-step agents

**Author:** Habib Ur Rehman
**Status:** measured results below are from `results.json` (seed 42); every
number in this document is taken from that file.

## The claim

Agent cost control is usually done per *request*: a router picks a model
for one call and caps its tokens. But an agent task is not a request — it
is a sequence of dependent steps, and what matters economically is the
cost of the whole task divided by whether the task's final answer was
right: **cost per correct decision**. budgetpilot treats the *task* as
the budget unit:

1. **Allocation.** Before the first model call, a cheap a-priori
   complexity estimate of the task spec (step count, operand digits,
   operation mix, fact-table size — no model call) is converted into a
   total completion-token budget for the whole task, sliced across
   steps by difficulty weight.
2. **Escalation.** Steps run on the small model first. A step whose
   answer confidence (exp of mean token logprob) falls below a fixed
   threshold is retried on the large model, paid from the task's
   remaining pool.
3. **Reallocation.** A step that overruns its slice (no parseable
   answer) borrows tokens from the slices of future steps; a step that
   underruns returns its surplus to them.
4. **Early exit.** On refinement tasks, a round whose answer is
   confident and stable versus the previous round ends the task
   immediately, returning the unspent budget.

All thresholds and allocation constants are fixed a priori in
`budget.py`; none were tuned on the benchmark suite.

## How this differs from my own earlier work (routewise, costlens)

- **routewise** routes *single, independent requests*: it classifies
  one request, assigns it a per-request token budget from a quality
  tier, and sends it to one backend. There is no task-level budget, no
  state carried across steps, no reallocation between steps, no early
  exit, and its metric is quality checks per request — not cost per
  correct decision over a multi-step task suite.
- **costlens** is post-hoc *accounting*: it attributes the cost of
  usage events that already happened and validates that attribution
  against ground truth. It makes no routing or budgeting decision and
  controls nothing at run time.

budgetpilot is the run-time control layer neither of them has: a budget
that lives for exactly one multi-step task, is allocated before the
task starts, and is actively re-managed (escalation, borrowing, early
exit) while the task runs.

## Relation to published prior art

Budget-aware agent execution is an active area; the closest works are
*Budget-Aware Agentic Routing via Boundary-Guided Training* (per-step
model selection under per-task spending limits, with an RL/SFT-trained
policy), *BudgetMLAgent* (a multi-agent ML-engineering system that
cascades to a paid expert model), and *Budget-Aware Value Tree Search*
(step-level budget-conditioned search inside a single model). budgetpilot
differs from all three in mechanism and setting: it is **training-free**
(no policy learning, no boundary trajectories), its budget is
**allocated up front from a complexity estimate** rather than only
imposed as a cap, it adds **cross-step reallocation and
confidence-based early exit**, and it is measured end-to-end on **local
CPU models at $0 API spend**. It does not claim to beat a trained
routing policy; it asks what a simple, explicit budget controller buys
you on small local models.

## Benchmark

- **Models (both local, CPU, bf16, greedy):** Qwen2.5-0.5B-Instruct
  ("small") and Qwen2.5-1.5B-Instruct ("large").
- **Suite:** 72 seeded, procedurally generated multi-step tasks with
  gold answers computed by the generator (and independently re-derived
  in the test suite): 36 arithmetic chains (2–4 dependent steps), 18
  relational lookup chains (2–3 hops over a printed fact table with
  distractors), 18 hard problems attacked in up to 3 refinement rounds.
- **Strategies:** always-small, always-large (fixed per-step completion
  cap of 96 tokens, all refinement rounds always run), and budgetpilot.
- **Cost model:** modeled prices, prompt + completion tokens billed:
  $0.20 / 1M tokens (small), $0.60 / 1M (large). Actual spend: $0 (local
  CPU). The budget unit is completion tokens.
- **Metric:** cost per correct decision = total modeled cost / correct
  final answers.

## Results

| Strategy | Correct | Accuracy | Total cost (modeled) | **Cost / correct decision** | Tokens | Calls |
|---|---|---|---|---|---|---|
| always-small | 10/72 | 13.9% | $0.007373 | $0.000737 | 36,867 | 207 |
| always-large | 30/72 | 41.7% | $0.021785 | **$0.000726** | 36,308 | 207 |
| budgetpilot | 20/72 | 27.8% | $0.021829 | $0.001091 | 58,918 | 325 |

Per family (correct / tasks, cost per correct decision):

| Family | always-small | always-large | budgetpilot |
|---|---|---|---|
| chain (36) | 2, $0.001394 | 18, $0.000461 | 13, $0.000742 |
| hop (18) | 6, $0.000550 | 11, $0.000900 | 7, $0.001383 |
| refine (18) | 2, $0.000642 | 1, $0.003583 | 0, — |

**Headline finding (negative for the controller as configured):**
budgetpilot doubled the small model's accuracy (10 → 20 correct;
head-to-head it rescued 14 tasks always-small failed and lost 4 that
always-small passed) but it spent the same total as always-large while
getting 10 fewer tasks right, so its cost per correct decision
($0.001091) is the worst of the three — 50% worse than always-large
($0.000726). always-large also edges always-small on the metric: its
3x accuracy gain slightly outpaced its 3x price.

**Why — three measured mechanism failures:**

1. **The confidence signal over-fires.** Escalation triggered 144
   times, in 70 of 72 tasks. Confidence is exp(mean logprob) over the
   *whole* generation; these models produce explanatory text around
   the answer, and the mean over all tokens lands below the 0.86
   threshold almost regardless of whether the answer is right. The
   controller degenerated into a "run both models" cascade that pays
   the small model's cost and then the large model's price on nearly
   every step.
2. **Prompt re-sends dominate cost at this scale.** budgetpilot used
   58,918 tokens of which only 2,907 were completions; 56,011 were
   prompt tokens re-sent per call (325 calls vs 207 for the fixed
   strategies). A budget denominated in completion tokens barely
   touches the real cost driver on short tasks; every escalation
   re-sends the full context (worst on `hop`, where the fact table is
   re-sent at large-model prices: budgetpilot $0.001383 vs always-large
   $0.000900 per correct answer).
3. **Two mechanisms never fired.** Slice borrowing (reallocation)
   triggered 0 times — every generation produced a parseable answer,
   so the overrun path was never taken. Early exit triggered 0 times —
   no refinement round ever reached the 0.93 confidence-and-stability
   bar. The budget pool was never exhausted (0 tasks). So of the four
   mechanisms, only allocation (as caps) and escalation were actually
   exercised by this suite, and escalation mis-fired.

**Design lessons the numbers support:** (i) confidence for escalation
should be computed over the answer span only, or from answer-token
logprobs, not the whole generation; (ii) a task budget should be
denominated in *total* tokens (prompt + completion), and an escalation
decision should be charged the prompt re-send it causes; (iii) an
escalation rule needs a cost-benefit bar (expected accuracy gain vs
certain added cost), not a confidence threshold alone. These are
conclusions from this measurement, not changes I made after seeing
the results — the constants in `budget.py` are the a-priori ones the
benchmark ran with.

## Honest limits

- The tasks are synthetic and short; real agent workloads have tool
  calls, retrieval, and much longer contexts, all of which change the
  economics (prompt tokens dominate there; here they are small).
- Confidence is exp(mean logprob) of the answer tokens — a crude
  signal, known to be miscalibrated; the fixed thresholds are design
  constants, not fitted values, and different thresholds would move the
  numbers.
- Two models of the same family at 0.5B/1.5B scale are weak reasoners;
  absolute accuracies are low and should not be extrapolated to
  frontier models.
- One seed, one machine, CPU timings are load-dependent; token counts
  and costs are exact, wall times are indicative only.
- budgetpilot's hard pool cap can end a task mid-chain
  (budget-exhausted tasks take their best-so-far state as the answer);
  on this suite that happened 0 times.
- Two of the four mechanisms — slice borrowing (reallocation) and
  early exit — triggered 0 times on this suite. They are implemented
  and covered by unit tests, but this benchmark provides **no
  evidence** about whether they help; the suite's generations were
  always parseable (no overruns to reallocate against) and never
  confident-and-stable (no early exits taken).
- Per-step intermediate answers are not separately scored in the
  headline metric; only final answers count, which is the economically
  relevant unit but hides where chains broke.
