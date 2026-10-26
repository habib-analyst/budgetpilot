"""Seeded generators for multi-step tasks with known (gold) answers.

Three families, each genuinely multi-step (the runner makes one model
call per step, carrying the agent's own state forward):

- ``chain``  : a k-step arithmetic chain. Step i applies one operation to
               the running value. Gold = exact integer after all steps.
- ``hop``    : a 2-3 hop relational lookup over a synthetic fact table
               printed in the prompt (with distractors). Step i resolves
               one relation. Gold = final entity string.
- ``refine`` : one hard problem (big multiplication, percentage-of-
               percentage, long sum) attacked in up to R independent
               refinement rounds; each round is a fresh attempt whose
               answer the controller may accept (early exit) or reject.

Step instructions may contain ``{prev}``, replaced by the runner with
the agent's own previous answer (never the gold). Gold answers are
computed by the generator itself with plain Python arithmetic / table
lookups, so tests can re-derive them independently. All generation is
deterministic given the seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Step:
    """One model call in a task."""

    instruction: str  # may contain {prev}
    gold: str  # gold value produced by this step (final step = task gold)


@dataclass
class Task:
    task_id: str
    family: str  # "chain" | "hop" | "refine"
    prompt: str  # full problem statement shown at every step
    steps: list[Step]
    gold: str  # final gold answer
    start_state: str = ""  # agent-visible initial state (chain: start value)
    features: dict = field(default_factory=dict)  # a-priori complexity features


# ---------------------------------------------------------------- chain ---

_OPS = ["add", "sub", "mul", "pct_up", "pct_down"]


def _gen_chain(rng: random.Random, task_id: str, k: int) -> Task:
    value = rng.randint(12, 240)
    start = value
    lines = [f"Start with the number {start}."]
    steps: list[Step] = []
    n_mul = n_pct = 0
    for i in range(k):
        op = rng.choice(_OPS)
        if op == "add":
            n = rng.randint(7, 180)
            value += n
            text = f"add {n}"
        elif op == "sub":
            n = rng.randint(5, 150)
            value -= n
            text = f"subtract {n}"
        elif op == "mul":
            n = rng.randint(2, 9)
            value *= n
            text = f"multiply by {n}"
            n_mul += 1
        elif op == "pct_up":
            p = rng.choice([10, 20, 25, 50])
            if (value * (100 + p)) % 100 == 0:
                value = value * (100 + p) // 100
                text = f"increase by {p}%"
                n_pct += 1
            else:
                n = rng.randint(7, 180)
                value += n
                text = f"add {n}"
        else:  # pct_down
            p = rng.choice([10, 20, 25, 50])
            if (value * (100 - p)) % 100 == 0:
                value = value * (100 - p) // 100
                text = f"decrease by {p}%"
                n_pct += 1
            else:
                n = rng.randint(5, 150)
                value -= n
                text = f"subtract {n}"
        lines.append(f"Step {i + 1}: {text}.")
        steps.append(
            Step(
                instruction=(
                    "The current value is {prev}. "
                    f"Apply this operation: {text}. "
                    "Reply with only the resulting number, nothing else."
                ),
                gold=str(value),
            )
        )
    prompt = (
        "Solve this multi-step calculation one step at a time.\n"
        + "\n".join(lines)
        + "\nYou will be asked for one step at a time. Keep every result exact."
    )
    feats = {
        "n_steps": k,
        "max_digits": max(len(str(abs(start))), len(str(abs(value)))),
        "n_mul": n_mul,
        "n_pct": n_pct,
    }
    return Task(task_id, "chain", prompt, steps, str(value), str(start), feats)


# ------------------------------------------------------------------ hop ---

_PEOPLE = [
    "Amara", "Boris", "Chen", "Dalia", "Emil", "Farah", "Goran", "Hana",
    "Ilya", "Jana", "Kofi", "Lena", "Milo", "Nadia", "Omar", "Priya",
    "Ravi", "Sana", "Tomas", "Uma", "Vera", "Wend", "Yusuf", "Zara",
]
_CITIES = [
    "Alder", "Bryn", "Corvik", "Dunmore", "Elston", "Falken", "Granton",
    "Havill", "Iskra", "Jorvik", "Kestrel", "Lormont", "Maren", "Norvik",
    "Ostend", "Pelham", "Quill", "Ravens", "Sorrel", "Talvik",
]
_COUNTRIES = [
    "Aldania", "Brevia", "Corvania", "Dalmora", "Estovar", "Falkia",
    "Grenmark", "Havania", "Istravia", "Jorland",
]
_CAPITALS = [
    "Aldermere", "Brynhold", "Corvane", "Dunwick", "Elmere", "Falkirk",
    "Granmere", "Haverton", "Iskane", "Jornhold",
]
_COMPANIES = [
    "Northbeam", "Quantumleaf", "Brightforge", "Silverpine", "Cloudharbor",
    "Datawillow", "Ironpeak", "Lumenfield", "Novacrest", "Omniroot",
]


def _gen_hop(rng: random.Random, task_id: str, hops: int) -> Task:
    people = rng.sample(_PEOPLE, 8)
    cities = rng.sample(_CITIES, 8)
    capitals = rng.sample(_CAPITALS, 8)
    companies = rng.sample(_COMPANIES, 6)

    city_of = {p: c for p, c in zip(people, cities)}
    country_of = {c: n for c, n in zip(cities, rng.sample(_COUNTRIES * 2, 8))}
    capital_of = {
        n: capitals[i] for i, n in enumerate(dict.fromkeys(country_of.values()))
    }
    hq_of = {c: rng.choice(companies) for c in cities}

    p0 = people[0]
    chain_entities = [p0, city_of[p0]]
    if hops >= 2:
        chain_entities.append(country_of[city_of[p0]])
    if hops >= 3:
        chain_entities.append(capital_of[country_of[city_of[p0]]])

    facts: list[str] = []
    for p in people:
        facts.append(f"{p} lives in {city_of[p]}.")
    for c in cities:
        facts.append(f"{c} is a city in {country_of[c]}.")
        facts.append(f"The headquarters of {hq_of[c]} is in {c}.")
    for n, cap in capital_of.items():
        facts.append(f"The capital of {n} is {cap}.")
    rng.shuffle(facts)

    steps: list[Step] = []
    for i in range(hops):
        nxt = chain_entities[i + 1]
        if i == 0:
            instr = (
                f"From the facts, find the city where {p0} lives. "
                "Reply with only the city name, nothing else."
            )
        elif i == 1:
            instr = (
                "The previous answer was {prev}. From the facts, find the "
                "country that {prev} is in. "
                "Reply with only the country name, nothing else."
            )
        else:
            instr = (
                "The previous answer was {prev}. From the facts, find the "
                "capital of {prev}. "
                "Reply with only the capital name, nothing else."
            )
        steps.append(Step(instruction=instr, gold=nxt))

    q = {
        1: f"In which city does {p0} live?",
        2: f"{p0} lives somewhere. In which country is that city?",
        3: f"{p0} lives somewhere. What is the capital of the country that city is in?",
    }[hops]
    prompt = "Facts:\n" + "\n".join(facts) + f"\nQuestion: {q}"
    feats = {"n_steps": hops, "n_facts": len(facts), "max_digits": 1, "n_mul": 0, "n_pct": 0}
    return Task(task_id, "hop", prompt, steps, chain_entities[-1], p0, feats)


# --------------------------------------------------------------- refine ---

def _gen_refine(rng: random.Random, task_id: str, kind: int) -> Task:
    if kind == 0:  # 3-digit x 2-digit multiplication
        a, b = rng.randint(123, 989), rng.randint(12, 98)
        problem = f"What is {a} x {b}?"
        gold = str(a * b)
        digits = len(str(a)) + len(str(b))
        n_mul, n_pct = 1, 0
    elif kind == 1:  # percentage of a percentage
        base = rng.choice([200, 240, 300, 400, 480, 500, 600, 800])
        p1 = rng.choice([10, 20, 25, 50, 75])
        p2 = rng.choice([10, 20, 25, 40, 50])
        problem = (
            f"A shop has {base} items. {p1}% of them are red. "
            f"{p2}% of the red ones are large. How many large red items are there?"
        )
        gold = str(base * p1 // 100 * p2 // 100)
        digits = 5
        n_mul, n_pct = 0, 2
    else:  # long sum with carries
        nums = [rng.randint(137, 989) for _ in range(5)]
        problem = "What is " + " + ".join(str(n) for n in nums) + "?"
        gold = str(sum(nums))
        digits = 4
        n_mul, n_pct = 0, 0
    step = Step(
        instruction=(
            "Solve the problem carefully, checking your arithmetic. "
            "Reply with only the final number, nothing else."
        ),
        gold=gold,
    )
    feats = {"n_steps": 1, "max_digits": digits, "n_mul": n_mul, "n_pct": n_pct}
    return Task(task_id, "refine", problem, [step], gold, "", feats)


# ---------------------------------------------------------------- suite ---

def build_suite(seed: int, n_chain: int = 36, n_hop: int = 18, n_refine: int = 18) -> list[Task]:
    """Deterministic suite. Default size: 72 tasks."""
    rng = random.Random(seed)
    tasks: list[Task] = []
    for i in range(n_chain):
        k = 2 + (i % 3)  # 2, 3, 4 steps balanced
        tasks.append(_gen_chain(rng, f"chain-{i:02d}", k))
    for i in range(n_hop):
        hops = 2 + (i % 2)  # 2 or 3 hops
        tasks.append(_gen_hop(rng, f"hop-{i:02d}", hops))
    for i in range(n_refine):
        tasks.append(_gen_refine(rng, f"refine-{i:02d}", i % 3))
    return tasks
