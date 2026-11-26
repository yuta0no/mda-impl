"""Check the Claude client by hand: count proposer-prompt tokens, or make one real call.

    uv run python tools/llm_check.py count   # free: the official count_tokens endpoint
    uv run python tools/llm_check.py call    # one real call (under $0.01), cached under cache/

Both read ANTHROPIC_API_KEY from `.env`. The prompts are stand-ins rendered from the
`chem_propose.txt` template with realistic data, until `feat/proposer` adds the real renderer.
`count` measures the smallest prompt (the init call, 3 observations, no refining section) and
the largest (an expansion at the last step: 25 observations and a full pool of 24 forms).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from mda.bench import INPUT_NAMES, World, WorldSpec, from_unit
from mda.hypothesis import HYPOTHESES_SCHEMA, parse_hypotheses
from mda.library import SEED_LIBRARY
from mda.llm import ClaudeLLM, LLMClient

SYSTEM = (
    "You are an enzyme kineticist proposing candidate rate laws to explain assay data. "
    "Reply JSON only."
)
CONTEXT = (
    "An enzyme catalyses a reaction with initial rate r0 [mM/min]. Controllable inputs: "
    "C_A [substrate, mM], C_I [inhibitor, mM], C_B [2nd substrate, mM], C_P [product, mM], "
    "Enz [enzyme, mg/mL], T [K], pH. Discover an algebraic law "
    "r0 = f(C_A,C_I,C_B,C_P,Enz,T,pH; theta) from the data."
)
SPEC = (
    "Each 'expr' is a Python expression in the 7 input names + your declared params, using only "
    "+ - * / ** and exp, log, sqrt. Give physically plausible positive param bounds. JSON schema: "
    '{"hypotheses":[{"name":str,"expr":str,"params":[{"name":str,"low":float,"high":float}]}]}'
)
# The repository root: `cache/` and `runs/` are git-ignored there, wherever the tool is run from
ROOT = Path(__file__).resolve().parents[1]
WORLD = "c10_mm_competitive_arrhenius"
# Passive data D0 (spec §1): C_A in {0.03, 0.3, 3}, other inputs at (0, 1, 0, 1, 310, 7)
D0 = np.array([[a, 0.0, 1.0, 0.0, 1.0, 310.0, 7.0] for a in (0.03, 0.3, 3.0)])


def experiments(n: int) -> str:
    """`n` observations of `WORLD`: D0, then random conditions standing in for designed ones."""
    world = World(WorldSpec(WORLD))
    X = np.vstack([D0, from_unit(np.random.default_rng(0).uniform(size=(n - 3, 7)))])
    lines = []
    for x in X:
        inputs = ", ".join(f"{name}={v:.4g}" for name, v in zip(INPUT_NAMES, x, strict=True))
        lines.append(f"  {inputs} -> r0={world.observe(x):.6g}")
    return "\n".join(lines)


def pool_forms(n: int) -> list[str]:
    """`n` distinct forms: the seeds, then the seeds with an extra inhibition or Arrhenius factor."""
    seeds = [h.expr for h in SEED_LIBRARY]
    forms = seeds + [f"({e}) / (1 + C_I/Ki)" for e in seeds]
    forms += [f"({e}) * exp(-Ea/(8.314*T))" for e in seeds]
    return forms[:n]


def init_prompt() -> str:
    return "\n\n".join(
        [
            CONTEXT,
            "Experiments (inputs -> r0):\n" + experiments(3),
            "Propose 12 distinct plausible rate laws (or refinements of those tried).\n" + SPEC,
        ]
    )


def expand_prompt(n_obs: int = 25, n_forms: int = 24) -> str:
    rng = np.random.default_rng(1)
    residuals = "\n".join(
        f"  {e}  -- median relative residual {r:.3f}"
        for e, r in zip(pool_forms(n_forms), rng.uniform(0.05, 2.0, n_forms), strict=True)
    )
    refining = (
        "Forms ALREADY TRIED (in the pool) and their residuals -- do NOT re-propose any of these; "
        "propose forms STRUCTURALLY DIFFERENT from all of them:\n"
        + residuals
        + "\n  The best form's residual correlates most with: T, pH -- consider a factor in "
        "those variable(s).\n"
        "Refine by proposing a DIFFERENT algebraic form -- do NOT patch with ad-hoc "
        "offset/softening terms."
    )
    budget = (
        "Experiment budget remaining: 1. Current phase: explore (explore = restructure the form; "
        "refine = tune an adequate form)."
    )
    return "\n\n".join(
        [
            CONTEXT,
            "Experiments (inputs -> r0):\n" + experiments(n_obs),
            refining,
            budget,
            "Propose 6 distinct plausible rate laws (or refinements of those tried).\n" + SPEC,
        ]
    )


def count(llm: LLMClient) -> None:
    print(f"model: {llm.model}")
    print(f"{'prompt':<28} {'chars':>6} {'tokens':>7} {'+ schema':>9}")
    for name, user in [
        ("init (3 obs)", init_prompt()),
        ("expand (25 obs, 24 forms)", expand_prompt()),
    ]:
        plain = llm.count_tokens(SYSTEM, user)
        with_schema = llm.count_tokens(SYSTEM, user, schema=HYPOTHESES_SCHEMA)
        print(f"{name:<28} {len(SYSTEM) + len(user):>6} {plain:>7} {with_schema:>9}")


def call(llm: LLMClient) -> None:
    for attempt in ("first", "second"):
        reply = llm.complete(SYSTEM, init_prompt(), schema=HYPOTHESES_SCHEMA, call_id="smoke/init")
        print(
            f"{attempt} call: cached={reply.cached} stop={reply.stop_reason} "
            f"request_id={reply.request_id} usage={reply.usage} cost=${reply.cost:.5f}"
        )
    if not reply.ok:
        sys.exit(f"unusable reply: stop_reason={reply.stop_reason} detail={reply.stop_detail}")
    hypotheses, rejected = parse_hypotheses(reply.text)
    print(f"{len(hypotheses)} valid hypotheses, {len(rejected)} rejected")
    for h in hypotheses:
        print(f"  {h.name}: {h.expr}")
    for r in rejected:
        print(f"  rejected: {r}")
    print(f"sent {llm.n_sent} request(s), total cost ${llm.cost:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["count", "call"])
    args = parser.parse_args()
    load_dotenv()
    llm = ClaudeLLM(cache_dir=ROOT / "cache", transcript=ROOT / "runs/smoke/transcript.jsonl")
    {"count": count, "call": call}[args.command](llm)


if __name__ == "__main__":
    sys.exit(main())
