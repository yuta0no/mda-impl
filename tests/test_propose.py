import itertools
import json
import math
import re
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any

import jaxtyping
import numpy as np
import pytest

import mda
from mda.agent import MDAAgent
from mda.bench import INPUT_NAMES, from_unit
from mda.config import DEFAULTS, MDAConfig
from mda.fit import FitResult, relative_residuals
from mda.hypothesis import (
    HYPOTHESES_SCHEMA,
    Hypothesis,
    Param,
    compile_hypothesis,
    normalise_expr,
)
from mda.library import SEED_LIBRARY
from mda.llm import ClaudeLLM, FakeLLM, LLMError, Reply, Request, StopReason, Usage
from mda.pool import Candidate, Origin
from mda.propose import (
    SYSTEM_PROMPT,
    CallRecord,
    LLMProposer,
    Proposal,
    Proposer,
    _load_template,
    claude_proposer,
    correlated_input,
    map_index,
    render_prompt,
)
from mda.run import RunLog, run_cell

# The template, read here without the module's parser: the prompts are checked against the file
TEMPLATE = (Path(mda.__file__).parent / "prompts" / "chem_propose.txt").read_text().splitlines()
TEMPLATE_TEXT = "\n".join(TEMPLATE)

PREFIX = "hybrid/c0_michaelis_menten/s0"
MM, COMPETITIVE = SEED_LIBRARY[0], SEED_LIBRARY[1]
CA, CI, TEMP, PH = (INPUT_NAMES.index(n) for n in ("C_A", "C_I", "T", "pH"))

# Passive data D0 (spec §1), then two conditions with more digits than the prompt shows
D0 = np.array([[a, 0.0, 1.0, 0.0, 1.0, 310.0, 7.0] for a in (0.03, 0.3, 3.0)])
X5 = np.vstack(
    [
        D0,
        [17.6312, 49.871, 0.11102, 4.97412, 1.60398, 311.284, 6.00712],
        [0.116712, 36.9703, 0.0122041, 6.56519, 0.996817, 336.418, 9.20431],
    ]
)
Y5 = np.array([0.031046, 0.30731, 2.2648, 0.54187, 0.022163])


def law(name: str, expr: str, *params: str) -> Hypothesis:
    return Hypothesis(name, expr, tuple(Param(p, 1e-3, 1e3) for p in params))


def power(p: int) -> Hypothesis:
    """Distinct one-parameter laws, none of which fits Michaelis-Menten data."""
    return law(f"power_{p}", f"k * C_A ** {p}", "k")


# A law of the LLM's, written over two lines with a comment
SQUARED = law("squared", "kcat * Enz * C_A  # saturating\n  / (Km + C_A) ** 2", "kcat", "Km")
BROKEN = law("broken", "k * log(C_I - 100)", "k")  # C_I <= 50: no finite prediction anywhere


def fitted(
    h: Hypothesis,
    rel: list[float] | None,
    X: np.ndarray = X5,
    y: np.ndarray = Y5,
    origin: Origin = Origin.SEED,
) -> Candidate:
    """A candidate whose fit predicts y * (1 - rel), or nothing finite when `rel` is None.

    Its `fit_err` is computed from the prediction, as `Pool.refit` does it.
    """
    theta = np.array([p.low for p in h.params])
    if rel is None:
        y_hat = np.full(len(y), np.nan)
        fit, err = FitResult(theta, math.inf, y_hat, False), math.inf
    else:
        y_hat = y * (1 - np.array(rel))
        fit = FitResult(theta, 1.0, y_hat, True)
        err = float(np.median(relative_residuals(y, y_hat)))
    return Candidate(h, compile_hypothesis(h), normalise_expr(h.expr), origin, 3, fit, err)


# The pool of the expansion snapshot, in pool order. The competitive-inhibition seed has the largest
# weight but not the smallest residual, and its residual correlates most with C_I (Spearman 0.89).
# The other two fits tie for their input: Michaelis-Menten's residual correlates equally with Enz
# (-0.67) and pH (0.67), and the squared law's with C_B (0.67), C_P and T (-0.67); the ties go to the
# earlier input in INPUT_NAMES. The weights have a tie at 0 and are not in pool order
POOL = (
    fitted(MM, [0.01, -0.02, 0.03, -0.04, 0.05]),
    fitted(COMPETITIVE, [0.08, -0.08, 0.0, 0.2, 0.5]),
    fitted(SQUARED, [-0.5, 0.4, 0.6, 0.2, -0.9], origin=Origin.INIT),
    fitted(BROKEN, None, origin=Origin.INIT),
)
WEIGHTS = np.array([0.25, 0.7, 0.05, 0.0])


def expand_prompt(
    weights: np.ndarray | None = WEIGHTS, budget_left: int = 19, n: int = 6, **kwargs: Any
) -> str:
    """The expansion prompt of the snapshot, or a variant of it."""
    return render_prompt(
        "expand", POOL, X5, Y5, weights=weights, budget_left=budget_left, n=n, **kwargs
    )


def init_prompt(n: int = 12) -> str:
    """The init prompt of the snapshot, or one asking for another number of laws."""
    return render_prompt("init", (), D0, Y5[:3], weights=None, budget_left=22, n=n)


def reply_text(*hypotheses: Hypothesis) -> str:
    """A reply in the format of the template."""
    items = [
        {
            "name": h.name,
            "expr": h.expr,
            "params": [{"name": p.name, "low": p.low, "high": p.high} for p in h.params],
        }
        for h in hypotheses
    ]
    return json.dumps({"hypotheses": items})


def reply(
    text: str = "", stop: StopReason = StopReason.END_TURN, request_id: str | None = None
) -> Reply:
    return Reply(text, stop, Usage(), "fake", request_id=request_id)


def laws(*hypotheses: Hypothesis, request_id: str | None = None) -> Reply:
    return reply(reply_text(*hypotheses), request_id=request_id)


def cut(request_id: str | None = None) -> Reply:
    return reply('{"hypotheses": [{"name": "a", "ex', StopReason.MAX_TOKENS, request_id)


def proposer(script, config: MDAConfig = DEFAULTS, **llm_kwargs) -> LLMProposer:
    return LLMProposer(FakeLLM(script, **llm_kwargs), call_prefix=PREFIX, config=config)


def fake(p: LLMProposer) -> FakeLLM:
    assert isinstance(p.llm, FakeLLM)
    return p.llm


def init(p: LLMProposer, pool=(), X=D0, y=Y5[:3]) -> Proposal:
    return p.propose("init", pool, X, y, weights=None, budget_left=22)


def expand(p: LLMProposer, pool=POOL, weights=WEIGHTS, budget_left: int = 19) -> Proposal:
    return p.propose("expand", pool, X5, Y5, weights=weights, budget_left=budget_left)


def call_ids(path: Path) -> list[str]:
    """The call IDs of a transcript, one per call that reached the client."""
    return [json.loads(line)["call_id"] for line in path.read_text().splitlines()]


def unindented(prompt: str) -> list[str]:
    """The prompt's lines from the template: not the experiments or the forms, which are indented."""
    return [line for line in prompt.split("\n") if line and not line.startswith("  ")]


def template_part(marker: str) -> list[str]:
    """The sent lines of the template part that the line starting with `marker` introduces."""
    i = next(k for k, line in enumerate(TEMPLATE) if line.startswith(marker))
    return [line for line in TEMPLATE[i + 1 : TEMPLATE.index("", i)] if not line.startswith("  <")]


# The template's user part is everything after the SYSTEM block and the line describing the parts.
# Of it, the lines in square brackets say when a part is present and the indented ones in angle
# brackets stand for content; the other lines are sent as they are, with their placeholders filled
SENT = [
    line
    for line in TEMPLATE[TEMPLATE.index("") + 2 :]
    if line and not line.startswith("[") and not line.startswith("  <")
]
REFINING = template_part("[when refining")
STATUS = template_part("[budget/phase")


def pattern(line: str) -> str:
    """A template line as a regular expression, its placeholders matching what fills them."""
    p = re.escape(line)
    p = p.replace(re.escape("<B>"), r"(\d+)")
    p = p.replace(re.escape("<explore|refine>"), "(explore|refine)")
    return p.replace(re.escape("Propose N "), re.escape("Propose ") + r"(\d+)\ ")


INIT_SNAPSHOT = """\
An enzyme catalyses a reaction with initial rate r0 [mM/min]. Controllable inputs: C_A [substrate, mM], C_I [inhibitor, mM], C_B [2nd substrate, mM], C_P [product, mM], Enz [enzyme, mg/mL], T [K], pH. Discover an algebraic law r0 = f(C_A,C_I,C_B,C_P,Enz,T,pH; theta) from the data.

Experiments (inputs -> r0):
  C_A=0.03, C_I=0, C_B=1, C_P=0, Enz=1, T=310, pH=7 -> r0=0.03105
  C_A=0.3, C_I=0, C_B=1, C_P=0, Enz=1, T=310, pH=7 -> r0=0.3073
  C_A=3, C_I=0, C_B=1, C_P=0, Enz=1, T=310, pH=7 -> r0=2.265

Propose 12 distinct plausible rate laws (or refinements of those tried).
Each 'expr' is a Python expression in the 7 input names + your declared params, using only + - * / ** and exp, log, sqrt. Give physically plausible positive param bounds. JSON schema: {"hypotheses":[{"name":str,"expr":str,"params":[{"name":str,"low":float,"high":float}]}]}"""

EXPAND_SNAPSHOT = """\
An enzyme catalyses a reaction with initial rate r0 [mM/min]. Controllable inputs: C_A [substrate, mM], C_I [inhibitor, mM], C_B [2nd substrate, mM], C_P [product, mM], Enz [enzyme, mg/mL], T [K], pH. Discover an algebraic law r0 = f(C_A,C_I,C_B,C_P,Enz,T,pH; theta) from the data.

Experiments (inputs -> r0):
  C_A=0.03, C_I=0, C_B=1, C_P=0, Enz=1, T=310, pH=7 -> r0=0.03105
  C_A=0.3, C_I=0, C_B=1, C_P=0, Enz=1, T=310, pH=7 -> r0=0.3073
  C_A=3, C_I=0, C_B=1, C_P=0, Enz=1, T=310, pH=7 -> r0=2.265
  C_A=17.63, C_I=49.87, C_B=0.111, C_P=4.974, Enz=1.604, T=311.3, pH=6.007 -> r0=0.5419
  C_A=0.1167, C_I=36.97, C_B=0.0122, C_P=6.565, Enz=0.9968, T=336.4, pH=9.204 -> r0=0.02216

Forms ALREADY TRIED (in the pool) and their residuals -- do NOT re-propose any of these; propose forms STRUCTURALLY DIFFERENT from all of them:
  kcat * Enz * C_A / (Km * (1 + C_I / Ki) + C_A) -- median relative residual 0.08; its residual correlates most with C_I -- consider a factor in that variable (current best form)
  kcat * Enz * C_A / (Km + C_A) -- median relative residual 0.03; its residual correlates most with Enz
  kcat * Enz * C_A / (Km + C_A) ** 2 -- median relative residual 0.5; its residual correlates most with C_B
  k * log(C_I - 100) -- fit failed (no finite prediction on the data)
Refine by proposing a DIFFERENT algebraic form -- do NOT patch with ad-hoc offset/softening terms.

Experiment budget remaining: 19. Current phase: explore (explore = restructure the form; refine = tune an adequate form).

Propose 6 distinct plausible rate laws (or refinements of those tried).
Each 'expr' is a Python expression in the 7 input names + your declared params, using only + - * / ** and exp, log, sqrt. Give physically plausible positive param bounds. JSON schema: {"hypotheses":[{"name":str,"expr":str,"params":[{"name":str,"low":float,"high":float}]}]}"""


# The template (spec §9)


def test_init_prompt_snapshot():
    assert init_prompt() == INIT_SNAPSHOT


def test_expand_prompt_snapshot():
    assert expand_prompt() == EXPAND_SNAPSHOT


def test_system_prompt_is_the_templates():
    assert TEMPLATE[0] == "SYSTEM:"
    assert TEMPLATE[1] == SYSTEM_PROMPT
    assert SYSTEM_PROMPT.startswith("You are an enzyme kineticist")


@pytest.mark.parametrize(
    ("prompt", "dropped", "n_parts"),
    [
        (lambda: init_prompt(), REFINING + STATUS, 3),
        (lambda: expand_prompt(), [], 5),
        (lambda: expand_prompt(phase=False), STATUS, 4),
    ],
    ids=["init", "expand", "expand without phase"],
)
def test_prompt_is_the_templates_lines_in_its_order(prompt, dropped, n_parts):
    # Every fixed word is the file's: each line not indented is a line of the template, in the
    # template's order, with only its placeholders filled. The init prompt has no refining part
    # and no budget/phase line (spec §4); `phase` drops the latter (D6)
    text: str = prompt()
    expected = [line for line in SENT if line not in dropped]
    assert len(REFINING) == 2 and len(STATUS) == 1 and len(SENT) == 7
    lines = unindented(text)
    assert len(lines) == len(expected)
    for line, template_line in zip(lines, expected, strict=True):
        assert re.fullmatch(pattern(template_line), line), (line, template_line)
    assert len(text.split("\n\n")) == n_parts
    assert "<" not in text.replace("C_I - 100", "") and "[when" not in text


def test_placeholders_are_filled():
    status = re.search(pattern(STATUS[0]), expand_prompt(budget_left=7))
    assert status is not None and status.groups() == ("7", "explore")
    request = [line for line in SENT if line.startswith("Propose")][0]
    for text, n in [(init_prompt(n=3), "3"), (expand_prompt(n=11), "11")]:
        found = re.search(pattern(request), text)
        assert found is not None and found.group(1) == n


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("\n\n[budget/phase", "\n[budget/phase"),
        ("SYSTEM:", "SYSTEM"),
        ("USER (", "USER: ("),
        ("[when refining", "when refining"),
        ("negative evidence:]", "negative evidence:"),
        ("[budget/phase", "(budget/phase"),
        ("  <one line per", "  one line per"),
        ("r0=..>", "r0=.."),
        ("  <per-form", "  per-form"),
        ("<B>", "B"),
        ("<explore|refine>", "explore"),
        ("Propose N distinct", "Propose 12 distinct"),
        ("Propose N distinct", "Propose N distinct N"),
    ],
    ids=[
        "two parts merged",
        "no SYSTEM header",
        "no USER line",
        "refining marker not opened",
        "refining marker not closed",
        "status marker",
        "experiments placeholder not opened",
        "experiments placeholder not closed",
        "forms placeholder",
        "no budget placeholder",
        "no phase placeholder",
        "no N",
        "two N",
    ],
)
def test_template_with_another_layout_is_refused(old: str, new: str):
    # The module splits the file by its layout, so an edit that changes the layout must raise
    # instead of dropping or misplacing a part of the prompt
    assert _load_template(TEMPLATE_TEXT).system == SYSTEM_PROMPT
    assert TEMPLATE_TEXT.count(old) == 1
    with pytest.raises(ValueError, match="layout"):
        _load_template(TEMPLATE_TEXT.replace(old, new))


# The fields of the prompt (D6)


def test_init_prompt_shows_only_the_data():
    # At initialisation the pool is the unfitted seed library, which the prompt does not list
    assert render_prompt("init", POOL, D0, Y5[:3], weights=WEIGHTS, budget_left=5, n=12) == (
        init_prompt()
    )


def test_experiments_are_the_data_in_order_with_four_significant_digits():
    X = np.array([[0.000123456, 0.0, 99.99, 1e-3, 12345.6, 299.95, 6.98765]] * 2)
    X[1, CA] = 2.0
    text = render_prompt("init", (), X, np.array([123456.7, 0.0]), weights=None, budget_left=0, n=1)
    rows = [line for line in text.split("\n") if line.startswith("  ")]
    assert rows == [
        "  C_A=0.0001235, C_I=0, C_B=99.99, C_P=0.001, Enz=1.235e+04, T=299.9, pH=6.988 "
        "-> r0=1.235e+05",
        "  C_A=2, C_I=0, C_B=99.99, C_P=0.001, Enz=1.235e+04, T=299.9, pH=6.988 -> r0=0",
    ]


def form_lines(text: str) -> list[str]:
    part = text.split("\n\n")[2].split("\n")
    return part[1:-1]


def test_forms_are_listed_by_weight_best_first():
    lines = form_lines(expand_prompt())
    keys = [line.split(" -- ")[0].strip() for line in lines[:4]]
    assert keys == [POOL[i].key for i in (1, 0, 2, 3)]
    assert [line.endswith("(current best form)") for line in lines[:4]] == [1, 0, 0, 0]


def test_ties_go_to_the_candidate_added_first():
    lines = form_lines(expand_prompt(weights=np.array([0.4, 0.2, 0.4, 0.0])))
    keys = [line.split(" -- ")[0].strip() for line in lines[:4]]
    assert keys == [POOL[i].key for i in (0, 2, 1, 3)]
    assert [line.endswith("(current best form)") for line in lines[:4]] == [1, 0, 0, 0]


def test_form_is_its_dedupe_key_and_a_failed_fit_says_so():
    lines = form_lines(expand_prompt())
    # SQUARED is written over two lines with a comment; the prompt shows its key on one
    assert (
        "  kcat * Enz * C_A / (Km + C_A) ** 2 -- median relative residual 0.5; "
        "its residual correlates most with C_B"
    ) in lines
    assert "  k * log(C_I - 100) -- fit failed (no finite prediction on the data)" in lines
    assert all("#" not in line for line in lines)


def test_full_size_prompt_lists_every_observation_and_every_form():
    # The largest prompt of a cell: 25 observations and a pool at N_max = 24, most of it at
    # weight 0. Nothing is cut, forms at the same weight keep the pool's order, and a form's
    # input is taken from all of the data
    rng = np.random.default_rng(0)
    X = np.vstack([D0, from_unit(rng.uniform(size=(22, 7)))])
    X[3:5, TEMP] = 310.0  # T is constant over the first five observations
    y = rng.uniform(0.1, 5.0, size=25)
    rel = rng.uniform(-0.5, 0.5, size=(24, 25))
    rel[7] = (X[:, TEMP] - 320.0) / (100.0 * y)  # this form's residual rises with T
    pool = tuple(fitted(power(p + 1), list(rel[p]), X, y) for p in range(24))
    weights = np.zeros(24)
    weights[[20, 3, 11]] = [0.5, 0.3, 0.2]
    text = render_prompt("expand", pool, X, y, weights=weights, budget_left=0, n=6)

    experiments = text.split("\n\n")[1].split("\n")[1:]
    assert [line.split(" -> r0=")[1] for line in experiments] == [f"{v:.4g}" for v in y]
    lines = form_lines(text)
    order = [20, 3, 11, *(i for i in range(24) if i not in (20, 3, 11))]
    assert [line.split(" -- ")[0].strip() for line in lines] == [pool[i].key for i in order]
    assert [line.endswith("(current best form)") for line in lines] == [True] + [False] * 23
    for i, line in zip(order, lines, strict=True):
        fit = pool[i].fit
        assert fit is not None
        assert shown_input(line) == correlated_input(X, y, fit.y_hat)
    assert shown_input(lines[order.index(7)]) == "T"


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        ([0.3, 0.7], 1),
        ([0.7, 0.3], 0),
        ([0.5, 0.5], 0),
        ([0.0, 0.5, 0.5], 1),
        ([0.0, 0.0], None),
        ([0.0, 1e-300], 1),
    ],
)
def test_map_index_is_the_largest_weight_first_on_ties(weights, expected):
    assert map_index(np.array(weights)) == expected


def test_map_form_is_the_agents():
    # Two spellings of one law fit the noise-free D0 exactly, so their BICs meet at the RSS floor
    # (D2) and their weights tie: the agent's MAP model is the first, and so is the prompt's best
    reordered = law("mm_reordered", "C_A * kcat * Enz / (Km + C_A)", "kcat", "Km")
    y0 = compile_hypothesis(MM)(np.array([2.5, 0.7]), D0)
    config = replace(DEFAULTS, n_starts=2, n_evals=9)
    agent = MDAAgent(D0, y0, seed=0, config=config, library=(MM, reordered, power(2)))
    assert agent.weights[0] == agent.weights[1] > agent.weights[2]
    assert map_index(agent.weights) == agent.map_index == 0
    text = render_prompt(
        "expand", agent.pool, agent.X, agent.y, weights=agent.weights, budget_left=22, n=6
    )
    [best] = [line for line in form_lines(text) if line.endswith("(current best form)")]
    assert best.strip().startswith(agent.get_model().key + " -- ")


def test_pool_without_a_finite_fit_has_no_best_form_and_explores():
    agent = MDAAgent(D0, Y5[:3], seed=0, config=replace(DEFAULTS, n_starts=1), library=(BROKEN,))
    assert agent.map_index is None and map_index(agent.weights) is None
    text = render_prompt(
        "expand", agent.pool, agent.X, agent.y, weights=agent.weights, budget_left=3, n=6
    )
    assert form_lines(text) == [
        "  k * log(C_I - 100) -- fit failed (no finite prediction on the data)"
    ]
    assert "Current phase: explore " in text


def rows(*columns: list[float]) -> np.ndarray:
    """Conditions with the given columns, C_A first, and the other inputs at D0's values."""
    X = np.tile(D0[0], (len(columns[0]), 1))
    for j, column in enumerate(columns):
        X[:, j] = column
    return X


@pytest.mark.parametrize(
    ("X", "residual", "expected"),
    [
        # C_I follows the residual (Spearman 1), ahead of C_A (0.9) though later in INPUT_NAMES
        (rows([0, 2, 1, 3, 4], [0, 1, 2, 3, 4]), [1, 2, 3, 4, 5], "C_I"),
        # The sign does not matter: C_A runs against the residual (-1), C_I with it (0.9)
        (rows([4, 3, 2, 1, 0], [0, 2, 1, 3, 4]), [1, 2, 3, 4, 5], "C_A"),
        # Only C_A varies, as on D0
        (D0, [0.1, -0.2, 0.3], "C_A"),
        # Equal correlations go to the earlier input, whatever their signs
        (rows([1, 2, 3], [1, 2, 3], [1, 2, 3]), [1, 2, 3], "C_A"),
        (rows([3, 2, 1], [1, 2, 3]), [1, 2, 3], "C_A"),
        (rows([1, 1, 1], [3, 2, 1], [1, 2, 3]), [1, 2, 3], "C_I"),
        # A constant residual correlates with nothing
        (rows([1, 2, 3], [3, 1, 2]), [0.5, 0.5, 0.5], None),
        # Nor does a non-finite one
        (rows([1, 2, 3], [3, 1, 2]), [0.5, np.nan, 0.5], None),
    ],
    ids=[
        "strongest",
        "sign",
        "only",
        "tie",
        "tie of signs",
        "tie after a constant",
        "constant",
        "non-finite",
    ],
)
def test_correlated_input_is_the_strongest(X, residual, expected):
    y = np.full(len(X), 10.0)
    assert correlated_input(np.asarray(X, dtype=float), y, y - np.array(residual)) == expected


ADVICE = " -- consider a factor in that variable"


def advised(text: str) -> list[str]:
    """The lines that carry the template's advice to consider a factor."""
    return [line for line in text.split("\n") if "consider a factor" in line]


def best_line(text: str) -> str:
    [line] = [x for x in form_lines(text) if x.endswith("(current best form)")]
    return line


def shown_input(line: str) -> str | None:
    """The input a form's line shows, or None."""
    found = re.search(r"; its residual correlates most with (\S+)", line)
    return found.group(1) if found else None


def test_each_form_shows_the_input_of_its_own_residual():
    # App. G: "which input each residual correlates most with". In the snapshot the three
    # fitted forms point at three different inputs; the failed fit shows none
    for weights in (WEIGHTS, np.array([0.1, 0.1, 0.7, 0.1]), np.zeros(4)):
        lines = form_lines(expand_prompt(weights=weights))
        for c in POOL:
            [line] = [x for x in lines if x.startswith(f"  {c.key} -- ")]
            expected = (
                None if c.fit is None or not c.fit.ok else (correlated_input(X5, Y5, c.fit.y_hat))
            )
            assert shown_input(line) == expected
            assert line.count("correlates most with") == (expected is not None)
            assert (ADVICE in line) == (
                expected is not None and line.endswith("(current best form)")
            )
    assert [correlated_input(X5, Y5, c.fit.y_hat) for c in POOL[:3] if c.fit] == [
        "Enz",
        "C_I",
        "C_B",
    ]


def test_failed_fit_shows_no_input_even_with_finite_predictions():
    # A fit fails when its RSS is not finite, which an overflow gives with finite predictions;
    # their residual would correlate with C_A
    overflow = Y5 * 1e200 * np.arange(1, 6)
    failed = replace(
        POOL[0], fit=FitResult(np.ones(2), math.inf, overflow, False), fit_err=math.inf
    )
    assert correlated_input(X5, Y5, overflow) is not None
    text = render_prompt("expand", (failed, *POOL[1:]), X5, Y5, weights=WEIGHTS, budget_left=1, n=6)
    assert f"  {failed.key} -- fit failed (no finite prediction on the data)" in form_lines(text)


def test_form_whose_residual_correlates_with_nothing_shows_only_its_residual():
    # A residual of exactly 0 everywhere, on a form that is not the best
    flat = replace(POOL[0], fit=FitResult(np.ones(2), 1.0, Y5.copy(), True))
    text = render_prompt("expand", (flat, *POOL[1:]), X5, Y5, weights=WEIGHTS, budget_left=1, n=6)
    assert f"  {flat.key} -- median relative residual 0.03" in form_lines(text)
    assert advised(text) == [best_line(text)]


def test_advice_is_on_the_map_forms_line_only():
    # The template's advice for the current best form, folded into that form's line, with
    # "that variable" for the template's "those variable(s)": there is one input. The best form
    # is the one with the largest weight, also when that is less than half
    for best, top in itertools.product((0, 1, 2), (0.7, 0.4)):
        weights = np.full(4, (1 - top) / 3)
        weights[best] = top
        c = POOL[best]
        assert c.fit is not None
        name = correlated_input(X5, Y5, c.fit.y_hat)
        assert name is not None
        text = expand_prompt(weights=weights)
        assert best_line(text) == (
            f"  {c.key} -- median relative residual {c.fit_err:.4g}; its residual correlates "
            f"most with {name}{ADVICE} (current best form)"
        )
        assert advised(text) == [best_line(text)]
        # The first form listed is the best, and the other forms carry no advice
        assert form_lines(text)[0] == best_line(text)
        assert not any(line.startswith("  The current best form's") for line in text.split("\n"))


def test_input_ranks_the_signed_residual_of_the_rate():
    # Rates over four decades: the residual y - y_hat ranks as C_A does (Spearman 1) and C_I less
    # (0.8), while its counterpart in log rates would rank the other way round
    X = rows([4, 3, 2, 1], [3, 4, 2, 1])
    y = np.array([10.0, 1.0, 0.1, 0.01])
    residual = np.array([0.4, 0.3, 0.02, 0.005])
    best = fitted(MM, list(residual / y), X, y)
    text = render_prompt("expand", (best,), X, y, weights=np.ones(1), budget_left=1, n=6)
    assert shown_input(best_line(text)) == "C_A"
    assert advised(text) == [best_line(text)] and f"C_A{ADVICE}" in best_line(text)


def test_no_advice_without_a_correlated_input_or_a_best_form():
    # A residual of exactly 0 everywhere, on the best form: its line shows no input and no advice
    flat = replace(POOL[1], fit=FitResult(np.ones(3), 1.0, Y5.copy(), True))
    pool = (POOL[0], flat, *POOL[2:])
    text = render_prompt("expand", pool, X5, Y5, weights=WEIGHTS, budget_left=1, n=6)
    assert advised(text) == []
    assert best_line(text) == f"  {flat.key} -- median relative residual 0.08 (current best form)"
    # No best form: each form still shows its own input, the first listed with no advice
    no_best = expand_prompt(weights=np.zeros(4))
    assert advised(no_best) == [] and "(current best form)" not in no_best
    assert sum("; its residual correlates most with" in x for x in form_lines(no_best)) == 3


def test_best_form_with_a_failed_fit_is_marked_without_advice():
    # The agent cannot give one, since a failed fit has weight 0; a hand-made pool can
    fit = POOL[1].fit
    assert fit is not None
    failed = replace(POOL[1], fit=replace(fit, ok=False, rss=math.inf), fit_err=math.inf)
    text = render_prompt(
        "expand", (POOL[0], failed, *POOL[2:]), X5, Y5, weights=WEIGHTS, budget_left=1, n=6
    )
    assert best_line(text) == (
        f"  {failed.key} -- fit failed (no finite prediction on the data) (current best form)"
    )
    assert advised(text) == []


def with_best_error(err: float) -> tuple[Candidate, ...]:
    return (POOL[0], replace(POOL[1], fit_err=err), *POOL[2:])


@pytest.mark.parametrize(
    ("err", "tau_fit", "phase"),
    [
        (0.04, 0.05, "refine"),
        (0.05, 0.05, "refine"),
        (0.06, 0.05, "explore"),
        (0.06, 0.1, "refine"),
        (0.04, 0.02, "explore"),
        (math.inf, 0.05, "explore"),
    ],
)
def test_phase_is_refine_once_the_best_form_fits(err, tau_fit, phase):
    # The best form's in-sample median relative residual against tau_fit (D6); the others' do not
    # matter, and the pool's first form (residual 0.03) is not the best
    text = render_prompt(
        "expand", with_best_error(err), X5, Y5, weights=WEIGHTS, budget_left=4, n=6, tau_fit=tau_fit
    )
    assert f"Experiment budget remaining: 4. Current phase: {phase} (" in text


def test_phase_without_a_best_form_is_explore():
    text = expand_prompt(weights=np.zeros(4), tau_fit=math.inf)
    assert "Current phase: explore (" in text


@pytest.mark.parametrize(
    ("err", "shown"),
    [(0.123456, "0.1235"), (0.0500049, "0.05"), (1.23456e-05, "1.235e-05"), (12.3456, "12.35")],
)
def test_residuals_have_four_significant_digits(err: float, shown: str):
    # Every number of the prompt has the format of the experiments (spec §9)
    text = render_prompt(
        "expand", with_best_error(err), X5, Y5, weights=WEIGHTS, budget_left=1, n=6
    )
    assert best_line(text).startswith(f"  {POOL[1].key} -- median relative residual {shown}; ")


def test_expansion_needs_fitted_candidates_and_their_weights():
    with pytest.raises(ValueError, match="weight"):
        expand_prompt(weights=None)
    with pytest.raises(ValueError, match="weight"):
        expand_prompt(weights=WEIGHTS[:3])
    with pytest.raises(ValueError, match="weight"):
        expand_prompt(weights=np.append(WEIGHTS, 0.0))
    unfitted = (POOL[0], replace(POOL[1], fit=None))
    with pytest.raises(ValueError, match="no fit"):
        render_prompt("expand", unfitted, X5, Y5, weights=WEIGHTS[:2], budget_left=1, n=6)


@pytest.fixture
def no_runtime_typecheck():
    """Switch off the jaxtyping/beartype hook of pytest, so a function sees what a run passes."""
    jaxtyping.config.update("jaxtyping_disable", True)
    yield
    jaxtyping.config.update("jaxtyping_disable", False)


@pytest.mark.usefixtures("no_runtime_typecheck")
def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        render_prompt("other", (), X5, Y5, weights=None, budget_left=1, n=6)  # ty: ignore[invalid-argument-type]
    p = proposer([])
    with pytest.raises(ValueError, match="kind"):
        p.propose("other", (), X5, Y5, weights=None, budget_left=1)  # ty: ignore[invalid-argument-type]
    assert fake(p).requests == []


# The request (spec §9, D15)


def test_llm_proposer_is_a_proposer():
    assert isinstance(proposer([]), Proposer)


@pytest.mark.parametrize("structured", [True, False])
def test_request_is_the_rendered_prompt(structured: bool):
    config = replace(DEFAULTS, structured_output=structured)
    p = proposer([laws(power(2)), laws(power(3))], config)
    # The init prompt is the same whatever pool and weights come with the call
    p.propose("init", POOL, D0, Y5[:3], weights=WEIGHTS, budget_left=5)
    expand(p)
    schema = HYPOTHESES_SCHEMA if structured else None
    assert fake(p).requests == [
        Request(SYSTEM_PROMPT, init_prompt(), schema),
        Request(SYSTEM_PROMPT, expand_prompt(), schema),
    ]


@pytest.mark.parametrize(
    ("changes", "init_has", "expand_has"),
    [
        ({}, "Propose 12 distinct", "Propose 6 distinct"),
        ({"n_init": 5, "n_new": 2}, "Propose 5 distinct", "Propose 2 distinct"),
        ({"tau_fit": 0.1}, "Propose 12 distinct", "Current phase: refine ("),
    ],
)
def test_config_reaches_the_prompt(changes, init_has, expand_has):
    p = proposer([laws(), laws()], replace(DEFAULTS, **changes))
    init(p)
    expand(p)
    first, second = fake(p).requests
    assert init_has in first.user and expand_has in second.user


def test_prompt_phase_off_drops_the_budget_and_phase_line():
    p = proposer([laws(), laws()], replace(DEFAULTS, prompt_phase=False))
    init(p)
    expand(p)
    user = fake(p).requests[1].user
    assert user == expand_prompt(phase=False)
    assert "Experiment budget remaining" not in user and "Forms ALREADY TRIED" in user


def test_budget_left_reaches_the_prompt():
    p = proposer([laws(), laws()])
    init(p)
    expand(p, budget_left=3)
    assert "Experiment budget remaining: 3. " in fake(p).requests[1].user


# Reply handling: parse, dedupe, cap (spec §4, D13)


def test_valid_hypotheses_are_returned_in_reply_order():
    bad = law("bad", "k * unknown_input", "k")
    p = proposer([laws(power(3), bad, power(2), request_id="req_a")])
    assert init(p) == Proposal(
        (power(3), power(2)), (CallRecord(f"{PREFIX}/call00/try1", "req_a"),)
    )


def seed_pool() -> tuple[Candidate, ...]:
    return tuple(
        Candidate(h, compile_hypothesis(h), normalise_expr(h.expr), Origin.SEED, 3)
        for h in SEED_LIBRARY
    )


def test_duplicates_of_the_pool_are_dropped_up_to_formatting():
    # At initialisation the pool is the seed library; at an expansion, the fitted pool
    mm_again = law("mm_again", "kcat*Enz*C_A / ((Km+C_A))  # Michaelis-Menten", "kcat", "Km")
    squared_again = law("squared_again", "kcat*Enz*C_A/(Km+C_A)**2", "kcat", "Km")
    # A renamed parameter makes a new expression (D13)
    renamed = law("renamed", "kcat*Enz*C_A/(K + C_A)", "kcat", "K")
    # A form whose fit failed is in the pool like any other
    broken_again = law("broken_again", "k*log(C_I-100)", "k")
    p = proposer(
        [
            laws(mm_again, power(2), renamed),
            laws(squared_again, broken_again, power(3), mm_again),
        ]
    )
    assert init(p, pool=seed_pool()).hypotheses == (power(2), renamed)
    assert expand(p).hypotheses == (power(3),)


def test_duplicates_within_the_reply_keep_the_first():
    again = law("again", "k*C_A**2", "k")
    p = proposer([laws(power(2), power(3), again)])
    assert init(p).hypotheses == (power(2), power(3))


def test_cap_is_applied_after_dedupe_in_reply_order():
    # Duplicates and invalid laws do not take places: the first N new valid laws are kept
    bad = law("bad", "k * unknown_input", "k")
    reply_laws = [
        SQUARED,
        power(2),
        law("again", "k*C_A**2", "k"),
        bad,
        power(3),
        power(4),
        power(5),
    ]
    p = proposer([laws(), laws(*reply_laws)], replace(DEFAULTS, n_new=2))
    init(p)
    assert expand(p).hypotheses == (power(2), power(3))
    p = proposer([laws(*reply_laws)], replace(DEFAULTS, n_init=3))
    assert init(p).hypotheses == (SQUARED, power(2), power(3))
    p = proposer([laws(*reply_laws)], replace(DEFAULTS, n_init=12))
    assert init(p).hypotheses == (SQUARED, power(2), power(3), power(4), power(5))


# Retries (D14)

UNUSABLE = {
    "max_tokens": reply(reply_text(power(9)), StopReason.MAX_TOKENS, "req_bad"),
    "refusal": reply(reply_text(power(9)), StopReason.REFUSAL, "req_bad"),
    "other": reply(reply_text(power(9)), StopReason.OTHER, "req_bad"),
    "not json": reply("I cannot answer in JSON.", request_id="req_bad"),
    "no hypotheses list": reply('{"laws": []}', request_id="req_bad"),
}


@pytest.mark.parametrize("kind", list(UNUSABLE))
def test_unusable_reply_is_retried_with_a_new_call_id(kind: str):
    # The usable text of a reply cut at max_tokens, refused or stopped otherwise is not used
    p = proposer([UNUSABLE[kind], laws(power(2), request_id="req_good")])
    proposal = init(p)
    assert proposal == Proposal(
        (power(2),),
        (
            CallRecord(f"{PREFIX}/call00/try1", "req_bad"),
            CallRecord(f"{PREFIX}/call00/try2", "req_good"),
        ),
    )
    first, second = fake(p).requests
    assert first == second  # the same request, made again


@pytest.mark.parametrize("max_attempts", [1, 2, 4])
def test_after_max_attempts_the_proposal_is_empty(max_attempts: int):
    unusable = list(UNUSABLE.values())
    script = [unusable[i % len(unusable)] for i in range(max_attempts)] + [laws(power(2))]
    p = proposer(script, replace(DEFAULTS, max_attempts=max_attempts))
    proposal = init(p)
    assert proposal.hypotheses == ()
    assert [c.call_id for c in proposal.calls] == [
        f"{PREFIX}/call00/try{a}" for a in range(1, max_attempts + 1)
    ]
    assert len(fake(p).requests) == max_attempts
    # The next call starts again from the first attempt
    assert expand(p).calls == (CallRecord(f"{PREFIX}/call01/try1"),)


@pytest.mark.parametrize(
    "text",
    [
        '{"hypotheses": []}',
        reply_text(law("bad", "k * unknown_input", "k"), law("unused", "C_A", "k")),
        reply_text(SQUARED, law("mm_again", "kcat*Enz*C_A/(Km + C_A)", "kcat", "Km")),
    ],
    ids=["empty", "all invalid", "all duplicates"],
)
def test_usable_reply_is_not_retried_even_when_it_adds_nothing(text: str):
    # D14 retries unusable replies only: a reply that parses is the call's answer
    p = proposer([laws(), reply(text, request_id="req_a"), laws(power(2))])
    init(p)
    proposal = expand(p)
    assert proposal == Proposal((), (CallRecord(f"{PREFIX}/call01/try1", "req_a"),))
    assert len(fake(p).requests) == 2


def test_failed_request_raises():
    p = proposer([LLMError("connection refused")])
    with pytest.raises(LLMError, match="connection refused"):
        init(p)


def test_empty_initial_batch_leaves_the_seed_library():
    # Every attempt unusable, or a usable reply with nothing new: the pool is the seeds (D14)
    config = replace(DEFAULTS, n_starts=1)
    y0 = compile_hypothesis(MM)(np.array([2.5, 0.7]), D0)
    for script, n_calls in [
        ([UNUSABLE["max_tokens"]] * 3, 3),
        ([laws(SEED_LIBRARY[3], SEED_LIBRARY[5])], 1),
    ]:
        p = proposer(script, config)
        agent = MDAAgent(D0, y0, seed=0, config=config, proposer=p)
        assert [c.hypothesis for c in agent.pool] == list(SEED_LIBRARY)
        assert agent.info.n_added == 0
        assert [c.call_id for c in agent.info.calls] == [
            f"{PREFIX}/call00/try{a}" for a in range(1, n_calls + 1)
        ]


# Call IDs (D17)


def test_call_ids_name_the_cell_the_call_and_the_attempt(tmp_path: Path):
    transcript = tmp_path / "transcript.jsonl"
    script = [
        laws(power(2), request_id="r0"),  # call00: usable at once
        cut("r1a"),  # call01: usable at the second attempt
        laws(power(3), request_id="r1b"),
        cut("r2a"),  # call02: never usable
        reply("no json", request_id="r2b"),
        reply("", StopReason.REFUSAL, request_id="r2c"),
        laws(power(4), request_id="r3"),  # call03
    ]
    p = proposer(script, transcript=transcript)
    proposals = [init(p), expand(p), expand(p), expand(p)]
    expected = [
        [("call00/try1", "r0")],
        [("call01/try1", "r1a"), ("call01/try2", "r1b")],
        [("call02/try1", "r2a"), ("call02/try2", "r2b"), ("call02/try3", "r2c")],
        [("call03/try1", "r3")],
    ]
    assert [list(q.calls) for q in proposals] == [
        [CallRecord(f"{PREFIX}/{call}", request_id) for call, request_id in calls]
        for calls in expected
    ]
    # Each attempt reached the client under its own call ID
    assert call_ids(transcript) == [f"{PREFIX}/{c}" for calls in expected for c, _ in calls]
    assert [q.hypotheses for q in proposals] == [(power(2),), (power(3),), (), (power(4),)]
    assert p.n_calls == 4


def test_one_proposer_serves_one_cell():
    p = proposer([laws(), laws()])
    with pytest.raises(RuntimeError, match="one cell"):
        expand(p)
    init(p)
    with pytest.raises(RuntimeError, match="one cell"):
        init(p)
    expand(p)
    assert len(fake(p).requests) == 2


def test_no_request_when_no_law_is_asked_for(tmp_path: Path):
    # N = 0 makes no call and adds nothing, and the call still takes its index
    transcript = tmp_path / "transcript.jsonl"
    p = proposer(
        [laws(power(2)), laws(power(3))], replace(DEFAULTS, n_init=0), transcript=transcript
    )
    assert init(p) == Proposal()
    assert expand(p).calls == (CallRecord(f"{PREFIX}/call01/try1"),)
    q = proposer([laws(power(2)), laws(power(3))], replace(DEFAULTS, n_new=0))
    assert init(q).hypotheses == (power(2),)
    assert expand(q) == expand(q) == Proposal()
    assert len(fake(q).requests) == 1 and q.n_calls == 3
    assert call_ids(transcript) == [f"{PREFIX}/call01/try1"]


# The hybrid arm on a short cell (spec §1, §8)

FRACTAL = "c93_fractal_competitive"  # no seed law fits it within tau_e, so every update expands
SHORT = replace(DEFAULTS, n_active=2, n_starts=1, n_evals=9)
FRACTAL_PREFIX = f"hybrid/{FRACTAL}/s0"


def hybrid_llm(**kwargs) -> FakeLLM:
    """Two laws at initialisation, one per expansion; the third request is cut at max_tokens."""
    counter = itertools.count()

    def respond(request: Request) -> Reply:
        i = next(counter)
        if i == 0:
            return laws(power(2), power(3), request_id="req_0")
        if i == 2:
            return cut("req_2")
        return laws(power(i + 3), request_id=f"req_{i}")

    return FakeLLM(respond, **kwargs)


@cache
def hybrid_log(cache_dir: Path) -> tuple[RunLog, FakeLLM]:
    llm = hybrid_llm(cache_dir=cache_dir)
    return run_cell(
        FRACTAL, 0, SHORT, LLMProposer(llm, call_prefix=FRACTAL_PREFIX, config=SHORT)
    ), llm


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cache")


def test_run_log_shows_the_expansions_and_the_llm_calls(cache_dir: Path):
    log, llm = hybrid_log(cache_dir)
    assert log.proposer == "LLMProposer"
    assert [s.llm_calls for s in log.steps] == [
        [CallRecord(f"{FRACTAL_PREFIX}/call00/try1", "req_0")],
        [CallRecord(f"{FRACTAL_PREFIX}/call01/try1", "req_1")],
        [
            CallRecord(f"{FRACTAL_PREFIX}/call02/try1", "req_2"),
            CallRecord(f"{FRACTAL_PREFIX}/call02/try2", "req_3"),
        ],
    ]
    assert [s.n_added for s in log.steps] == [2, 1, 1]
    assert [s.expanded for s in log.steps] == [True, True, True]
    assert [s.n_e for s in log.steps] == [0, 1, 2]
    added = {m.name: (m.origin, m.added_at) for m in log.steps[-1].pool if m.origin != "seed"}
    assert added == {
        "power_2": ("init", 3),
        "power_3": ("init", 3),
        "power_4": ("expand", 4),
        "power_6": ("expand", 5),
    }
    assert RunLog.from_json(log.to_json()) == log
    # The prompts the agent's state gave: the data of each step and the budget left
    users = [r.user for r in llm.requests]
    assert "Forms ALREADY TRIED" not in users[0] and "Propose 12 distinct" in users[0]
    assert "Experiment budget remaining: 1. " in users[1] and "Propose 6 distinct" in users[1]
    assert users[2] == users[3] and "Experiment budget remaining: 0. " in users[2]
    assert [u.count(" -> r0=") for u in users] == [3, 4, 5, 5]
    assert all(f"  {normalise_expr(m.expr)} -- " in users[2] for m in log.steps[1].pool)
    assert (llm.n_sent, llm.n_cached) == (4, 0)


def test_rerun_with_the_cache_sends_nothing_and_gives_the_same_log(cache_dir: Path):
    log, _ = hybrid_log(cache_dir)
    offline = FakeLLM(
        lambda request: AssertionError("a request reached the provider"), cache_dir=cache_dir
    )
    rerun = run_cell(
        FRACTAL, 0, SHORT, LLMProposer(offline, call_prefix=FRACTAL_PREFIX, config=SHORT)
    )
    assert rerun == log
    assert (offline.requests, offline.n_sent, offline.n_cached) == ([], 0, 4)


def test_proposer_must_hold_the_config_of_its_cell():
    # The run log records one config, the agent's. A proposer built from another would ask for
    # other numbers of laws or render another prompt than the log says, so it is refused before
    # any request is made
    p = proposer([laws(power(2))])  # built from DEFAULTS
    with pytest.raises(ValueError, match=r"config differs .*\['n_active', 'n_starts', 'n_evals'\]"):
        run_cell(FRACTAL, 0, SHORT, p)
    cell = replace(DEFAULTS, n_starts=1, prompt_phase=False)
    with pytest.raises(ValueError, match=r"config differs .*\['n_starts', 'prompt_phase'\]"):
        MDAAgent(D0, Y5[:3], seed=0, config=cell, proposer=p)
    assert fake(p).requests == [] and p.n_calls == 0
    # An equal config is the cell's, whichever object holds it
    q = proposer([laws(power(2))], replace(cell))
    agent = MDAAgent(D0, Y5[:3], seed=0, config=cell, proposer=q)
    assert agent.info.n_added == 1 and len(fake(q).requests) == 1


# The hybrid arm's client (spec §9)


def test_claude_proposer_builds_its_client_from_the_config(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    config = replace(DEFAULTS, llm_effort="low", llm_max_tokens=8000, llm_max_retries=1, n_new=3)
    p = claude_proposer(
        config, call_prefix=PREFIX, cache_dir=tmp_path / "cache", transcript=tmp_path / "t.jsonl"
    )
    assert isinstance(p.llm, ClaudeLLM)
    assert (p.llm.model, p.llm.effort, p.llm.max_tokens) == ("claude-haiku-5-5", "low", 8000)
    assert p.llm._client.max_retries == 1
    assert p.llm.cache is not None and p.llm.cache.root == tmp_path / "cache"
    assert p.llm.transcript == tmp_path / "t.jsonl"
    assert (p.call_prefix, p.config, p.n_calls) == (PREFIX, config, 0)
    plain = claude_proposer(DEFAULTS, call_prefix=PREFIX)
    assert isinstance(plain.llm, ClaudeLLM)
    assert (plain.llm.cache, plain.llm.transcript) == (None, None)
    assert (plain.llm.effort, plain.llm.max_tokens) == ("medium", 16_000)
    assert plain.llm._client.max_retries == DEFAULTS.llm_max_retries
    # The model is the config's: one without pricing is refused
    with pytest.raises(ValueError, match="no pricing"):
        claude_proposer(replace(DEFAULTS, llm_model="claude-unknown"), call_prefix=PREFIX)
