import json
import math

import numpy as np
import pytest

from mda.bench import INPUT_NAMES, World, WorldSpec, from_unit
from mda.hypothesis import Hypothesis, HypothesisError, Param, compile_hypothesis
from mda.library import SEED_BY_WORLD, SEED_LIBRARY
from mda.pool import N_STARTS, Origin, Pool
from mda.posterior import adaptive_cap, bic_fixed, bic_profiled, log_weights, prune_top, weights

MM = Hypothesis(
    "michaelis_menten",
    "Vmax * C_A / (Km + C_A)",
    (Param("Vmax", 1e-3, 1e3), Param("Km", 1e-3, 1e3)),
)
HILL = Hypothesis(
    "hill",
    "Vmax * C_A**n / (K**n + C_A**n)",
    (Param("Vmax", 1e-3, 1e3), Param("K", 1e-3, 1e3), Param("n", 0.1, 8.0)),
)
LINEAR = Hypothesis("linear", "k * C_A", (Param("k", 1e-3, 1e3),))
# Non-finite everywhere: C_I is within [0, 50], so the argument of the log is negative
BROKEN = Hypothesis("broken", "k * log(C_I - 100)", (Param("k", 0.1, 10.0),))
# j does not affect the predictions, so the optimiser never moves it: the fitted j is its start
FLAT = Hypothesis("flat", "k * C_A + 0 * j", (Param("k", 0.1, 10.0), Param("j", 0.0, 100.0)))
MM_TRUE = np.array([2.5, 0.7])
CA = INPUT_NAMES.index("C_A")

# Passive data D0 (spec §1): C_A in {0.03, 0.3, 3}, other inputs at (0, 1, 0, 1, 310, 7)
D0 = np.array([[a, 0.0, 1.0, 0.0, 1.0, 310.0, 7.0] for a in (0.03, 0.3, 3.0)])


def mm_data(n: int = 25) -> tuple[np.ndarray, np.ndarray]:
    X = from_unit(np.random.default_rng(0).uniform(size=(n, len(INPUT_NAMES))))
    return X, compile_hypothesis(MM)(MM_TRUE, X)


def pool_of(*hypotheses: Hypothesis, origin: Origin = Origin.INIT) -> Pool:
    pool = Pool()
    pool.add(hypotheses, origin, added_at=3)
    return pool


def refit(pool: Pool, X: np.ndarray, y: np.ndarray, seed: int = 0, **kwargs) -> None:
    pool.refit(X, y, rng=np.random.default_rng(seed), **kwargs)


def names(pool: Pool) -> list[str]:
    return [c.hypothesis.name for c in pool]


def test_defaults_follow_spec():
    assert N_STARTS == 8  # D8


def test_add_returns_unfitted_candidates():
    pool = Pool()
    assert len(pool) == 0
    added = pool.add(SEED_LIBRARY, Origin.SEED, added_at=3)
    assert len(pool) == len(added) == 9
    assert [c.hypothesis for c in pool] == list(SEED_LIBRARY)
    assert all(a is c for a, c in zip(added, pool, strict=True))
    for c in pool:
        assert c.origin is Origin.SEED
        assert c.added_at == 3
        assert c.fit is None
        assert math.isnan(c.fit_err)
    X, y = mm_data(5)
    np.testing.assert_array_equal(pool_of(MM)[0].f(MM_TRUE, X), y)


@pytest.mark.parametrize(
    "expr",
    [
        "Vmax*C_A/(Km+C_A)",
        "  Vmax * C_A\t/ ( Km + C_A )\r\n",
        "Vmax * C_A  # saturating in A\n/ (Km + C_A)",
        "(Vmax * C_A) / ((Km) + C_A)",
    ],
    ids=["no-spaces", "whitespace", "comment", "redundant-parentheses"],
)
def test_dedupe_ignores_formatting(expr):
    pool = pool_of(MM)
    assert pool.add([Hypothesis("again", expr, MM.params)], Origin.EXPAND, added_at=7) == []
    assert names(pool) == ["michaelis_menten"]
    assert (pool[0].origin, pool[0].added_at) == (Origin.INIT, 3)


def test_renamed_parameter_is_a_new_expression():
    # D13: exact-expression matching does not merge equivalent laws
    pool = pool_of(MM)
    renamed = Hypothesis(
        "renamed", "V * C_A / (K + C_A)", (Param("V", 1e-3, 1e3), Param("K", 1e-3, 1e3))
    )
    reordered = Hypothesis("reordered", "Vmax * C_A / (C_A + Km)", MM.params)
    added = pool.add([renamed, reordered], Origin.EXPAND, added_at=7)
    assert [c.hypothesis.name for c in added] == ["renamed", "reordered"]
    assert [c.origin for c in pool] == [Origin.INIT, Origin.EXPAND, Origin.EXPAND]
    assert [c.added_at for c in pool] == [3, 7, 7]


def test_dedupe_ignores_names_and_bounds():
    pool = pool_of(MM)
    other = Hypothesis("other", MM.expr, (Param("Vmax", 1.0, 2.0), Param("Km", 0.5, 0.5)))
    assert pool.add([other], Origin.EXPAND, added_at=4) == []
    assert pool[0].hypothesis is MM


def test_dedupe_within_a_batch_keeps_the_first():
    twin = Hypothesis("twin", "Vmax*C_A/(Km+C_A)", MM.params)
    pool = pool_of(LINEAR)
    added = pool.add([MM, twin, HILL, LINEAR], Origin.INIT, added_at=3)
    assert [c.hypothesis.name for c in added] == ["michaelis_menten", "hill"]
    assert names(pool) == ["linear", "michaelis_menten", "hill"]


def test_origin_values_are_the_run_log_strings():
    assert [o.value for o in Origin] == ["seed", "init", "expand"]
    assert (
        json.dumps({"origin": Origin.EXPAND, "added_at": 7})
        == '{"origin": "expand", "added_at": 7}'
    )
    assert Origin("init") is Origin.INIT


def test_add_rejects_an_invalid_hypothesis_as_a_whole():
    pool = pool_of(MM)
    unknown_name = Hypothesis("bad", "q * C_A", (Param("k", 0.0, 1.0),))
    with pytest.raises(HypothesisError):
        pool.add([HILL, unknown_name], Origin.INIT, added_at=3)
    with pytest.raises(HypothesisError):
        pool.add([HILL, Hypothesis("bad", "C_A +", ())], Origin.INIT, added_at=3)
    assert names(pool) == ["michaelis_menten"]


def test_n_free_counts_only_free_parameters():
    # D20: `fit` holds a parameter with low == high fixed, so the BIC does not count it
    half_fixed = Hypothesis(
        "half", "V * C_A / (K + C_A)", (Param("V", 1.0, 5.0), Param("K", 2.0, 2.0))
    )
    all_fixed = Hypothesis("all", "a * C_A + b", (Param("a", 1.0, 1.0), Param("b", 0.0, 0.0)))
    no_params = Hypothesis("none", "C_A / (1 + C_A)", ())
    pool = pool_of(MM, half_fixed, all_fixed, no_params, HILL)
    assert [c.n_free for c in pool] == [2, 1, 0, 0, 3]
    n_free = pool.n_free
    np.testing.assert_array_equal(n_free, [2, 1, 0, 0, 3])
    assert np.issubdtype(n_free.dtype, np.integer)
    assert Pool().n_free.shape == (0,)


def test_fit_arrays_need_a_refit():
    X, y = mm_data()
    pool = pool_of(MM)
    for prop in ("rss", "fit_err"):
        with pytest.raises(RuntimeError, match="michaelis_menten"):
            getattr(pool, prop)
    refit(pool, X, y)
    assert pool.rss.shape == pool.fit_err.shape == (1,)
    pool.add([HILL], Origin.EXPAND, added_at=25)  # fitted and unfitted never mix silently
    for prop in ("rss", "fit_err"):
        with pytest.raises(RuntimeError, match="hill"):
            getattr(pool, prop)
    assert Pool().rss.shape == Pool().fit_err.shape == (0,)


def test_refit_fits_every_candidate():
    X, y = mm_data()
    pool = pool_of(MM, LINEAR, HILL)
    refit(pool, X, y)
    mm, linear, hill = pool
    assert mm.fit is not None and linear.fit is not None and hill.fit is not None
    assert mm.fit.ok and linear.fit.ok and hill.fit.ok
    np.testing.assert_allclose(mm.fit.theta, MM_TRUE, rtol=1e-4)
    np.testing.assert_allclose(mm.fit.y_hat, y, rtol=1e-4)
    np.testing.assert_allclose(hill.fit.theta, [*MM_TRUE, 1.0], rtol=1e-4)  # n = 1 is MM
    np.testing.assert_array_equal(pool.rss, [mm.fit.rss, linear.fit.rss, hill.fit.rss])
    assert pool.rss[0] < 1e-8 and pool.rss[2] < 1e-8
    assert pool.rss[1] > 100.0  # a line cannot saturate
    # The originals are untouched: refit replaces the candidates
    assert [c.hypothesis for c in pool] == [MM, LINEAR, HILL]
    assert [(c.origin, c.added_at) for c in pool] == [(Origin.INIT, 3)] * 3


def test_fit_err_is_the_median_relative_residual():
    X, y = mm_data()
    pool = pool_of(MM, LINEAR)
    refit(pool, X, y)
    assert pool.fit_err[0] < 1e-6
    linear = pool[1]
    assert linear.fit is not None
    rel = np.abs(y - linear.fit.y_hat) / np.maximum(np.abs(y), 0.01)
    assert linear.fit_err == pytest.approx(float(np.median(rel)))
    assert pool.fit_err[1] == linear.fit_err > 0.05
    # A larger floor c shrinks the relative residual of the points below it
    refit(pool, X, y, rel_floor=1e3)
    assert pool.fit_err[1] == pytest.approx(float(np.median(np.abs(y - linear.fit.y_hat))) / 1e3)


def test_failed_fit_has_infinite_rss_and_fit_err():
    X, y = mm_data()
    pool = pool_of(BROKEN, MM)
    refit(pool, X, y)
    broken = pool[0]
    assert broken.fit is not None and not broken.fit.ok
    assert pool.rss[0] == np.inf
    assert pool.fit_err[0] == np.inf
    # It gets weight 0 under both BIC variants and is the first to be pruned
    for bic_fn in (bic_profiled, bic_fixed):
        bic = bic_fn(pool.rss, pool.n_free, len(y))
        np.testing.assert_array_equal(weights(bic), [0.0, 1.0])
        np.testing.assert_array_equal(prune_top(log_weights(bic), 1)[0], [1])


def test_refit_warm_starts_from_the_previous_estimate():
    X, _ = mm_data(10)
    y = 2.0 * X[:, CA]
    pool = pool_of(FLAT)
    refit(pool, X, y, seed=1, n_starts=1)
    first = pool[0].fit
    assert first is not None
    j = first.theta[1]
    # Without random starts, the fit has nowhere to start but the previous estimate
    refit(pool, X, y, seed=2, n_starts=0)
    second = pool[0].fit
    assert second is not None and second is not first
    assert second.theta[1] == pytest.approx(j)
    assert second.theta[0] == pytest.approx(2.0)


def test_refit_that_raises_leaves_the_pool_unchanged():
    X, y = mm_data()
    pool = pool_of(MM)
    refit(pool, X, y)
    before = pool[0]
    pool.add([HILL], Origin.EXPAND, added_at=25)
    with pytest.raises(ValueError, match="no starts"):
        refit(pool, X, y, n_starts=0)  # HILL has no previous estimate to start from
    assert pool[0] is before
    assert pool[1].fit is None


def test_refit_is_deterministic_for_a_given_rng():
    X, y = mm_data()
    y = y * np.exp(np.random.default_rng(1).normal(0.0, 0.2, size=y.shape))
    a, b = pool_of(MM, HILL, LINEAR), pool_of(MM, HILL, LINEAR)
    refit(a, X, y, seed=7)
    refit(b, X, y, seed=7)
    np.testing.assert_array_equal(a.rss, b.rss)
    for ca, cb in zip(a, b, strict=True):
        assert ca.fit is not None and cb.fit is not None
        np.testing.assert_array_equal(ca.fit.theta, cb.fit.theta)


def test_refit_uses_all_the_data():
    X, y = mm_data()
    pool = pool_of(MM, LINEAR)
    refit(pool, X[:3], y[:3])
    assert all(c.fit is not None and c.fit.y_hat.shape == (3,) for c in pool)
    refit(pool, X, y)
    assert all(c.fit is not None and c.fit.y_hat.shape == (25,) for c in pool)


def test_keep_preserves_pool_order_and_fits():
    X, y = mm_data()
    pool = pool_of(MM, LINEAR, HILL, FLAT)
    refit(pool, X, y)
    mm, _, hill, _ = pool
    rss = pool.rss
    pool.keep([2, 0, 2])
    assert list(pool) == [mm, hill]
    np.testing.assert_array_equal(pool.rss, rss[[0, 2]])
    pool.keep(np.array([1]))  # as returned by prune_top
    assert list(pool) == [hill]
    pool.keep([])
    assert len(pool) == 0


@pytest.mark.parametrize("indices", [[3], [-1], [0, 5]])
def test_keep_rejects_indices_outside_the_pool(indices):
    pool = pool_of(MM, LINEAR, HILL)
    with pytest.raises(IndexError):
        pool.keep(indices)
    assert len(pool) == 3


def test_removed_expression_can_be_added_again():
    pool = pool_of(MM, HILL)
    pool.keep([1])
    added = pool.add([MM], Origin.EXPAND, added_at=9)
    assert names(pool) == ["hill", "michaelis_menten"]
    assert (added[0].origin, added[0].added_at) == (Origin.EXPAND, 9)
    assert added[0].fit is None


def test_fixed_parameter_is_not_penalised():
    # Two exact fits of the same data: the law with K held fixed has one free parameter fewer,
    # so it is ahead by a factor exp(log(N) / 2) under either BIC
    X, y = mm_data()
    fixed = Hypothesis(
        "fixed", "V * C_A / (K + C_A)", (Param("V", 1e-3, 1e3), Param("K", 0.7, 0.7))
    )
    pool = pool_of(MM, fixed)
    refit(pool, X, y)
    assert np.all(pool.rss < 1e-8)
    np.testing.assert_array_equal(pool.n_free, [2, 1])
    for bic_fn in (bic_profiled, bic_fixed):
        w = weights(bic_fn(pool.rss, pool.n_free, len(y)))
        assert w[1] / w[0] == pytest.approx(math.sqrt(len(y)), rel=1e-6)


def test_exact_fits_at_b3_tie_and_the_parameter_count_decides():
    # D2 on the 3 passive points of the Michaelis-Menten world, noise-free: the 2-parameter seed
    # and the 3-parameter Hill seed (n = 1) both fit exactly, far below the RSS floor
    y = World(WorldSpec("c0_michaelis_menten")).truth(D0)
    pool = pool_of(
        SEED_BY_WORLD["c8_hill_cooperativity"],
        SEED_BY_WORLD["c0_michaelis_menten"],
        origin=Origin.SEED,
    )
    refit(pool, D0, y)
    assert np.all(pool.rss / 3 < 1e-12)
    np.testing.assert_array_equal(pool.n_free, [3, 2])
    w = weights(bic_profiled(pool.rss, pool.n_free, 3))
    assert np.all(np.isfinite(w))
    assert w.sum() == pytest.approx(1.0)
    assert w[1] / w[0] == pytest.approx(math.sqrt(3))


def test_refit_then_weights_reproduces_alg_9():
    # 25 noisy observations of a canonical world, and the seed library as the pool. Two other
    # seeds have the inhibitor of this world, in a different place.
    world = "c1_competitive_inhibition"
    w = World(WorldSpec(world))
    X = w.heldout(0, 25)
    y = np.array([w.observe(x) for x in X])
    n = len(y)
    pool = pool_of(*SEED_LIBRARY, origin=Origin.SEED)
    refit(pool, X, y)
    true = list(SEED_BY_WORLD).index(world)

    # Alg. 9 written out per model: score = -BIC / 2, then a softmax
    fits = list(zip(pool.rss.tolist(), pool.n_free.tolist(), strict=True))
    scores = {
        bic_profiled: [-0.5 * (n * math.log(rss / n) + k * math.log(n)) for rss, k in fits],
        bic_fixed: [-0.5 * (rss + k * math.log(n)) for rss, k in fits],
    }
    for bic_fn, score in scores.items():
        top = max(score)
        expected = np.array([math.exp(s - top) for s in score])
        expected /= expected.sum()
        wts = weights(bic_fn(pool.rss, pool.n_free, n))
        np.testing.assert_allclose(wts, expected, rtol=1e-9, atol=1e-300)
        assert int(np.argmax(wts)) == true
        assert wts[true] > 0.99

    # The rest of an update (spec §4): the cap drops to 6, and the 6 best models are refitted
    log_w = log_weights(bic_profiled(pool.rss, pool.n_free, n))
    cap = adaptive_cap(np.exp(log_w), pool.fit_err[int(np.argmax(log_w))])
    assert cap == 6
    best = np.argsort(-log_w)[:cap]
    keep, log_w = prune_top(log_w, cap)
    assert true in keep
    np.testing.assert_array_equal(keep, np.sort(best))
    kept = [pool[int(i)].hypothesis for i in keep]
    pool.keep(keep)
    assert [c.hypothesis for c in pool] == kept
    assert np.exp(log_w).sum() == pytest.approx(1.0)
    refit(pool, X, y, seed=1)
    wts = weights(bic_profiled(pool.rss, pool.n_free, n))
    assert wts[list(keep).index(true)] > 0.99
