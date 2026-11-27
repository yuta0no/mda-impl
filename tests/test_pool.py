import json
import math

import numpy as np
import pytest

from mda.bench import INPUT_NAMES, World, WorldSpec, from_unit
from mda.config import DEFAULTS
from mda.fit import sigma
from mda.hypothesis import Hypothesis, HypothesisError, Param, compile_hypothesis
from mda.library import SEED_BY_WORLD, SEED_LIBRARY
from mda.pool import Origin, Pool
from mda.posterior import adaptive_cap, bic_fixed, bic_profiled, log_weights, prune_top, weights
from mda.smc import SMCResult, log_likelihood, log_model_weights, tempered_smc

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
# Michaelis-Menten and a line in boxes that 100 particles can search: the seed bounds, six
# decades wide, are beyond a tempered SMC under the uniform prior (blueprint §11)
MM_BOX = Hypothesis(
    "michaelis_menten",
    "Vmax * C_A / (Km + C_A)",
    (Param("Vmax", 1.0, 10.0), Param("Km", 0.1, 5.0)),
)
LINE_BOX = Hypothesis("linear", "k * C_A", (Param("k", 0.0, 10.0),))

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


# refit_smc: the SMC control (spec §8)


def smc_of(c) -> SMCResult:
    assert c.smc is not None
    return c.smc


def test_refit_smc_runs_the_sampler_for_every_candidate_in_pool_order():
    X, y = mm_data()
    pool = pool_of(MM_BOX, LINE_BOX, BROKEN)
    pool.refit_smc(X, y, rng=np.random.default_rng(3))
    # One generator, passed from one candidate to the next
    rng = np.random.default_rng(3)
    for c in pool:
        expected = tempered_smc(c.hypothesis, X, y, rng=rng, sigma_n=sigma(y), f=c.f)
        np.testing.assert_array_equal(smc_of(c).particles, expected.particles)
        np.testing.assert_array_equal(smc_of(c).loglik, expected.loglik)
        assert smc_of(c).log_z == expected.log_z
    np.testing.assert_array_equal(pool.log_z, [smc_of(c).log_z for c in pool])
    # The data are Michaelis-Menten's: a line does far worse, and `broken` explains nothing
    assert pool.log_z[0] > pool.log_z[1] + 100 and pool.log_z[2] == -math.inf
    assert [c.hypothesis for c in pool] == [MM_BOX, LINE_BOX, BROKEN]
    assert [(c.origin, c.added_at) for c in pool] == [(Origin.INIT, 3)] * 3


def test_point_estimate_of_refit_smc_is_the_best_fitting_particle():
    X, y = mm_data()
    pool = pool_of(MM_BOX, LINE_BOX)
    pool.refit_smc(X, y, rng=np.random.default_rng(0))
    for c in pool:
        run, fit = smc_of(c), c.fit
        assert fit is not None and fit.ok
        np.testing.assert_array_equal(fit.theta, run.particles[run.best])
        assert not np.shares_memory(fit.theta, run.particles)
        np.testing.assert_array_equal(fit.y_hat, c.f(fit.theta, X))
        # The RSS is the whitened one of `mda.fit`, the smallest among the particles, and the
        # likelihood of the SMC is exp(-RSS / 2) up to a constant
        rss = [float(np.sum(((y - c.f(theta, X)) / sigma(y)) ** 2)) for theta in run.particles]
        assert fit.rss == pytest.approx(min(rss), rel=1e-12)
        const = log_likelihood(y, y, sigma(y))
        assert run.loglik.max() == pytest.approx(const - 0.5 * fit.rss, rel=1e-12)
        rel = np.abs(y - fit.y_hat) / np.maximum(np.abs(y), 0.01)
        assert c.fit_err == pytest.approx(float(np.median(rel)))
    # What reads a fit reads these: the pool's arrays, as after `refit`
    np.testing.assert_array_equal(pool.rss, [c.fit.rss for c in pool if c.fit is not None])
    np.testing.assert_array_equal(pool.fit_err, [c.fit_err for c in pool])
    # A posterior draw near the mode, not the mode: close to the truth, and not on it
    mm = pool[0].fit
    assert mm is not None
    np.testing.assert_allclose(mm.theta, MM_TRUE, rtol=0.05)
    assert 0 < mm.rss < 5 and 0 < pool.fit_err[0] < 0.02
    assert pool.fit_err[1] > 0.05  # a line cannot saturate
    # A larger floor c shrinks the relative residual, as it does for `refit`
    pool.refit_smc(X, y, rng=np.random.default_rng(0), rel_floor=1e3)
    assert pool[0].fit_err == pytest.approx(float(np.median(np.abs(y - mm.y_hat))) / 1e3)


def test_candidate_without_evidence_has_a_failed_fit():
    X, y = mm_data()
    pool = pool_of(BROKEN, MM_BOX)
    pool.refit_smc(X, y, rng=np.random.default_rng(0))
    broken = pool[0]
    assert broken.fit is not None and not broken.fit.ok and not smc_of(broken).ok
    assert pool.log_z[0] == -math.inf and pool.rss[0] == np.inf and pool.fit_err[0] == np.inf
    # It gets weight 0 and is the first to be pruned
    log_w = log_model_weights(pool.log_z, pool.n_free, 2.0)
    np.testing.assert_array_equal(np.exp(log_w), [0.0, 1.0])
    np.testing.assert_array_equal(prune_top(log_w, 1)[0], [1])


def test_refit_smc_passes_its_settings_to_the_sampler():
    X, y = mm_data()

    def reference(
        *,
        n_particles: int = 40,
        ess_fraction: float = 0.7,
        n_moves: int = 2,
        max_rungs: int = 3,
        step_scale: float = 0.3,
        # Above any standard deviation the particles can have in these boxes, so that the
        # floor sets the proposal scale
        step_floor: float = 5.0,
        log_span: float | None = 5.0,
        # The rates run from 0.06 to 2.5, so each of the two sets part of the noise scale
        rho: float = 0.5,
        sigma_floor: float = 0.5,
    ) -> SMCResult:
        """The run of `tempered_smc` itself, with every setting away from its default."""
        return tempered_smc(
            MM_BOX,
            X,
            y,
            rng=np.random.default_rng(0),
            n_particles=n_particles,
            ess_fraction=ess_fraction,
            n_moves=n_moves,
            max_rungs=max_rungs,
            step_scale=step_scale,
            step_floor=step_floor,
            log_span=log_span,
            sigma_n=sigma(y, rho, sigma_floor),
        )

    pool = pool_of(MM_BOX)
    pool.refit_smc(
        X,
        y,
        rng=np.random.default_rng(0),
        n_particles=40,
        ess_fraction=0.7,
        n_moves=2,
        max_rungs=3,
        step_scale=0.3,
        step_floor=5.0,
        log_span=5.0,
        rho=0.5,
        sigma_floor=0.5,
    )
    run = smc_of(pool[0])
    np.testing.assert_array_equal(run.particles, reference().particles)
    assert run.log_z == reference().log_z
    # No setting is idle here: with any one of them at its default, the run is another one
    d = DEFAULTS
    others = [
        reference(n_particles=d.n_particles),
        reference(ess_fraction=d.ess_fraction),
        reference(n_moves=d.n_moves),
        reference(max_rungs=d.max_rungs),
        reference(step_scale=d.step_scale),
        reference(step_floor=d.step_floor),
        reference(log_span=d.log_span),
        reference(log_span=None),
        reference(rho=d.rho),
        reference(sigma_floor=d.sigma_floor),
    ]
    assert all(other.log_z != run.log_z for other in others)
    # The fit's RSS uses the noise scale of the likelihood
    fit = pool[0].fit
    assert fit is not None
    expected = float(np.sum(((y - fit.y_hat) / sigma(y, 0.5, 0.5)) ** 2))
    assert fit.rss == pytest.approx(expected, rel=1e-12)


def test_evidence_needs_a_refit_smc():
    X, y = mm_data()
    pool = pool_of(MM_BOX)
    with pytest.raises(RuntimeError, match="michaelis_menten"):
        _ = pool.log_z
    pool.refit_smc(X, y, rng=np.random.default_rng(0))
    assert pool.log_z.shape == (1,)
    pool.add([LINE_BOX], Origin.EXPAND, added_at=25)  # with and without evidence never mix
    with pytest.raises(RuntimeError, match="linear"):
        _ = pool.log_z
    # A least-squares refit replaces the point estimate, so the particles are no longer its
    # posterior and are dropped
    refit(pool, X, y)
    assert [c.smc for c in pool] == [None, None]
    with pytest.raises(RuntimeError, match="michaelis_menten.*linear"):
        _ = pool.log_z
    assert Pool().log_z.shape == (0,)


def test_refit_smc_that_raises_leaves_the_pool_unchanged():
    X, y = mm_data()
    pool = pool_of(MM_BOX, LINE_BOX)
    pool.refit_smc(X, y, rng=np.random.default_rng(0))
    before = list(pool)
    with pytest.raises(ValueError, match="n_particles"):
        pool.refit_smc(X, y, rng=np.random.default_rng(1), n_particles=0)
    assert list(pool) == before


def test_keep_preserves_the_particles():
    X, y = mm_data()
    pool = pool_of(MM_BOX, BROKEN, LINE_BOX)
    pool.refit_smc(X, y, rng=np.random.default_rng(0))
    mm, _, line = pool
    log_z = pool.log_z
    pool.keep([0, 2])
    assert list(pool) == [mm, line] and pool[1].smc is line.smc
    np.testing.assert_array_equal(pool.log_z, log_z[[0, 2]])


def test_refit_smc_starts_from_the_prior_each_time():
    # Nothing is carried over: a second run with the same generator state repeats the first,
    # whatever the candidate held before
    X, y = mm_data()
    pool = pool_of(MM_BOX)
    pool.refit_smc(X[:3], y[:3], rng=np.random.default_rng(0))
    assert smc_of(pool[0]).loglik.shape == (DEFAULTS.n_particles,)
    pool.refit_smc(X, y, rng=np.random.default_rng(5))
    fresh = pool_of(MM_BOX)
    fresh.refit_smc(X, y, rng=np.random.default_rng(5))
    np.testing.assert_array_equal(smc_of(pool[0]).particles, smc_of(fresh[0]).particles)
    assert pool[0].fit is not None and pool[0].fit.y_hat.shape == (25,)
