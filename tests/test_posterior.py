import math

import numpy as np
import pytest

from mda.posterior import (
    adaptive_cap,
    bic_fixed,
    bic_profiled,
    log_prior,
    log_weights,
    prune_top,
    weights,
)

LN2, LN10 = math.log(2), math.log(10)


def counts(*c: int) -> np.ndarray:
    return np.array(c, dtype=int)


def test_bic_profiled_matches_hand_computation():
    # N = 10: 10 ln(5 / 10) + 2 ln 10 and 10 ln(20 / 10) + 3 ln 10
    bic = bic_profiled(np.array([5.0, 20.0]), counts(2, 3), 10)
    np.testing.assert_allclose(bic, [-10 * LN2 + 2 * LN10, 10 * LN2 + 3 * LN10], rtol=1e-13)
    np.testing.assert_allclose(bic, [-2.326301619611361, 13.839227084581591], rtol=1e-13)


def test_bic_fixed_matches_hand_computation():
    # N = 10: 5 + 2 ln 10 and 20 + 3 ln 10
    bic = bic_fixed(np.array([5.0, 20.0]), counts(2, 3), 10)
    np.testing.assert_allclose(bic, [9.605170185988092, 26.907755278982138], rtol=1e-13)


def test_bic_penalises_each_parameter_by_log_n():
    rss = np.array([4.0, 4.0, 4.0])
    for bic_fn in (bic_profiled, bic_fixed):
        bic = bic_fn(rss, counts(0, 1, 5), 7)
        np.testing.assert_allclose(np.diff(bic), [math.log(7), 4 * math.log(7)], rtol=1e-13)


@pytest.mark.filterwarnings("error")
def test_zero_rss_is_floored():
    # D2: exact fits tie on the first term, so the parameter count decides between them
    rss = np.array([0.0, 0.0, 1e-30, 3.0])
    bic = bic_profiled(rss, counts(3, 2, 2, 2), 3)
    assert np.all(np.isfinite(bic))
    np.testing.assert_allclose(bic[:3] - 3 * math.log(1e-12), np.array([3, 2, 2]) * math.log(3))
    w = weights(bic)
    assert np.all(np.isfinite(w))
    assert w[1] == w[2]
    assert w[1] / w[0] == pytest.approx(math.sqrt(3))  # one parameter fewer: exp(log(3) / 2)
    assert w[3] < 1e-15


def test_rss_floor_is_an_argument():
    bic = bic_profiled(np.array([0.0]), counts(0), 4, rss_floor=1e-3)
    assert bic[0] == pytest.approx(4 * math.log(1e-3))


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("bic_fn", [bic_profiled, bic_fixed])
def test_non_finite_rss_gets_weight_zero(bic_fn):
    bic = bic_fn(np.array([2.0, np.inf, np.nan, 3.0]), counts(2, 1, 1, 2), 5)
    np.testing.assert_array_equal(np.isfinite(bic), [True, False, False, True])
    np.testing.assert_array_equal(bic[1:3], [np.inf, np.inf])
    w = weights(bic)
    np.testing.assert_array_equal(w[1:3], [0.0, 0.0])
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[3] > 0.0


@pytest.mark.parametrize("bic_fn", [bic_profiled, bic_fixed])
def test_bic_needs_an_observation(bic_fn):
    with pytest.raises(ValueError, match="at least one observation"):
        bic_fn(np.array([1.0]), counts(1), 0)


def test_log_prior():
    np.testing.assert_array_equal(log_prior(counts(0, 2, 5), 2.0), [0.0, -4.0, -10.0])
    np.testing.assert_array_equal(log_prior(counts(0, 2, 5)), [0.0, 0.0, 0.0])  # lam = 0 (BIC arm)


def test_weights_match_hand_computation():
    # exp(-BIC / 2) = 1, 1/3, 1/6, which sum to 3/2
    bic = np.array([0.0, 2 * math.log(3), 2 * math.log(6)])
    np.testing.assert_allclose(weights(bic), [2 / 3, 2 / 9, 1 / 9], rtol=1e-13)
    np.testing.assert_allclose(np.exp(log_weights(bic)), [2 / 3, 2 / 9, 1 / 9], rtol=1e-13)
    # A common shift of the BIC changes nothing
    np.testing.assert_allclose(weights(bic + 1234.5), [2 / 3, 2 / 9, 1 / 9], rtol=1e-10)


def test_prior_reweights():
    # Equal BIC, and p(m) ∝ exp(-lam C) with lam = ln 2: 1/2 against 1/8
    bic, c = np.zeros(2), counts(1, 3)
    np.testing.assert_allclose(weights(bic, log_prior(c, LN2)), [0.8, 0.2], rtol=1e-13)
    np.testing.assert_allclose(weights(bic, log_prior(c)), [0.5, 0.5], rtol=1e-13)
    np.testing.assert_allclose(weights(bic), [0.5, 0.5], rtol=1e-13)


def test_weights_sum_to_one_in_bic_order():
    rng = np.random.default_rng(0)
    rss = rng.uniform(1.0, 40.0, size=24)
    c = rng.integers(1, 6, size=24)
    for bic_fn in (bic_profiled, bic_fixed):
        bic = bic_fn(rss, c, 25)
        w = weights(bic)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(w >= 0.0)
        np.testing.assert_array_equal(np.argsort(-w, kind="stable"), np.argsort(bic, kind="stable"))


@pytest.mark.filterwarnings("error")
def test_huge_bic_gaps_do_not_overflow():
    bic = np.array([0.0, 1e6, -1e6])
    np.testing.assert_array_equal(weights(bic), [0.0, 0.0, 1.0])
    # The log weights keep the order that the weights lose to underflow
    np.testing.assert_allclose(log_weights(bic), [-5e5, -1e6, 0.0])


@pytest.mark.filterwarnings("error")
def test_no_finite_model_gives_all_zero_weights():
    bic = np.array([np.inf, np.nan])
    np.testing.assert_array_equal(log_weights(bic), [-np.inf, -np.inf])
    np.testing.assert_array_equal(weights(bic), [0.0, 0.0])
    assert weights(np.array([])).shape == (0,)


def test_prune_keeps_the_largest_weights():
    w = np.array([0.1, 0.4, 0.2, 0.3])
    keep, log_w = prune_top(np.log(w), 2)
    np.testing.assert_array_equal(keep, [1, 3])  # in pool order, not by weight
    np.testing.assert_allclose(np.exp(log_w), [4 / 7, 3 / 7], rtol=1e-13)  # renormalised
    keep, log_w = prune_top(np.log(w), 3)
    np.testing.assert_array_equal(keep, [1, 2, 3])
    np.testing.assert_allclose(np.exp(log_w), [4 / 9, 2 / 9, 3 / 9], rtol=1e-13)


def test_prune_keeps_everything_under_the_cap():
    log_w = np.log(np.array([0.1, 0.4, 0.2, 0.3]))
    for n in (4, 24):
        keep, kept = prune_top(log_w, n)
        np.testing.assert_array_equal(keep, [0, 1, 2, 3])
        np.testing.assert_allclose(kept, log_w, rtol=1e-13)


def test_prune_ties_go_to_the_earlier_model():
    keep, log_w = prune_top(np.log(np.full(4, 0.25)), 2)
    np.testing.assert_array_equal(keep, [0, 1])
    np.testing.assert_allclose(np.exp(log_w), [0.5, 0.5], rtol=1e-13)


@pytest.mark.filterwarnings("error")
def test_prune_drops_non_finite_models_first():
    # Fixed-scale BIC: the weights of models 1 and 3 underflow to exactly 0, like the non-finite
    # model 2, but their log weights still rank them
    bic = bic_fixed(np.array([1.0, 5000.0, np.inf, 4000.0]), counts(2, 2, 2, 2), 25)
    np.testing.assert_array_equal(weights(bic), [1.0, 0.0, 0.0, 0.0])
    log_w = log_weights(bic)
    np.testing.assert_array_equal(prune_top(log_w, 3)[0], [0, 1, 3])
    keep, kept = prune_top(log_w, 2)
    np.testing.assert_array_equal(keep, [0, 3])
    np.testing.assert_allclose(kept, [0.0, -1999.5], atol=1e-9)


@pytest.mark.filterwarnings("error")
def test_prune_edge_cases():
    log_w = log_weights(np.array([np.inf, np.inf, np.inf]))
    keep, kept = prune_top(log_w, 2)  # nothing to rank: the earlier models stay
    np.testing.assert_array_equal(keep, [0, 1])
    np.testing.assert_array_equal(kept, [-np.inf, -np.inf])
    keep, kept = prune_top(np.log(np.array([0.5, 0.5])), 0)
    assert keep.shape == kept.shape == (0,)
    keep, kept = prune_top(np.array([]), 6)
    assert keep.shape == kept.shape == (0,)
    with pytest.raises(ValueError, match="non-negative"):
        prune_top(log_w, -1)


def w_with_max(top: float) -> np.ndarray:
    return np.array([(1.0 - top) / 2, top, (1.0 - top) / 2])


@pytest.mark.parametrize(
    ("top", "fit_err", "cap"),
    [
        (0.95, 0.01, 6),
        (0.9 + 1e-9, 0.05, 6),  # just above tau_c, and a fit error of exactly tau_fit
        (0.9, 0.01, 24),  # the largest weight must exceed tau_c
        (0.9 - 1e-9, 0.01, 24),
        (0.95, 0.05 + 1e-9, 24),  # just above tau_fit
        (0.95, 0.05 - 1e-9, 6),
        (1.0, 0.0, 6),
        (0.5, 0.0, 24),
        (1.0, math.inf, 24),  # the MAP model's fit failed
        (1.0, math.nan, 24),  # the MAP model is not fitted
    ],
)
def test_adaptive_cap_thresholds(top, fit_err, cap):
    assert adaptive_cap(w_with_max(top), fit_err) == cap


def test_adaptive_cap_takes_its_thresholds_as_arguments():
    w = w_with_max(0.7)
    assert adaptive_cap(w, 0.08) == 24
    assert adaptive_cap(w, 0.08, tau_c=0.6, tau_fit=0.1, n_min=3, n_max=12) == 3
    assert adaptive_cap(w, 0.08, tau_c=0.6, n_min=3, n_max=12) == 12


def test_adaptive_cap_of_an_empty_pool():
    assert adaptive_cap(np.array([]), 0.0) == 24
