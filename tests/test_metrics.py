import numpy as np

from mda.metrics import nmsle, passed, rmsle

Y = np.array([0.1, 1.0, 10.0, 100.0])


def test_perfect_prediction():
    assert rmsle(Y, Y) == 0.0
    assert nmsle(Y, Y) == 0.0
    assert passed(Y, Y)


def test_rmsle_known_value():
    # log1p(e-1) - log1p(0)= log e = 1
    y_true = np.zeros(3)
    y_pred = np.full(3, np.e - 1)
    assert np.isclose(rmsle(y_pred, y_true), 1.0)


def test_nmsle_of_mean_predictor_is_one():
    y_pred = np.full_like(Y, np.expm1(np.mean(np.log1p(Y))))
    assert np.isclose(nmsle(y_pred, Y), 1.0)


def test_non_finite_prediction_fails():
    y_pred = Y.copy()
    y_pred[0] = np.nan
    assert rmsle(y_pred, Y) == np.inf
    assert nmsle(y_pred, Y) == np.inf
    assert not passed(y_pred, Y)
