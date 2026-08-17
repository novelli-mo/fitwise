import numpy as np
import pytest

from fitwise.selection.criteria import aic, akaike_weights, bic, r_squared


def test_aic_penalizes_more_params():
    ll = -50.0
    assert aic(ll, 2) < aic(ll, 4)


def test_bic_penalizes_more_params():
    ll = -50.0
    assert bic(ll, 2, 100) < bic(ll, 4, 100)


def test_akaike_weights_sum_to_one():
    aics = [10.0, 12.0, 15.0, 20.0]
    weights = akaike_weights(aics)
    assert abs(sum(weights) - 1.0) < 1e-10


def test_akaike_weights_best_model_has_highest_weight():
    aics = [10.0, 12.0, 15.0]
    weights = akaike_weights(aics)
    assert weights[0] == max(weights)


def test_akaike_weights_single_model():
    weights = akaike_weights([5.0])
    assert weights == [1.0]


def test_r_squared_perfect_fit():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(y, y) == pytest.approx(1.0)


def test_r_squared_mean_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full_like(y, y.mean())
    assert r_squared(y, y_pred) == pytest.approx(0.0)


def test_r_squared_constant_y():
    y = np.array([3.0, 3.0, 3.0])
    y_pred = np.array([3.0, 3.0, 3.0])
    assert r_squared(y, y_pred) == 1.0


def test_aic_better_fit_wins():
    # same params, better fit (higher ll) → lower AIC
    assert aic(-10.0, 2) < aic(-20.0, 2)
