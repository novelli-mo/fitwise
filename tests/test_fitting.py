import warnings

import numpy as np
import pytest

from fitwise._errors import FitError
from fitwise.fitting.curve import fit_curve_from_arrays
from fitwise.models import instantiate_models
from tests.conftest import linear_data, quadratic_data


def test_fit_curve_returns_curve_result(rng):
    df = linear_data(rng)
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    models = instantiate_models(["linear", "quadratic"])
    result = fit_curve_from_arrays(x, y, models)
    assert result.ranking
    assert result.best_model.name in ("linear", "quadratic")


def test_ranking_ordered_by_aic(rng):
    df = linear_data(rng)
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    models = instantiate_models(["linear", "quadratic", "cubic"])
    result = fit_curve_from_arrays(x, y, models)
    aics = [m.aic for m in result.ranking]
    assert aics == sorted(aics)


def test_failed_model_recorded(rng):
    x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    models = instantiate_models(["linear", "power_law"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = fit_curve_from_arrays(x, y, models)
    assert any("power_law" in m.name for m in result.failed_models)
    assert result.best_model.name == "linear"
    assert len(w) >= 1


def test_fit_error_when_all_fail():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    # Force all models to fail by giving a dict of broken models
    from fitwise._errors import ConvergenceError
    from fitwise.models.base import CandidateModel

    class AlwaysFailModel(CandidateModel):
        name = "broken"

        def fit(self, x, y, random_state=None):
            raise ConvergenceError("always fails")

        def predict(self, x):
            return x

        def log_likelihood(self, x, y):
            return 0.0

        @property
        def n_params(self):
            return 1

        @property
        def params(self):
            return {}

    models = {"broken": AlwaysFailModel()}
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with pytest.raises(FitError):
            fit_curve_from_arrays(x, y, models)


def test_akaike_weights_sum_to_one(rng):
    df = quadratic_data(rng)
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    models = instantiate_models(["linear", "quadratic", "cubic"])
    result = fit_curve_from_arrays(x, y, models)
    total = sum(m.akaike_weight for m in result.ranking)
    assert abs(total - 1.0) < 1e-10
