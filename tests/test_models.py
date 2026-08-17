import numpy as np
import pytest

from fitwise._errors import ConvergenceError
from fitwise.models.cubic import CubicModel
from fitwise.models.exponential import ExponentialModel
from fitwise.models.linear import LinearModel
from fitwise.models.power_law import PowerLawModel
from fitwise.models.quadratic import QuadraticModel
from fitwise.models.sigmoid import SigmoidModel
from tests.conftest import (
    cubic_data,
    exponential_data,
    linear_data,
    power_law_data,
    quadratic_data,
    sigmoid_data,
)


def test_linear_fit(rng):
    df = linear_data(rng, noise=0.01)
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    m = LinearModel()
    m.fit(x, y)
    assert abs(m.params["a"] - 3.0) < 0.1
    assert abs(m.params["b"] - 5.0) < 0.1
    assert m.n_params == 2
    ll = m.log_likelihood(x, y)
    assert ll > -1000


def test_quadratic_fit(rng):
    df = quadratic_data(rng, noise=0.05)
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    m = QuadraticModel()
    m.fit(x, y)
    assert abs(m.params["a"] - 2.0) < 0.2
    assert abs(m.params["b"] - (-3.0)) < 0.2
    assert m.n_params == 3


def test_cubic_fit(rng):
    df = cubic_data(rng, noise=0.1)
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    m = CubicModel()
    m.fit(x, y)
    assert abs(m.params["a"] - 0.5) < 0.1
    assert m.n_params == 4


def test_exponential_fit(rng):
    df = exponential_data(rng, noise=0.01)
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    m = ExponentialModel()
    m.fit(x, y, random_state=0)
    assert abs(m.params["a"] - 2.0) < 0.2
    assert abs(m.params["b"] - 0.8) < 0.1
    assert m.n_params == 2


def test_power_law_fit(rng):
    df = power_law_data(rng, noise=0.01)
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    m = PowerLawModel()
    m.fit(x, y, random_state=0)
    assert abs(m.params["a"] - 3.0) < 0.3
    assert abs(m.params["b"] - 1.5) < 0.1
    assert m.n_params == 2


def test_power_law_rejects_nonpositive_x(rng):
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = PowerLawModel()
    with pytest.raises(ConvergenceError, match="x > 0"):
        m.fit(x, y)


def test_sigmoid_fit(rng):
    df = sigmoid_data(rng, noise=0.01)
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    m = SigmoidModel()
    m.fit(x, y, random_state=0)
    assert abs(m.params["L"] - 5.0) < 0.5
    assert abs(m.params["x0"] - 1.0) < 0.5
    assert m.n_params == 3


def test_predict_shape(rng):
    df = linear_data(rng)
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    m = LinearModel()
    m.fit(x, y)
    assert m.predict(x).shape == x.shape


def test_log_likelihood_is_float(rng):
    df = linear_data(rng)
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    m = LinearModel()
    m.fit(x, y)
    assert isinstance(m.log_likelihood(x, y), float)
