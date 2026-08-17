import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def make_df(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"x": x, "y": y})


def _noise(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    return np.asarray(rng.normal(0.0, scale, size=n))


def linear_data(rng: np.random.Generator, noise: float = 0.1) -> pd.DataFrame:
    x = np.linspace(1, 10, 50)
    y = 3.0 * x + 5.0 + _noise(rng, len(x), noise)
    return make_df(x, y)


def quadratic_data(rng: np.random.Generator, noise: float = 0.5) -> pd.DataFrame:
    x = np.linspace(-5, 5, 60)
    y = 2.0 * x**2 - 3.0 * x + 1.0 + _noise(rng, len(x), noise)
    return make_df(x, y)


def cubic_data(rng: np.random.Generator, noise: float = 1.0) -> pd.DataFrame:
    x = np.linspace(-3, 3, 60)
    y = 0.5 * x**3 - 2.0 * x**2 + x + 4.0 + _noise(rng, len(x), noise)
    return make_df(x, y)


def exponential_data(rng: np.random.Generator, noise: float = 0.05) -> pd.DataFrame:
    x = np.linspace(0, 3, 50)
    y = 2.0 * np.exp(0.8 * x) + _noise(rng, len(x), noise)
    return make_df(x, y)


def power_law_data(rng: np.random.Generator, noise: float = 0.1) -> pd.DataFrame:
    x = np.linspace(1, 10, 50)
    y = 3.0 * x**1.5 + _noise(rng, len(x), noise)
    return make_df(x, y)


def sigmoid_data(rng: np.random.Generator, noise: float = 0.05) -> pd.DataFrame:
    x = np.linspace(-6, 6, 80)
    y = 5.0 / (1.0 + np.exp(-1.5 * (x - 1.0))) + _noise(rng, len(x), noise)
    return make_df(x, y)
