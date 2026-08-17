import numpy as np
from scipy.optimize import curve_fit

from fitwise._errors import ConvergenceError
from fitwise.models._shared import gaussian_log_likelihood
from fitwise.models.base import CandidateModel


def _exp_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.exp(b * x)  # type: ignore[return-value, no-any-return]


def _initial_guess(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> list[float]:
    safe_y = np.where(y > 0, y, 1e-10)
    try:
        coeffs: list[float] = list(np.polyfit(x, np.log(safe_y), 1))
        a0 = float(np.exp(coeffs[1])) + float(rng.normal(0, 0.1))
        b0 = float(coeffs[0]) + float(rng.normal(0, 0.01))
    except Exception:
        a0, b0 = float(rng.uniform(0.1, 10)), float(rng.uniform(-1, 1))
    return [a0, b0]


class ExponentialModel(CandidateModel):
    name = "exponential"

    def __init__(self) -> None:
        self._a: float | None = None
        self._b: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, random_state: int | None = None) -> None:
        rng = np.random.default_rng(random_state)
        p0 = _initial_guess(x, y, rng)
        try:
            popt, _ = curve_fit(_exp_func, x, y, p0=p0, maxfev=10000)
            self._a, self._b = float(popt[0]), float(popt[1])
        except RuntimeError as e:
            raise ConvergenceError(str(e)) from e

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _exp_func(x, self._a, self._b)  # type: ignore[arg-type]

    def log_likelihood(self, x: np.ndarray, y: np.ndarray) -> float:
        return gaussian_log_likelihood(y, self.predict(x))

    @property
    def n_params(self) -> int:
        return 2

    @property
    def params(self) -> dict[str, float]:
        assert self._a is not None and self._b is not None
        return {"a": self._a, "b": self._b}
