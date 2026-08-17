import numpy as np
from scipy.optimize import curve_fit

from fitwise._errors import ConvergenceError
from fitwise.models._shared import gaussian_log_likelihood
from fitwise.models.base import CandidateModel


def _sigmoid_func(x: np.ndarray, amplitude: float, k: float, x0: float) -> np.ndarray:
    return amplitude / (1.0 + np.exp(-k * (x - x0)))  # type: ignore[return-value, no-any-return]


def _initial_guess(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> list[float]:
    amplitude0 = float(np.max(y) - np.min(y)) + float(rng.normal(0, 0.1))
    x0_0 = float(x[np.argmax(np.abs(np.gradient(y, x)))])
    x_range = float(x[-1] - x[0])
    k0 = 4.0 / x_range if x_range > 0 else 1.0
    return [amplitude0, k0, x0_0]


class SigmoidModel(CandidateModel):
    name = "sigmoid"

    def __init__(self) -> None:
        self._amplitude: float | None = None
        self._k: float | None = None
        self._x0: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, random_state: int | None = None) -> None:
        rng = np.random.default_rng(random_state)
        p0 = _initial_guess(x, y, rng)
        try:
            popt, _ = curve_fit(_sigmoid_func, x, y, p0=p0, maxfev=10000)
            self._amplitude = float(popt[0])
            self._k = float(popt[1])
            self._x0 = float(popt[2])
        except RuntimeError as e:
            raise ConvergenceError(str(e)) from e

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid_func(x, self._amplitude, self._k, self._x0)  # type: ignore[arg-type]

    def log_likelihood(self, x: np.ndarray, y: np.ndarray) -> float:
        return gaussian_log_likelihood(y, self.predict(x))

    @property
    def n_params(self) -> int:
        return 3

    @property
    def params(self) -> dict[str, float]:
        assert self._amplitude is not None and self._k is not None and self._x0 is not None
        return {"L": self._amplitude, "k": self._k, "x0": self._x0}
