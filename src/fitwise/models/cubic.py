import numpy as np

from fitwise._errors import ConvergenceError
from fitwise.models._shared import gaussian_log_likelihood
from fitwise.models.base import CandidateModel


class CubicModel(CandidateModel):
    name = "cubic"

    def __init__(self) -> None:
        self._coeffs: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, random_state: int | None = None) -> None:
        try:
            self._coeffs = np.polyfit(x, y, 3)
        except np.linalg.LinAlgError as e:
            raise ConvergenceError(str(e)) from e

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.polyval(self._coeffs, x)  # type: ignore[arg-type]

    def log_likelihood(self, x: np.ndarray, y: np.ndarray) -> float:
        return gaussian_log_likelihood(y, self.predict(x))

    @property
    def n_params(self) -> int:
        return 4

    @property
    def params(self) -> dict[str, float]:
        a, b, c, d = self._coeffs  # type: ignore[misc]
        return {"a": float(a), "b": float(b), "c": float(c), "d": float(d)}
