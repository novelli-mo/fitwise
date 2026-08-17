from abc import ABC, abstractmethod

import numpy as np


class CandidateModel(ABC):
    name: str

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, random_state: int | None = None) -> None: ...

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def log_likelihood(self, x: np.ndarray, y: np.ndarray) -> float: ...

    @property
    @abstractmethod
    def n_params(self) -> int: ...

    @property
    @abstractmethod
    def params(self) -> dict[str, float]: ...
