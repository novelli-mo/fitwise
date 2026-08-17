import numpy as np


def aic(log_likelihood: float, n_params: int) -> float:
    return 2.0 * n_params - 2.0 * log_likelihood


def bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    return float(n_params * np.log(n_obs) - 2.0 * log_likelihood)


def akaike_weights(aics: list[float]) -> list[float]:
    arr = np.array(aics)
    delta = arr - np.min(arr)
    unnorm = np.exp(-delta / 2.0)
    return list(unnorm / np.sum(unnorm))


def r_squared(y: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return 1.0
    return 1.0 - ss_res / ss_tot
