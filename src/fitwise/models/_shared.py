import numpy as np


def gaussian_log_likelihood(y: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y)
    residuals = y - y_pred
    sigma_sq = float(np.sum(residuals**2) / n)
    if sigma_sq <= 0.0:
        return 1e10
    return float(-n / 2.0 * (np.log(2 * np.pi * sigma_sq) + 1.0))
