import warnings

import numpy as np

from fitwise._errors import ConvergenceError, FitError
from fitwise.models.base import CandidateModel
from fitwise.results import CurveResult, FailedModel, ModelFit
from fitwise.selection.criteria import aic, akaike_weights, bic, r_squared


def fit_curve_from_arrays(
    x: np.ndarray,
    y: np.ndarray,
    models: dict[str, CandidateModel],
    random_state: int | None = None,
) -> CurveResult:
    fitted, failed = _fit_all(x, y, models, random_state)
    if not fitted:
        raise FitError(
            f"No model converged for this curve (n={len(y)}). "
            f"Failed models: {[f.name for f in failed]}. "
            f"Check that the data has sufficient range and no pathological values."
        )
    ranking = _build_ranking(x, y, fitted)
    return CurveResult(ranking=ranking, failed_models=failed)


def _fit_all(
    x: np.ndarray,
    y: np.ndarray,
    models: dict[str, CandidateModel],
    random_state: int | None,
) -> tuple[list[tuple[str, CandidateModel]], list[FailedModel]]:
    fitted: list[tuple[str, CandidateModel]] = []
    failed: list[FailedModel] = []
    for name, model in models.items():
        result = _try_fit(name, model, x, y, random_state)
        if isinstance(result, FailedModel):
            failed.append(result)
        else:
            fitted.append(result)
    return fitted, failed


def _try_fit(
    name: str,
    model: CandidateModel,
    x: np.ndarray,
    y: np.ndarray,
    random_state: int | None,
) -> tuple[str, CandidateModel] | FailedModel:
    try:
        model.fit(x, y, random_state=random_state)
        return (name, model)
    except ConvergenceError as e:
        warnings.warn(f"Model '{name}' failed to converge: {e}", stacklevel=4)
        return FailedModel(name=name, reason=str(e))


def _build_ranking(
    x: np.ndarray,
    y: np.ndarray,
    fitted: list[tuple[str, CandidateModel]],
) -> list[ModelFit]:
    n = len(y)
    lls = [model.log_likelihood(x, y) for _, model in fitted]
    aics = [aic(ll, model.n_params) for (_, model), ll in zip(fitted, lls, strict=True)]
    bics = [bic(ll, model.n_params, n) for (_, model), ll in zip(fitted, lls, strict=True)]
    weights = akaike_weights(aics)

    results = [
        ModelFit(
            name=name,
            params=model.params,
            aic=a,
            bic=b,
            akaike_weight=w,
            r_squared=r_squared(y, model.predict(x)),
            log_likelihood=ll,
            n_params=model.n_params,
            n_obs=n,
        )
        for (name, model), ll, a, b, w in zip(fitted, lls, aics, bics, weights, strict=True)
    ]
    return sorted(results, key=lambda m: m.aic)
