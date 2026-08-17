from fitwise.results import CurveResult, GroupResult, ModelFit
from fitwise.selection.criteria import aic, akaike_weights, bic


def build_group_result(curve_results: list[CurveResult]) -> GroupResult:
    model_names = _models_with_any_success(curve_results)
    combined_lls = _sum_log_likelihoods(curve_results, model_names)
    combined_n_obs = _sum_n_obs(curve_results, model_names)
    n_params_map = _n_params_map(curve_results, model_names)
    ranking = _group_ranking(model_names, combined_lls, combined_n_obs, n_params_map)
    gain = _confidence_gain(ranking, curve_results)
    return GroupResult(ranking=ranking, curve_results=curve_results, confidence_gain=gain)


def _models_with_any_success(curve_results: list[CurveResult]) -> list[str]:
    names: set[str] = set()
    for cr in curve_results:
        for mf in cr.ranking:
            names.add(mf.name)
    return sorted(names)


def _sum_log_likelihoods(
    curve_results: list[CurveResult], model_names: list[str]
) -> dict[str, float]:
    totals = {name: 0.0 for name in model_names}
    for cr in curve_results:
        for mf in cr.ranking:
            if mf.name in totals:
                totals[mf.name] += mf.log_likelihood
    return totals


def _sum_n_obs(curve_results: list[CurveResult], model_names: list[str]) -> dict[str, int]:
    totals: dict[str, int] = {name: 0 for name in model_names}
    for cr in curve_results:
        for mf in cr.ranking:
            if mf.name in totals:
                totals[mf.name] += mf.n_obs
    return totals


def _n_params_map(curve_results: list[CurveResult], model_names: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for cr in curve_results:
        for mf in cr.ranking:
            if mf.name in model_names and mf.name not in result:
                result[mf.name] = mf.n_params
    return result


def _group_ranking(
    model_names: list[str],
    combined_lls: dict[str, float],
    combined_n_obs: dict[str, int],
    n_params_map: dict[str, int],
) -> list[ModelFit]:
    aics = [aic(combined_lls[name], n_params_map[name]) for name in model_names]
    bics = [
        bic(combined_lls[name], n_params_map[name], combined_n_obs[name]) for name in model_names
    ]
    weights = akaike_weights(aics)

    results = [
        ModelFit(
            name=name,
            params={},
            aic=a,
            bic=b,
            akaike_weight=w,
            r_squared=float("nan"),
            log_likelihood=combined_lls[name],
            n_params=n_params_map[name],
            n_obs=combined_n_obs[name],
        )
        for name, a, b, w in zip(model_names, aics, bics, weights, strict=True)
    ]
    return sorted(results, key=lambda m: m.aic)


def _confidence_gain(ranking: list[ModelFit], curve_results: list[CurveResult]) -> float:
    if not ranking:
        return float("nan")
    winner_name = ranking[0].name
    group_weight = ranking[0].akaike_weight
    best_individual = _best_individual_weight(winner_name, curve_results)
    if best_individual <= 0.0:
        return float("nan")
    return group_weight / best_individual


def _best_individual_weight(winner_name: str, curve_results: list[CurveResult]) -> float:
    weights = []
    for cr in curve_results:
        for mf in cr.ranking:
            if mf.name == winner_name:
                weights.append(mf.akaike_weight)
                break
    return max(weights) if weights else 0.0
