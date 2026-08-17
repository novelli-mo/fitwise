from dataclasses import dataclass, field


@dataclass
class ModelFit:
    name: str
    params: dict[str, float]
    aic: float
    bic: float
    akaike_weight: float
    r_squared: float
    log_likelihood: float
    n_params: int
    n_obs: int


@dataclass
class FailedModel:
    name: str
    reason: str


@dataclass
class CurveResult:
    ranking: list[ModelFit]
    failed_models: list[FailedModel] = field(default_factory=list)

    @property
    def best_model(self) -> ModelFit:
        return self.ranking[0]


@dataclass
class GroupResult:
    ranking: list[ModelFit]
    curve_results: list[CurveResult]
    confidence_gain: float

    def summary(self) -> dict[str, object]:
        n_curves = len(self.curve_results)
        n_successful = sum(1 for cr in self.curve_results if cr.ranking)
        model_success = _count_model_success(self.curve_results)
        model_failures = _collect_model_failures(self.curve_results)
        best = self.ranking[0].name if self.ranking else None

        return {
            "n_curves": n_curves,
            "n_successful_curves": n_successful,
            "best_model": best,
            "confidence_gain": self.confidence_gain,
            "model_success_rate": {name: count / n_curves for name, count in model_success.items()},
            "model_failures": model_failures,
        }


def _count_model_success(curve_results: list[CurveResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cr in curve_results:
        for mf in cr.ranking:
            counts[mf.name] = counts.get(mf.name, 0) + 1
    return counts


def _collect_model_failures(curve_results: list[CurveResult]) -> dict[str, list[str]]:
    failures: dict[str, list[str]] = {}
    for cr in curve_results:
        for fm in cr.failed_models:
            failures.setdefault(fm.name, []).append(fm.reason)
    return failures
