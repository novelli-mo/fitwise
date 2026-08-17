from fitwise.models.base import CandidateModel
from fitwise.models.cubic import CubicModel
from fitwise.models.exponential import ExponentialModel
from fitwise.models.linear import LinearModel
from fitwise.models.power_law import PowerLawModel
from fitwise.models.quadratic import QuadraticModel
from fitwise.models.sigmoid import SigmoidModel

VALID_MODELS: frozenset[str] = frozenset(
    ["linear", "quadratic", "cubic", "exponential", "power_law", "sigmoid"]
)

_REGISTRY: dict[str, type[CandidateModel]] = {
    "linear": LinearModel,
    "quadratic": QuadraticModel,
    "cubic": CubicModel,
    "exponential": ExponentialModel,
    "power_law": PowerLawModel,
    "sigmoid": SigmoidModel,
}


def validate_model_names(names: list[str]) -> None:
    invalid = sorted(set(names) - VALID_MODELS)
    if invalid:
        raise ValueError(
            f"Invalid model name(s): {invalid}. "
            f"Valid names are: {sorted(VALID_MODELS)}. "
            f"See docs/design.md for the API contract."
        )


def instantiate_models(names: list[str] | None) -> dict[str, CandidateModel]:
    resolved = sorted(VALID_MODELS) if names is None else names
    validate_model_names(resolved)
    return {name: _REGISTRY[name]() for name in resolved}
