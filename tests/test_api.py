import pandas as pd
import pytest

import fitwise
from fitwise.results import CurveResult, GroupResult
from tests.conftest import (
    cubic_data,
    exponential_data,
    linear_data,
    power_law_data,
    quadratic_data,
    sigmoid_data,
)

# --- fit_curve: each functional form is identified as best ---


def test_fit_curve_identifies_linear(rng):
    result = fitwise.fit_curve(linear_data(rng, noise=0.01), random_state=0)
    assert isinstance(result, CurveResult)
    assert result.best_model.name == "linear"


def test_fit_curve_identifies_quadratic(rng):
    result = fitwise.fit_curve(quadratic_data(rng, noise=0.05), random_state=0)
    assert result.best_model.name == "quadratic"


def test_fit_curve_identifies_cubic(rng):
    result = fitwise.fit_curve(cubic_data(rng, noise=0.1), random_state=0)
    assert result.best_model.name == "cubic"


def test_fit_curve_identifies_exponential(rng):
    result = fitwise.fit_curve(
        exponential_data(rng, noise=0.01),
        models=["linear", "exponential", "quadratic"],
        random_state=0,
    )
    assert result.best_model.name == "exponential"


def test_fit_curve_identifies_power_law(rng):
    result = fitwise.fit_curve(
        power_law_data(rng, noise=0.01),
        models=["linear", "power_law", "quadratic"],
        random_state=0,
    )
    assert result.best_model.name == "power_law"


def test_fit_curve_identifies_sigmoid(rng):
    result = fitwise.fit_curve(
        sigmoid_data(rng, noise=0.01),
        models=["linear", "sigmoid", "quadratic"],
        random_state=0,
    )
    assert result.best_model.name == "sigmoid"


# --- fit_curve: structural guarantees ---


def test_fit_curve_returns_all_models_in_ranking(rng):
    result = fitwise.fit_curve(linear_data(rng), models=["linear", "quadratic"], random_state=0)
    names = {m.name for m in result.ranking}
    assert names == {"linear", "quadratic"}


def test_fit_curve_best_model_shortcut(rng):
    result = fitwise.fit_curve(linear_data(rng), models=["linear", "quadratic"])
    assert result.best_model is result.ranking[0]


def test_fit_curve_model_fit_fields(rng):
    result = fitwise.fit_curve(linear_data(rng), models=["linear"])
    mf = result.best_model
    assert isinstance(mf.aic, float)
    assert isinstance(mf.bic, float)
    assert 0.0 <= mf.akaike_weight <= 1.0
    assert mf.r_squared <= 1.0
    assert isinstance(mf.params, dict)


# --- fit_curve: invalid model name raises immediately ---


def test_invalid_model_name_raises_value_error():
    with pytest.raises(ValueError, match="Invalid model name"):
        fitwise.fit_curve(pd.DataFrame({"x": [1], "y": [1]}), models=["banana"])


def test_invalid_model_name_mentions_valid_names():
    with pytest.raises(ValueError, match="linear"):
        fitwise.fit_curve(pd.DataFrame({"x": [1], "y": [1]}), models=["bad_model"])


# --- fit_group: list of DataFrames ---


def test_fit_group_returns_group_result(rng):
    dfs = [linear_data(rng, noise=0.05) for _ in range(4)]
    result = fitwise.fit_group(dfs, random_state=0)
    assert isinstance(result, GroupResult)
    assert result.ranking
    assert len(result.curve_results) == 4


def test_fit_group_single_df_with_group_column(rng):
    dfs = [linear_data(rng, noise=0.05) for _ in range(3)]
    combined = pd.concat([df.assign(group=i) for i, df in enumerate(dfs)], ignore_index=True)
    result = fitwise.fit_group(combined, random_state=0)
    assert len(result.curve_results) == 3


def test_fit_group_single_df_without_group_raises():
    df = pd.DataFrame({"x": [1, 2], "y": [1, 2]})
    with pytest.raises(ValueError, match="group"):
        fitwise.fit_group(df)


# --- THE CENTRAL TEST: confidence_gain > 1 when aggregating multiple curves ---


def test_confidence_gain_increases_with_group(rng):
    """Aggregating multiple curves gives stronger evidence than any single curve."""
    dfs = [linear_data(rng, noise=0.3) for _ in range(8)]
    group_result = fitwise.fit_group(dfs, models=["linear", "quadratic", "cubic"], random_state=0)

    assert group_result.confidence_gain > 1.0, (
        f"Expected confidence_gain > 1.0, got {group_result.confidence_gain:.3f}. "
        "Group aggregation must produce stronger evidence than any single curve."
    )


def test_confidence_gain_increases_with_more_curves(rng):
    """More curves in the group → stronger confidence gain."""
    small_dfs = [linear_data(rng, noise=0.5) for _ in range(2)]
    large_dfs = [linear_data(rng, noise=0.5) for _ in range(10)]
    small = fitwise.fit_group(small_dfs, models=["linear", "quadratic"], random_state=0)
    large = fitwise.fit_group(large_dfs, models=["linear", "quadratic"], random_state=0)
    assert large.confidence_gain >= small.confidence_gain


# --- fit_group: summary ---


def test_summary_contains_expected_keys(rng):
    dfs = [linear_data(rng) for _ in range(3)]
    result = fitwise.fit_group(dfs, models=["linear", "quadratic"], random_state=0)
    s = result.summary()
    assert "n_curves" in s
    assert "n_successful_curves" in s
    assert "best_model" in s
    assert "confidence_gain" in s
    assert "model_success_rate" in s
    assert s["n_curves"] == 3


def test_summary_success_rate_in_range(rng):
    dfs = [linear_data(rng) for _ in range(4)]
    result = fitwise.fit_group(dfs, models=["linear", "quadratic"], random_state=0)
    for rate in result.summary()["model_success_rate"].values():
        assert 0.0 <= rate <= 1.0
