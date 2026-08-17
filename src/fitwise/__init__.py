import pandas as pd

from fitwise.aggregation import build_group_result
from fitwise.fitting import fit_curve_from_arrays
from fitwise.models import instantiate_models
from fitwise.results import CurveResult, GroupResult


def fit_curve(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    models: list[str] | None = None,
    random_state: int | None = None,
) -> CurveResult:
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    candidates = instantiate_models(models)
    return fit_curve_from_arrays(x, y, candidates, random_state)


def fit_group(
    dfs: list[pd.DataFrame] | pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    models: list[str] | None = None,
    random_state: int | None = None,
) -> GroupResult:
    frames = _normalize_group_input(dfs)
    model_names = list(instantiate_models(models).keys())
    curve_results = [
        fit_curve_from_arrays(
            df[x_col].to_numpy(dtype=float),
            df[y_col].to_numpy(dtype=float),
            instantiate_models(model_names),
            random_state,
        )
        for df in frames
    ]
    return build_group_result(curve_results)


def _normalize_group_input(
    dfs: list[pd.DataFrame] | pd.DataFrame,
) -> list[pd.DataFrame]:
    if isinstance(dfs, pd.DataFrame):
        if "group" not in dfs.columns:
            raise ValueError(
                "When passing a single DataFrame, it must have a 'group' column. "
                "Alternatively, pass a list of DataFrames."
            )
        return [g.drop(columns="group").reset_index(drop=True) for _, g in dfs.groupby("group")]
    return list(dfs)
