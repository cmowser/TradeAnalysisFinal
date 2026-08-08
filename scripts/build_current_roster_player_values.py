from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

# Resolve every input and output from the organized project root.
BASE_PATH = Path(__file__).resolve().parents[1]
PROCESSED_PATH = BASE_PATH / "data" / "processed"

ROSTER_SOURCE_PATH = BASE_PATH / "data" / "interim" / "season_Totals_Enriched.csv"
MODEL_BUNDLE_PATH = PROCESSED_PATH / "draft_weight_model_bundle_efficiency_load_role_scaled_base_regularized.json"
OUTPUT_PATH = PROCESSED_PATH / "current_roster_player_values_efficiency_load_role_scaled_base_regularized.parquet"

TEAM_COLUMN = "END_TEAM_FULL_NAME"
PLAYER_NAME_COLUMN = "PLAYER_NAME"
PLAYER_ID_COLUMN = "PLAYER_ID"
VALUE_COLUMN = "PLAYER_PRODUCTION_VALUE"
ELIGIBLE_COLUMN = "PRODUCTION_ELIGIBLE"
REASON_COLUMN = "PRODUCTION_INELIGIBILITY_REASON"
ROLE_ADJUSTED_BASE_COLUMN = "ROLE_ADJUSTED_BASE_VALUE"


# Map model metrics to the current-roster columns derived below.
DERIVED_CURRENT_COLUMNS = {
    "season_true_shooting_attempts_per_100": ("TRUE_SHOOTING_ATTEMPTS_PER_100"),
    "season_efficiency_points_added_per_100": ("EFFICIENCY_POINTS_ADDED_PER_100"),
    "season_turnovers_per_100": "TURNOVERS_PER_100",
}


QUALITY_OUTPUT_COLUMNS = {
    "raw": "PLAYER_RAW_QUALITY_SCORE",
    "reliability": "PLAYER_RELIABILITY",
    "shrunk": "PLAYER_RELIABILITY_SHRUNK_QUALITY_SCORE",
    "role_capacity": "PLAYER_ROLE_CAPACITY",
    "role_adjusted": "PLAYER_ROLE_ADJUSTED_QUALITY_SCORE",
    "final": "PLAYER_QUALITY_SCORE",
}


def require_columns(data: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")


def as_numeric(data: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")


def normalize_true_shooting_percentage(series: pd.Series) -> tuple[pd.Series, bool]:
    """Normalize TS% to a decimal scale while rejecting mixed/invalid scales."""
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return values, False

    q95 = float(finite.quantile(0.95))
    converted = False
    if q95 > 2.0:
        if float(finite.max()) > 100.0:
            raise ValueError("TRUE_SHOOTING_PERCENTAGE contains values above 100 and " "cannot be normalized safely.")
        values = values / 100.0
        converted = True

    invalid = values.notna() & ~values.between(0.0, 1.5)
    if invalid.any():
        raise ValueError("TRUE_SHOOTING_PERCENTAGE contains values outside the expected " "decimal range after normalization.")
    return values, converted


def main() -> None:
    # Apply the saved historical model parameters to the current roster.
    if not ROSTER_SOURCE_PATH.exists():
        raise FileNotFoundError(f"Current roster source not found: {ROSTER_SOURCE_PATH}")
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Regularized model bundle not found: {MODEL_BUNDLE_PATH}")

    roster = pd.read_csv(ROSTER_SOURCE_PATH)
    with MODEL_BUNDLE_PATH.open("r", encoding="utf-8") as file:
        model = json.load(file)


    require_columns(roster, [PLAYER_NAME_COLUMN, TEAM_COLUMN], "Roster data")

    metric_reference = model["player_metric_reference"]
    metric_columns = list(model["player_metric_columns"])
    direct_mapping = dict(model["current_roster_metric_mapping"])
    raw_mapping = dict(model["current_roster_raw_column_mapping"])
    domain_metric_weights = model["player_domain_metric_weights"]
    domain_weights = model["player_domain_weights"]

    minimum_minutes = float(model["minimum_season_minutes"])
    base_player_value = float(model["base_player_value"])
    minimum_player_base = float(model["minimum_player_base"])
    quality_slope = float(model["quality_multiplier_slope"])
    zscore_clip_limit = float(model["zscore_clip_limit"])
    reference_ts = float(model["reference_true_shooting_percentage"])
    reliability_constant = float(model["reliability_shrinkage_constant"])
    role_full_mpg = float(model["role_capacity_full_minutes_per_game"])
    role_exponent = float(model["role_capacity_exponent"])

    if minimum_player_base < 0:
        raise ValueError("minimum_player_base cannot be negative.")
    if minimum_player_base > base_player_value:
        raise ValueError("minimum_player_base cannot exceed base_player_value.")
    if reliability_constant < 0:
        raise ValueError("reliability_shrinkage_constant cannot be negative.")
    if role_full_mpg <= 0:
        raise ValueError("role_capacity_full_minutes_per_game must be positive.")
    if role_exponent <= 0:
        raise ValueError("role_capacity_exponent must be positive.")
    if zscore_clip_limit <= 0:
        raise ValueError("zscore_clip_limit must be positive.")

    required_raw_columns = list(dict.fromkeys(raw_mapping.values()))
    required_direct_columns = list(dict.fromkeys(direct_mapping.values()))
    require_columns(roster, required_raw_columns + required_direct_columns, "Roster data")

    as_numeric(roster, sorted(set(required_raw_columns + required_direct_columns)))

    ts_column = raw_mapping["season_true_shooting_percentage"]
    roster[ts_column], ts_was_converted = normalize_true_shooting_percentage(roster[ts_column])

    games = roster[raw_mapping["season_games_played"]]
    minutes = roster[raw_mapping["season_minutes"]]
    possessions = roster[raw_mapping["season_estimated_player_possessions"]]
    fga = roster[raw_mapping["season_field_goals_attempted"]]
    fta = roster[raw_mapping["season_free_throws_attempted"]]
    turnovers = roster[raw_mapping["season_turnovers"]]
    true_shooting_percentage = roster[ts_column]

    true_shooting_attempts = fga + 0.44 * fta
    true_shooting_attempts_per_100 = np.where(
        possessions.gt(0) & true_shooting_attempts.notna(), 100.0 * true_shooting_attempts / possessions, np.nan
    )
    efficiency_points_added_per_100 = 2.0 * true_shooting_attempts_per_100 * (true_shooting_percentage - reference_ts)
    turnovers_per_100 = np.where(possessions.gt(0) & turnovers.notna(), 100.0 * turnovers / possessions, np.nan)

    roster[DERIVED_CURRENT_COLUMNS["season_true_shooting_attempts_per_100"]] = true_shooting_attempts_per_100
    roster[DERIVED_CURRENT_COLUMNS["season_efficiency_points_added_per_100"]] = efficiency_points_added_per_100
    roster[DERIVED_CURRENT_COLUMNS["season_turnovers_per_100"]] = turnovers_per_100

    model_to_current_column = {**direct_mapping, **DERIVED_CURRENT_COLUMNS}
    missing_metric_mappings = [metric for metric in metric_columns if metric not in model_to_current_column]
    if missing_metric_mappings:
        raise KeyError("No current-roster mapping is defined for model metrics: " f"{missing_metric_mappings}")

    current_metric_columns = [model_to_current_column[metric] for metric in metric_columns]
    complete_metrics = roster[current_metric_columns].notna().all(axis=1)
    eligible = games.gt(0) & minutes.ge(minimum_minutes) & possessions.gt(0) & complete_metrics

    zscore_columns: dict[str, str] = {}
    for model_metric in metric_columns:
        current_column = model_to_current_column[model_metric]
        reference = metric_reference[model_metric]
        mean = float(reference["mean"])
        std = float(reference["std_ddof_0"])
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid reference standard deviation for " f"{model_metric}: {std}")

        z_column = f"{model_metric}_zscore"
        roster[z_column] = ((roster[current_column] - mean) / std).clip(lower=-zscore_clip_limit, upper=zscore_clip_limit)
        roster.loc[~eligible, z_column] = np.nan
        zscore_columns[model_metric] = z_column

    domain_columns: list[str] = []
    for domain_name, metric_weights in domain_metric_weights.items():
        domain_column = f"{domain_name}_quality_score"
        domain_score = pd.Series(0.0, index=roster.index, dtype="float64")
        for z_column, signed_weight in metric_weights.items():
            if z_column not in roster.columns:
                raise KeyError(f"Domain {domain_name} references missing z-score " f"column: {z_column}")
            domain_score = domain_score + float(signed_weight) * roster[z_column]
        roster[domain_column] = domain_score
        roster.loc[~eligible, domain_column] = np.nan
        domain_columns.append(domain_column)

    raw_quality = pd.Series(0.0, index=roster.index, dtype="float64")
    for domain_name, domain_weight in domain_weights.items():
        raw_quality = raw_quality + float(domain_weight) * roster[f"{domain_name}_quality_score"]
    raw_quality.loc[~eligible] = np.nan

    reliability = np.where(possessions.gt(0), possessions / (possessions + reliability_constant), np.nan)
    minutes_per_game = np.where(games.gt(0), minutes / games, np.nan)
    role_capacity = np.power(np.clip(minutes_per_game / role_full_mpg, 0.0, 1.0), role_exponent)

    roster["MINUTES_PER_GAME"] = minutes_per_game
    roster[QUALITY_OUTPUT_COLUMNS["raw"]] = raw_quality
    roster[QUALITY_OUTPUT_COLUMNS["reliability"]] = reliability
    roster[QUALITY_OUTPUT_COLUMNS["shrunk"]] = roster[QUALITY_OUTPUT_COLUMNS["raw"]] * roster[QUALITY_OUTPUT_COLUMNS["reliability"]]
    roster[QUALITY_OUTPUT_COLUMNS["role_capacity"]] = role_capacity
    roster[QUALITY_OUTPUT_COLUMNS["role_adjusted"]] = (
        roster[QUALITY_OUTPUT_COLUMNS["shrunk"]] * roster[QUALITY_OUTPUT_COLUMNS["role_capacity"]]
    )
    roster[QUALITY_OUTPUT_COLUMNS["final"]] = roster[QUALITY_OUTPUT_COLUMNS["role_adjusted"]]

    for column in QUALITY_OUTPUT_COLUMNS.values():
        roster.loc[~eligible, column] = np.nan

    roster[ROLE_ADJUSTED_BASE_COLUMN] = np.nan
    roster.loc[eligible, ROLE_ADJUSTED_BASE_COLUMN] = (
        minimum_player_base + (base_player_value - minimum_player_base) * roster.loc[eligible, QUALITY_OUTPUT_COLUMNS["role_capacity"]]
    )

    roster[VALUE_COLUMN] = np.nan
    roster.loc[eligible, VALUE_COLUMN] = np.clip(
        roster.loc[eligible, ROLE_ADJUSTED_BASE_COLUMN] + quality_slope * roster.loc[eligible, QUALITY_OUTPUT_COLUMNS["final"]], 0.0, None
    )
    roster[ELIGIBLE_COLUMN] = eligible & roster[VALUE_COLUMN].notna()

    reasons: list[str] = []
    for row_index in roster.index:
        row_reasons: list[str] = []
        if pd.isna(games.at[row_index]):
            row_reasons.append(f"{raw_mapping['season_games_played']} missing")
        elif games.at[row_index] <= 0:
            row_reasons.append(f"{raw_mapping['season_games_played']} <= 0")
        if pd.isna(minutes.at[row_index]):
            row_reasons.append(f"{raw_mapping['season_minutes']} missing")
        elif minutes.at[row_index] < minimum_minutes:
            row_reasons.append(f"{raw_mapping['season_minutes']} below " f"{minimum_minutes:g}")
        if pd.isna(possessions.at[row_index]):
            row_reasons.append(f"{raw_mapping['season_estimated_player_possessions']} " "missing")
        elif possessions.at[row_index] <= 0:
            row_reasons.append(f"{raw_mapping['season_estimated_player_possessions']} <= 0")
        for column in current_metric_columns:
            if pd.isna(roster.at[row_index, column]):
                row_reasons.append(f"{column} missing")
        reasons.append(", ".join(dict.fromkeys(row_reasons)))

    roster[REASON_COLUMN] = reasons

    duplicate_names = (
        roster.loc[roster[PLAYER_NAME_COLUMN].astype(str).duplicated(keep=False), PLAYER_NAME_COLUMN].astype(str).unique().tolist()
    )
    if duplicate_names:
        raise ValueError("The app uses player names as selectors, so names must be " f"unique. Duplicate names: {sorted(duplicate_names)}")

    output_columns = [
        *([PLAYER_ID_COLUMN] if PLAYER_ID_COLUMN in roster.columns else []),
        PLAYER_NAME_COLUMN,
        TEAM_COLUMN,
        raw_mapping["season_games_played"],
        raw_mapping["season_minutes"],
        "MINUTES_PER_GAME",
        raw_mapping["season_estimated_player_possessions"],
        raw_mapping["season_field_goals_attempted"],
        raw_mapping["season_free_throws_attempted"],
        raw_mapping["season_turnovers"],
        raw_mapping["season_true_shooting_percentage"],
        *current_metric_columns,
        *zscore_columns.values(),
        *domain_columns,
        *QUALITY_OUTPUT_COLUMNS.values(),
        ROLE_ADJUSTED_BASE_COLUMN,
        VALUE_COLUMN,
        ELIGIBLE_COLUMN,
        REASON_COLUMN,
    ]
    output_columns = list(dict.fromkeys(output_columns))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    roster[output_columns].to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Players: {len(roster):,}")
    print(f"Eligible: {int(roster[ELIGIBLE_COLUMN].sum()):,}")
    print(f"Production-ineligible: {int((~roster[ELIGIBLE_COLUMN]).sum()):,}")
    print("True shooting input converted from percentage points: " f"{ts_was_converted}")
    print(f"Reliability shrinkage constant: {reliability_constant:g}")
    print("Role-scaled base: minimum " f"{minimum_player_base:g}, full base {base_player_value:g}")
    print("Role capacity: full at " f"{role_full_mpg:g} MPG, exponent {role_exponent:g}")
    print(roster.loc[roster[ELIGIBLE_COLUMN], VALUE_COLUMN].describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))
    print("\nTop 25 current player values:")
    print(
        roster.loc[
            roster[ELIGIBLE_COLUMN],
            [
                PLAYER_NAME_COLUMN,
                TEAM_COLUMN,
                "MINUTES_PER_GAME",
                QUALITY_OUTPUT_COLUMNS["reliability"],
                QUALITY_OUTPUT_COLUMNS["role_capacity"],
                ROLE_ADJUSTED_BASE_COLUMN,
                VALUE_COLUMN,
            ],
        ]
        .sort_values(VALUE_COLUMN, ascending=False)
        .head(25)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
