from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

# Keep generated inventory and its inputs inside the organized data folders.
BASE_PATH = Path(__file__).resolve().parents[1]
PROCESSED_PATH = BASE_PATH / "data" / "processed"

ROSTER_VALUES_PATH = PROCESSED_PATH / "current_roster_player_values_efficiency_load_role_scaled_base_regularized.parquet"
PICK_VALUES_PATH = PROCESSED_PATH / "anchored_efficiency_load_role_scaled_base_regularized_pick_values.parquet"
MODEL_BUNDLE_PATH = PROCESSED_PATH / "draft_weight_model_bundle_efficiency_load_role_scaled_base_regularized.json"
PROJECTION_CONFIG_PATH = BASE_PATH / "data" / "reference" / "own_pick_tier_projections.csv"
OUTPUT_PATH = PROCESSED_PATH / "pick_Inventory.csv"

TEAM_COLUMN = "END_TEAM_FULL_NAME"
DRAFT_YEARS = (2027, 2028, 2029)
ROUNDS = (1, 2)

# 2027 is the currently projected draft and is not decayed.
# Reduce confidence for picks further into the future.
TIER_CONFIDENCE_BY_DRAFT_YEAR = {2027: 1.00, 2028: 0.85, 2029: 0.65}

NEXT_LOWER_TIER = {
    "projected_top_5_first": "projected_lottery_first",
    "projected_lottery_first": "projected_late_first",
    "projected_late_first": "projected_early_second",
    "projected_early_second": "projected_late_second",
    "projected_late_second": "projected_late_second",
}

DEFAULT_TIER_BY_ROUND = {1: "projected_late_first", 2: "projected_early_second"}

TEAM_ABBREVIATIONS = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def require_columns(data: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")


def team_abbreviation(team_name: str) -> str:
    if team_name in TEAM_ABBREVIATIONS:
        return TEAM_ABBREVIATIONS[team_name]

    tokens = re.findall(r"[A-Za-z0-9]+", str(team_name))
    if not tokens:
        raise ValueError(f"Could not create abbreviation for team: {team_name}")

    if len(tokens) >= 3:
        return "".join(token[0] for token in tokens[:3]).upper()
    return "".join(tokens).upper()[:3]


def build_default_projection_config(teams: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for team in teams:
        for draft_year in DRAFT_YEARS:
            for round_number in ROUNDS:
                records.append(
                    {"owning_team": team, "draft_year": draft_year, "round": round_number, "tier": DEFAULT_TIER_BY_ROUND[round_number]}
                )
    return pd.DataFrame(records)


def load_or_create_projection_config(teams: list[str]) -> pd.DataFrame:
    expected = build_default_projection_config(teams)

    if not PROJECTION_CONFIG_PATH.exists():
        expected.to_csv(PROJECTION_CONFIG_PATH, index=False)
        print("Created projection template with default tiers: " f"{PROJECTION_CONFIG_PATH}")
        print(
            "The projection configuration requires top-five, lottery, late-first, "
            "early-second, or late-second assignments."
        )
        return expected

    projection_config = pd.read_csv(PROJECTION_CONFIG_PATH)
    require_columns(projection_config, ["owning_team", "draft_year", "round", "tier"], "Projection config")

    projection_config["draft_year"] = pd.to_numeric(projection_config["draft_year"], errors="coerce")
    projection_config["round"] = pd.to_numeric(projection_config["round"], errors="coerce")

    if projection_config[["draft_year", "round"]].isna().any().any():
        raise ValueError("Projection config contains nonnumeric draft_year or round values.")

    projection_config["draft_year"] = projection_config["draft_year"].astype(int)
    projection_config["round"] = projection_config["round"].astype(int)

    duplicate_keys = projection_config.duplicated(["owning_team", "draft_year", "round"], keep=False)
    if duplicate_keys.any():
        raise ValueError(
            "Projection config contains duplicate team/year/round rows:\n"
            + projection_config.loc[duplicate_keys, ["owning_team", "draft_year", "round", "tier"]].to_string(index=False)
        )

    expected_keys = set(map(tuple, expected[["owning_team", "draft_year", "round"]].to_numpy()))
    actual_keys = set(map(tuple, projection_config[["owning_team", "draft_year", "round"]].to_numpy()))

    missing_keys = sorted(expected_keys - actual_keys)
    extra_keys = sorted(actual_keys - expected_keys)
    if missing_keys or extra_keys:
        raise ValueError(
            "Projection config does not match the roster teams and required "
            f"2027-2029 rounds. Missing: {missing_keys[:10]}; "
            f"extra: {extra_keys[:10]}"
        )

    return projection_config.sort_values(["owning_team", "draft_year", "round"]).reset_index(drop=True)


def main() -> None:
    # Convert team projections into app-ready pick values and labels.
    if not ROSTER_VALUES_PATH.exists():
        raise FileNotFoundError(f"Current-roster player-value dependency not found: {ROSTER_VALUES_PATH}")
    if not PICK_VALUES_PATH.exists():
        raise FileNotFoundError("Base pick-value table not found. Missing: " f"{PICK_VALUES_PATH}")
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError("Regularized model bundle not found. Missing: " f"{MODEL_BUNDLE_PATH}")

    with MODEL_BUNDLE_PATH.open("r", encoding="utf-8") as file:
        model_bundle = json.load(file)

    roster = pd.read_parquet(ROSTER_VALUES_PATH)
    require_columns(roster, [TEAM_COLUMN], "Roster values")
    teams = sorted(roster[TEAM_COLUMN].dropna().astype(str).unique().tolist())
    if not teams:
        raise ValueError("Roster values contain no team names.")

    pick_values = pd.read_parquet(PICK_VALUES_PATH)
    require_columns(pick_values, ["tier", "estimated_value"], "Pick-value table")
    pick_values["estimated_value"] = pd.to_numeric(pick_values["estimated_value"], errors="coerce")
    if pick_values["estimated_value"].isna().any():
        raise ValueError("Pick-value table contains invalid estimated values.")

    tier_values = pick_values.drop_duplicates("tier").set_index("tier")["estimated_value"].astype(float).to_dict()

    required_tiers = set(NEXT_LOWER_TIER)
    missing_tiers = sorted(required_tiers - set(tier_values))
    if missing_tiers:
        raise KeyError(f"Pick-value table is missing required tiers: {missing_tiers}")

    projection_config = load_or_create_projection_config(teams)

    unsupported_tiers = sorted(set(projection_config["tier"].dropna()) - required_tiers)
    if unsupported_tiers:
        raise ValueError(f"Projection config contains unsupported tiers: {unsupported_tiers}")

    first_round_tiers = {"projected_top_5_first", "projected_lottery_first", "projected_late_first"}
    second_round_tiers = {"projected_early_second", "projected_late_second"}

    bad_first = projection_config.loc[projection_config["round"].eq(1) & ~projection_config["tier"].isin(first_round_tiers)]
    bad_second = projection_config.loc[projection_config["round"].eq(2) & ~projection_config["tier"].isin(second_round_tiers)]
    if not bad_first.empty or not bad_second.empty:
        raise ValueError("First-round rows must use first-round tiers and second-round " "rows must use second-round tiers.")

    records: list[dict[str, object]] = []
    for row in projection_config.itertuples(index=False):
        team = str(row.owning_team)
        draft_year = int(row.draft_year)
        round_number = int(row.round)
        tier = str(row.tier)

        confidence = float(TIER_CONFIDENCE_BY_DRAFT_YEAR[draft_year])
        floor_tier = NEXT_LOWER_TIER[tier]
        base_value = float(tier_values[tier])
        floor_value = float(tier_values[floor_tier])
        adjusted_value = confidence * base_value + (1.0 - confidence) * floor_value
        adjusted_value = max(adjusted_value, floor_value)

        abbreviation = team_abbreviation(team)
        round_label = "R1" if round_number == 1 else "R2"
        pick_id = f"{abbreviation}_{draft_year}_{round_label}_OWN"

        records.append(
            {
                "pick_id": pick_id,
                "pick_label": (f"{draft_year} own Round {round_number} | " f"{tier.replace('projected_', '').replace('_', ' ')}"),
                "owning_team": team,
                "origin_team": team,
                "draft_year": draft_year,
                "round": round_number,
                "tier": tier,
                "base_value": base_value,
                "floor_tier": floor_tier,
                "floor_value": floor_value,
                "tier_confidence": confidence,
                "tier_decay": 1.0 - confidence,
                "adjusted_value": float(adjusted_value),
                "currently_owned": True,
                "available_for_trade": True,
                "inventory_mode": "hypothetical_own_picks",
                "ownership_disclaimer": (
                    "Hypothetical own-pick inventory; actual ownership, " "protections, swaps, and Stepien restrictions are not enforced."
                ),
            }
        )

    inventory = pd.DataFrame(records).sort_values(["owning_team", "draft_year", "round"]).reset_index(drop=True)

    if not inventory["pick_id"].is_unique:
        duplicates = inventory.loc[inventory["pick_id"].duplicated(keep=False), "pick_id"].tolist()
        raise ValueError(f"Generated duplicate pick IDs: {duplicates}")

    expected_rows = len(teams) * len(DRAFT_YEARS) * len(ROUNDS)
    if len(inventory) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} picks; generated {len(inventory)}.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Teams: {len(teams):,}")
    print(f"Picks: {len(inventory):,}")
    print("Confidence schedule:")
    for year, confidence in TIER_CONFIDENCE_BY_DRAFT_YEAR.items():
        print(f"  {year}: {confidence:.0%} confidence, " f"{1.0 - confidence:.0%} decay")
    print("\nTier counts:")
    print(inventory["tier"].value_counts().sort_index())


if __name__ == "__main__":
    main()
