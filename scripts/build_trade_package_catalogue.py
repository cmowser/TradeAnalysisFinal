"""Build a precomputed per-team hybrid trade-package catalogue.

The catalogue contains every combination of:
- zero, one, or two eligible players from one team; and
- zero through five currently owned, trade-available picks from that team.

It is intentionally a proof-of-concept build artifact. The app does not perform
fingerprint or freshness validation against the source roster and pick files.
The runtime search evaluates every valid catalogue completion; it does not use
a nearest-value shortlist or approximate pick completion. Catalogue freshness
depends on the current player-value and pick-inventory artifacts.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import json

import numpy as np
import pandas as pd

# Read and write all catalogue artifacts relative to the project root.
BASE_PATH = Path(__file__).resolve().parents[1]
PROCESSED_PATH = BASE_PATH / "data" / "processed"

ROSTER_VALUES_PATH = PROCESSED_PATH / "current_roster_player_values_efficiency_load_role_scaled_base_regularized.parquet"
PICK_INVENTORY_PATH = PROCESSED_PATH / "pick_Inventory.csv"
MODEL_BUNDLE_PATH = PROCESSED_PATH / "draft_weight_model_bundle_efficiency_load_role_scaled_base_regularized.json"
OUTPUT_PATH = PROCESSED_PATH / "team_hybrid_package_catalogue.parquet"

TEAM_COLUMN = "END_TEAM_FULL_NAME"
PLAYER_NAME_COLUMN = "PLAYER_NAME"
PLAYER_VALUE_COLUMN = "PLAYER_PRODUCTION_VALUE"
ELIGIBLE_COLUMN = "PRODUCTION_ELIGIBLE"

# Bound the exhaustive search to combinations the app can evaluate quickly.
MAX_PLAYERS_PER_PACKAGE = 2
MAX_PICKS_PER_PACKAGE = 5


def _as_bool(series: pd.Series) -> pd.Series:
    """Normalize ordinary CSV boolean representations."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _slot_adjusted_value(values: tuple[float, ...], slot_cost: float) -> float:
    if not values:
        return 0.0
    ordered = np.sort(np.asarray(values, dtype="float64"))[::-1]
    return float(ordered[0] + np.clip(ordered[1:] - float(slot_cost), 0.0, None).sum())


def _build_player_packages(team_rows: pd.DataFrame, slot_cost: float) -> list[dict]:
    players = [
        (str(row[PLAYER_NAME_COLUMN]), float(row[PLAYER_VALUE_COLUMN])) for _, row in team_rows.sort_values(PLAYER_NAME_COLUMN).iterrows()
    ]

    records: list[dict] = []
    package_index = 0
    for count in range(0, min(MAX_PLAYERS_PER_PACKAGE, len(players)) + 1):
        for package in combinations(players, count):
            names = tuple(item[0] for item in package)
            values = tuple(float(item[1]) for item in package)
            records.append(
                {
                    "player_package_id": f"P{package_index:04d}",
                    "player_names_json": json.dumps(list(names)),
                    "player_values_json": json.dumps(list(values)),
                    "player_count": int(count),
                    "player_raw_value": float(sum(values)),
                    "player_standalone_slot_adjusted_value": (_slot_adjusted_value(values, slot_cost)),
                    "player_1_name": names[0] if len(names) >= 1 else None,
                    "player_1_value": values[0] if len(values) >= 1 else np.nan,
                    "player_2_name": names[1] if len(names) >= 2 else None,
                    "player_2_value": values[1] if len(values) >= 2 else np.nan,
                }
            )
            package_index += 1
    return records


def _build_pick_packages(team_rows: pd.DataFrame) -> list[dict]:
    picks = team_rows.sort_values([column for column in ["draft_year", "round", "pick_id"] if column in team_rows]).to_dict(
        orient="records"
    )

    records: list[dict] = []
    package_index = 0
    for count in range(0, min(MAX_PICKS_PER_PACKAGE, len(picks)) + 1):
        for package in combinations(picks, count):
            pick_ids = tuple(str(item["pick_id"]) for item in package)
            pick_values = tuple(float(item["adjusted_value"]) for item in package)
            records.append(
                {
                    "pick_package_id": f"D{package_index:04d}",
                    "pick_ids_json": json.dumps(list(pick_ids)),
                    "pick_count": int(count),
                    "pick_value": float(sum(pick_values)),
                    **{
                        f"pick_{position}_id": (pick_ids[position - 1] if len(pick_ids) >= position else None)
                        for position in range(1, MAX_PICKS_PER_PACKAGE + 1)
                    },
                }
            )
            package_index += 1
    return records


def main() -> None:
    # Precompute every allowed player-and-pick package for each team.
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    roster = pd.read_parquet(ROSTER_VALUES_PATH)
    pick_inventory = pd.read_csv(PICK_INVENTORY_PATH)
    with MODEL_BUNDLE_PATH.open("r", encoding="utf-8") as file:
        model_bundle = json.load(file)

    required_roster = {TEAM_COLUMN, PLAYER_NAME_COLUMN, PLAYER_VALUE_COLUMN, ELIGIBLE_COLUMN}
    missing_roster = sorted(required_roster - set(roster.columns))
    if missing_roster:
        raise KeyError(f"Roster player-value file is missing: {missing_roster}")

    required_picks = {"pick_id", "owning_team", "adjusted_value", "currently_owned", "available_for_trade"}
    missing_picks = sorted(required_picks - set(pick_inventory.columns))
    if missing_picks:
        raise KeyError(f"Pick inventory is missing: {missing_picks}")

    slot_cost = float(model_bundle["additional_player_slot_cost"])

    roster = roster.copy()
    roster[PLAYER_VALUE_COLUMN] = pd.to_numeric(roster[PLAYER_VALUE_COLUMN], errors="coerce")
    roster = roster.loc[roster[ELIGIBLE_COLUMN].fillna(False).astype(bool) & roster[PLAYER_VALUE_COLUMN].gt(0)].copy()

    pick_inventory = pick_inventory.copy()
    pick_inventory["adjusted_value"] = pd.to_numeric(pick_inventory["adjusted_value"], errors="coerce")
    pick_inventory = pick_inventory.loc[
        _as_bool(pick_inventory["currently_owned"])
        & _as_bool(pick_inventory["available_for_trade"])
        & pick_inventory["adjusted_value"].notna()
        & pick_inventory["adjusted_value"].ge(0)
    ].copy()

    teams = sorted(set(roster[TEAM_COLUMN].dropna().astype(str)) | set(pick_inventory["owning_team"].dropna().astype(str)))

    all_records: list[dict] = []
    summary_records: list[dict] = []

    for team in teams:
        team_roster = roster.loc[roster[TEAM_COLUMN].astype(str).eq(team), [PLAYER_NAME_COLUMN, PLAYER_VALUE_COLUMN]].copy()
        team_picks = pick_inventory.loc[pick_inventory["owning_team"].astype(str).eq(team)].copy()

        player_packages = _build_player_packages(team_roster, slot_cost)
        pick_packages = _build_pick_packages(team_picks)

        team_row_count = 0
        for player_package in player_packages:
            for pick_package in pick_packages:
                total_assets = int(player_package["player_count"]) + int(pick_package["pick_count"])
                all_records.append(
                    {
                        "team": team,
                        "hybrid_package_id": (f"{team}|{player_package['player_package_id']}|" f"{pick_package['pick_package_id']}"),
                        **player_package,
                        **pick_package,
                        "total_assets": total_assets,
                        "standalone_total_value": float(
                            player_package["player_standalone_slot_adjusted_value"] + pick_package["pick_value"]
                        ),
                    }
                )
                team_row_count += 1

        summary_records.append(
            {
                "team": team,
                "eligible_players": len(team_roster),
                "available_picks": len(team_picks),
                "player_packages": len(player_packages),
                "pick_packages": len(pick_packages),
                "hybrid_packages": team_row_count,
            }
        )

    catalogue = pd.DataFrame(all_records)
    if catalogue.empty:
        raise ValueError("No hybrid packages were generated.")

    catalogue.to_parquet(OUTPUT_PATH, index=False)

    summary = pd.DataFrame(summary_records)
    print(summary.to_string(index=False))
    print()
    print(f"Wrote {len(catalogue):,} hybrid package rows to:")
    print(OUTPUT_PATH)
    print(f"Additional-player slot cost: {slot_cost:.6f}")


if __name__ == "__main__":
    main()
