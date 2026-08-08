"""Assign point-in-time draft-capital tiers from team records."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from draft_asset_parser import extract_transaction_draft_assets

# Historical transaction and record artifacts live in the interim layer.
BASE_PATH = Path(__file__).resolve().parents[1]
INTERIM_PATH = BASE_PATH / "data" / "interim"


def _norm_team(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    aliases = {
        "sixers": "76ers",
        "philadelphia76ers": "76ers",
        "blazers": "trailblazers",
        "portlandtrailblazers": "trailblazers",
        "sonics": "supersonics",
        "seattlesupersonics": "supersonics",
        "newjerseynets": "nets",
        "brooklynnets": "nets",
        "washingtonbullets": "bullets",
    }
    return aliases.get(text, text)


def _previous_season_from_date(date: pd.Timestamp) -> str:
    return f"{date.year - 1}-{str(date.year)[-2:]}"


def _record_season(row: pd.Series) -> str:
    # Offseason picks use the prior completed season as their record signal.
    if str(row["transaction_season"]).lower() == "offseason":
        return _previous_season_from_date(pd.Timestamp(row["transaction_date"]))
    return str(row["transaction_season"])


def _build_team_name_lookup(records: pd.DataFrame) -> dict[tuple[str, str], int]:
    lookup: dict[tuple[str, str], int] = {}
    for row in records.itertuples(index=False):
        lookup[(str(row.Season), _norm_team(row.teamName))] = int(row.teamID)
    global_names = (
        records.assign(_name=records["teamName"].map(_norm_team))
        .groupby("_name")["teamID"]
        .agg(lambda x: list(pd.unique(x.dropna().astype(int))))
    )
    for name, ids in global_names.items():
        if len(ids) == 1:
            lookup[("*", name)] = ids[0]
    return lookup


def _explicit_source_team_id(source_text: Any, season: str, lookup: dict[tuple[str, str], int]) -> int | None:
    if pd.isna(source_text):
        return None
    source_string = str(source_text).strip()
    simple_counterparty = re.fullmatch(r"trade with ([A-Za-z0-9 .'-]+)", source_string, re.IGNORECASE)
    if simple_counterparty:
        source_string = simple_counterparty.group(1)
    normalized = _norm_team(source_string)
    false_sources = {"earliertrade", "previoustrade", "anearliertrade", "priortrade", "ealiertrade", "117to123in2003"}
    if normalized in false_sources:
        return None
    direct = lookup.get((season, normalized))
    if direct is not None:
        return direct
    direct = lookup.get(("*", normalized))
    if direct is not None:
        return direct
    words = re.findall(r"[A-Za-z0-9]+", source_string)
    token_ids = set()
    for width in (1, 2, 3):
        for start in range(0, len(words) - width + 1):
            phrase = _norm_team(" ".join(words[start : start + width]))
            candidate = lookup.get((season, phrase), lookup.get(("*", phrase)))
            if candidate is not None:
                token_ids.add(candidate)
    if len(token_ids) == 1:
        return next(iter(token_ids))
    # Accept an unambiguous team-name phrase, but not compound descriptions.
    candidates = {
        team_id
        for (candidate_season, team_name), team_id in lookup.items()
        if candidate_season == season and (team_name in normalized or normalized in team_name) and len(team_name) >= 4
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def _league_records_as_of(
    games: pd.DataFrame, season_records: pd.DataFrame, season: str, date: pd.Timestamp, offseason: bool
) -> pd.DataFrame:
    teams = season_records.loc[season_records["Season"].eq(season), ["teamID", "Wins", "Losses"]].copy()
    teams["teamID"] = teams["teamID"].astype(int)
    if offseason:
        teams = teams.rename(columns={"Wins": "wins", "Losses": "losses"})
    else:
        prior = games.loc[games["Season"].eq(season) & games["game_Date"].lt(date)]
        home = prior[["hometeamId", "winner"]].rename(columns={"hometeamId": "teamID"})
        away = prior[["awayteamId", "winner"]].rename(columns={"awayteamId": "teamID"})
        appearances = pd.concat([home, away], ignore_index=True).dropna(subset=["teamID"])
        appearances["teamID"] = appearances["teamID"].astype(int)
        appearances["win"] = appearances["teamID"].eq(appearances["winner"]).astype(int)
        current = appearances.groupby("teamID").agg(wins=("win", "sum"), games=("win", "size"))
        current["losses"] = current["games"] - current["wins"]
        teams = (
            teams[["teamID"]]
            .merge(current[["wins", "losses"]], left_on="teamID", right_index=True, how="left")
            .fillna({"wins": 0, "losses": 0})
        )

    teams["games"] = teams["wins"] + teams["losses"]
    teams["win_percentage"] = np.where(teams["games"].gt(0), teams["wins"] / teams["games"], np.nan)
    # Worst record projects to pick 1. Use wins as a deterministic secondary
    # key while retaining the point-in-time win percentage as the main signal.
    ranked = teams.sort_values(["win_percentage", "wins", "teamID"], na_position="last").reset_index(drop=True)
    ranked["projected_draft_position"] = np.arange(1, len(ranked) + 1)
    return ranked


def rank_draft_assets(assets: pd.DataFrame, transactions: pd.DataFrame, games: pd.DataFrame, season_records: pd.DataFrame) -> pd.DataFrame:
    # Assign first- and second-round tiers from point-in-time team strength.
    assets = assets.copy()
    assets["transaction_date"] = pd.to_datetime(assets["transaction_date"])
    games = games.copy()
    games["game_Date"] = pd.to_datetime(games["game_Date"])
    season_records = season_records.copy()
    lookup = _build_team_name_lookup(season_records)

    assets["record_season"] = assets.apply(_record_season, axis=1)
    assets["record_basis"] = np.where(
        assets["transaction_season"].astype(str).str.lower().eq("offseason"),
        "previous_season_final_record",
        "current_record_at_transaction",
    )

    def resolved_transaction_team(row: pd.Series) -> float:
        if pd.notna(row["transaction_team_id"]):
            return float(row["transaction_team_id"])
        team_id = lookup.get((row["record_season"], _norm_team(row["transaction_team"])))
        if team_id is None:
            team_id = lookup.get(("*", _norm_team(row["transaction_team"])))
        return float(team_id) if team_id is not None else np.nan

    assets["resolved_transaction_team_id"] = assets.apply(resolved_transaction_team, axis=1)

    # The relinquishing row identifies the team conveying the asset. Mirror
    # that identity to the acquiring row using date + point-in-time asset text.
    conveyed = assets.loc[
        assets["transaction_side"].eq("Relinquished") & assets["resolved_transaction_team_id"].notna(),
        ["transaction_date", "known_at_transaction_text", "resolved_transaction_team_id", "transaction_team"],
    ].drop_duplicates()
    conveyed_lookup = {key: group.to_dict("records") for key, group in conveyed.groupby(["transaction_date", "known_at_transaction_text"])}

    def source_id(row: pd.Series) -> float:
        explicit = _explicit_source_team_id(row["source_team_text"], row["record_season"], lookup)
        if explicit is None:
            explicit = _explicit_source_team_id(row["known_at_transaction_text"], row["record_season"], lookup)
        if explicit is not None:
            return float(explicit)
        if row["transaction_side"] == "Relinquished" and pd.notna(row["resolved_transaction_team_id"]):
            return float(row["resolved_transaction_team_id"])
        candidates = conveyed_lookup.get((row["transaction_date"], row["known_at_transaction_text"]), [])
        if len(candidates) == 1:
            return float(candidates[0]["resolved_transaction_team_id"])
        notes = _norm_team(row["transaction_notes"])
        note_matches = [candidate for candidate in candidates if _norm_team(candidate["transaction_team"]) in notes]
        ids = {candidate["resolved_transaction_team_id"] for candidate in note_matches}
        if len(ids) == 1:
            return float(next(iter(ids)))
        notes_team = _explicit_source_team_id(row["transaction_notes"], row["record_season"], lookup)
        return float(notes_team) if notes_team is not None else np.nan

    assets["projected_pick_team_id"] = assets.apply(source_id, axis=1)

    record_cache: dict[tuple[str, pd.Timestamp, bool], pd.DataFrame] = {}
    projections = []
    for row in assets.itertuples(index=False):
        offseason = row.record_basis == "previous_season_final_record"
        key = (row.record_season, pd.Timestamp(row.transaction_date), offseason)
        if key not in record_cache:
            record_cache[key] = _league_records_as_of(
                games, season_records, row.record_season, pd.Timestamp(row.transaction_date), offseason
            )
        league = record_cache[key]
        team = league.loc[league["teamID"].eq(row.projected_pick_team_id)]
        if team.empty:
            projections.append((np.nan, np.nan, np.nan, len(league)))
        else:
            item = team.iloc[0]
            projections.append((item.wins, item.losses, item.projected_draft_position, len(league)))

    assets[["record_wins", "record_losses", "projected_draft_position", "league_team_count"]] = pd.DataFrame(
        projections, index=assets.index
    )

    def tier(row: pd.Series) -> str:
        if row["asset_type"] == "draft_rights":
            return "draft_rights_unranked"
        if pd.isna(row["draft_round"]):
            return "unknown_round_unranked"
        draft_round = int(row["draft_round"])
        if draft_round >= 3:
            return "zero_value_later_round"
        if pd.isna(row["projected_draft_position"]):
            return "record_unavailable"
        position = int(row["projected_draft_position"])
        suffix = "_swap" if row["asset_type"] == "pick_swap" else ""
        if draft_round == 1:
            if position <= 5:
                return "projected_top_5_first" + suffix
            if position <= min(14, int(row["league_team_count"])):
                return "projected_lottery_first" + suffix
            return "projected_late_first" + suffix
        midpoint = int(np.ceil(row["league_team_count"] / 2))
        return ("projected_early_second" if position <= midpoint else "projected_late_second") + suffix

    assets["point_in_time_tier"] = assets.apply(tier, axis=1)
    assets["assigned_zero_value"] = assets["point_in_time_tier"].eq("zero_value_later_round")

    # Remove information only known after the transaction. It remains in the
    # parser logic for QA but is deliberately absent from this modeling table.
    future_columns = ["realized_year", "realized_pick_number", "realized_player_name", "not_exercised_or_extinguished"]
    return assets.drop(columns=[c for c in future_columns if c in assets])


def main(output_directory: str | Path = INTERIM_PATH) -> None:
    output_directory = Path(output_directory)
    transactions = pd.read_csv(INTERIM_PATH / "transactions_with_WL.csv")
    # Build in memory so an open review CSV does not block ranked output.
    assets = extract_transaction_draft_assets(transactions).drop(columns=["raw_asset_text"])
    games = pd.read_csv(INTERIM_PATH / "all_regular_season_games.csv")
    records = pd.read_csv(INTERIM_PATH / "team_season_records.csv")
    ranked = rank_draft_assets(assets, transactions, games, records)
    ranked.to_parquet(output_directory / "transaction_draft_assets_ranked.parquet", index=False)
    ranked.to_csv(output_directory / "transaction_draft_assets_ranked.csv", index=False)
    tier_summary = (
        ranked.assign(
            tier_column=lambda d: (d["transaction_side"].str.lower() + "__" + d["point_in_time_tier"]),
            asset_units=lambda d: d["pick_count"].fillna(1).clip(lower=1),
        )
        .pivot_table(index="source_transaction_index", columns="tier_column", values="asset_units", aggfunc="sum", fill_value=0)
        .add_prefix("draft_tier__")
        .reset_index()
    )
    tier_summary.columns.name = None
    tier_summary.to_parquet(output_directory / "transaction_draft_tier_summary.parquet", index=False)
    print(ranked["point_in_time_tier"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
