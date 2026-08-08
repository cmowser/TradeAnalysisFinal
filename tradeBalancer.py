"""
trade_balancer.py

Reusable trade-scoring and recommendation functions for the historical
NBA trade-compensation project.

Current scope
-------------
- Scores player, pick, and mixed-asset proposals from both team perspectives.
- Uses an additive player-value proxy for additional-player recommendations.
- Supports players-first, picks-first, and hybrid best-fit recommendations.
- Supports specific picks in the initial proposal and constrains later pick
  recommendations to remaining owned assets available for trade.
- Values outright picks only. Pick swaps are intentionally unsupported.

Player-value contract
---------------------
Each qualified player has a nonnegative finalized production value built
from shooting load, efficiency points added, playmaking, defensive/rebounding
events, plus-minus impact, reliability shrinkage, demonstrated role capacity,
and a role-scaled baseline that reduces the automatic value assigned to
low-minute players. Multi-player packages apply the slot-adjusted rule exported by notebook
04: the highest-valued player contributes full value, and every additional
player contributes ``max(individual value - slot cost, 0)``.
"""

from __future__ import annotations

from itertools import combinations, product
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "load_model_bundle",
    "build_pick_value_table",
    "create_empty_draft_counts",
    "get_roster_player",
    "get_roster_players",
    "calculate_one_for_one_raw_differential",
    "score_player_production_differential",
    "add_draft_compensation_score",
    "score_one_for_one_trade",
    "score_two_team_one_for_one_trade",
    "score_two_team_multi_player_trade",
    "calculate_slot_adjusted_package_value",
    "calculate_player_match_profile",
    "generate_player_addition_alternatives",
    "rank_player_addition_alternatives",
    "apply_player_addition_phase",
    "validate_pick_inventory",
    "get_available_pick_inventory",
    "resolve_initial_pick_assets",
    "calculate_initial_pick_adjustment",
    "build_initial_draft_counts",
    "recommend_inventory_constrained_pick_package",
    "enumerate_inventory_constrained_pick_packages",
    "apply_pick_top_up",
    "reserve_selected_picks",
    "generate_hybrid_trade_options",
    "generate_hybrid_trade_options_from_catalogue",
    "rank_hybrid_trade_options",
    "recommend_trade_adjustments",
    "balance_trade",
]


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    """Load the JSON bundle exported by the constrained draft-weight notebook."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def build_pick_value_table(
    outright_pick_hierarchy: Sequence[str],
    outright_pick_weights: Sequence[float],
    outright_pick_net_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Create the tier/value lookup used by the recommendation engine."""
    hierarchy = list(outright_pick_hierarchy)
    weights = np.asarray(outright_pick_weights, dtype="float64")

    if len(hierarchy) != len(weights):
        raise ValueError(
            "outright_pick_hierarchy and outright_pick_weights must have "
            "the same length."
        )

    if outright_pick_net_columns is None:
        net_columns = [f"net_{tier}_count" for tier in hierarchy]
    else:
        net_columns = list(outright_pick_net_columns)

    if len(net_columns) != len(hierarchy):
        raise ValueError(
            "outright_pick_net_columns must match the hierarchy length."
        )

    return pd.DataFrame(
        {
            "tier": hierarchy,
            "net_count_column": net_columns,
            "estimated_value": weights,
        }
    )


def create_empty_draft_counts(
    outright_pick_net_columns: Sequence[str],
) -> dict[str, int]:
    """Return zero net counts for every supported outright-pick tier."""
    return {column: 0 for column in outright_pick_net_columns}


def get_roster_player(
    roster_data: pd.DataFrame,
    player_name: str,
    player_name_column: str = "candidate_player",
) -> pd.Series:
    """Return one unique roster row for a player."""
    if player_name_column not in roster_data.columns:
        raise KeyError(f"Roster is missing column: {player_name_column}")

    matches = roster_data.loc[roster_data[player_name_column].eq(player_name)]

    if matches.empty:
        raise ValueError(f"Player not found: {player_name}")

    if len(matches) > 1:
        raise ValueError(f"Multiple roster rows found for: {player_name}")

    return matches.iloc[0]


def _normalize_player_names(
    player_names: str | Sequence[str] | None,
    argument_name: str = "player_names",
    allow_empty: bool = False,
) -> list[str]:
    """Normalize one player name or a sequence into a unique ordered list."""
    if player_names is None:
        normalized = []
    elif isinstance(player_names, str):
        normalized = [player_names]
    else:
        normalized = list(player_names)

    normalized = [str(name) for name in normalized if str(name).strip()]

    if not normalized and not allow_empty:
        raise ValueError(f"{argument_name} must contain at least one player.")

    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{argument_name} contains duplicate player names.")

    return normalized


def get_roster_players(
    roster_data: pd.DataFrame,
    player_names: str | Sequence[str] | None,
    player_name_column: str = "candidate_player",
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Return roster rows for a unique ordered package of players."""
    names = _normalize_player_names(
        player_names,
        allow_empty=allow_empty,
    )

    if player_name_column not in roster_data.columns:
        raise KeyError(f"Roster is missing column: {player_name_column}")

    if not names:
        return roster_data.iloc[0:0].copy().reset_index(drop=True)

    matched = roster_data.loc[
        roster_data[player_name_column].isin(names)
    ].copy()

    found_names = set(matched[player_name_column].astype(str))
    missing_names = [name for name in names if name not in found_names]
    if missing_names:
        raise ValueError(f"Players not found: {missing_names}")

    duplicate_names = matched.loc[
        matched[player_name_column].duplicated(keep=False),
        player_name_column,
    ].astype(str).unique().tolist()
    if duplicate_names:
        raise ValueError(
            "Multiple roster rows found for players: "
            f"{sorted(duplicate_names)}"
        )

    order_map = {name: index for index, name in enumerate(names)}
    matched["__requested_order"] = (
        matched[player_name_column].astype(str).map(order_map)
    )

    return (
        matched.sort_values("__requested_order")
        .drop(columns="__requested_order")
        .reset_index(drop=True)
    )


def calculate_one_for_one_raw_differential(
    incoming_player: pd.Series,
    outgoing_player: pd.Series,
    production_metric_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Calculate incoming minus outgoing for the raw production metrics.

    The roster CSV currently retains transaction-style ``net_`` column names,
    but each row is treated as an individual player's value.
    """
    differential: dict[str, float] = {}

    for column in production_metric_columns:
        if column not in incoming_player.index:
            raise KeyError(f"Incoming player is missing metric: {column}")
        if column not in outgoing_player.index:
            raise KeyError(f"Outgoing player is missing metric: {column}")

        differential[column] = (
            float(incoming_player[column]) - float(outgoing_player[column])
        )

    return pd.DataFrame([differential])


def score_player_production_differential(
    raw_differential: pd.DataFrame,
    production_reference: pd.DataFrame,
    production_metric_columns: Sequence[str],
    production_domains: Mapping[str, Sequence[str]],
    turnover_zscore_column: str,
) -> pd.DataFrame:
    """
    Standardize a raw player-package differential against the development
    reference, build the domains, and return the final production z-score.
    """
    metric_columns = list(production_metric_columns)
    zscore_columns = [f"{column}_zscore" for column in metric_columns]

    missing_raw = [
        column for column in metric_columns if column not in raw_differential.columns
    ]
    missing_reference = [
        column for column in metric_columns if column not in production_reference.columns
    ]

    if missing_raw:
        raise KeyError(f"Raw differential is missing metrics: {missing_raw}")
    if missing_reference:
        raise KeyError(
            f"Production reference is missing metrics: {missing_reference}"
        )

    if "on_court_production_differential" not in production_reference.columns:
        raise KeyError(
            "Production reference is missing "
            "'on_court_production_differential'."
        )

    mapped_zscores = stats.zmap(
        raw_differential[metric_columns],
        production_reference[metric_columns],
        axis=0,
        ddof=0,
        nan_policy="omit",
    )

    differential_zscores = pd.DataFrame(
        np.asarray(mapped_zscores),
        index=raw_differential.index,
        columns=zscore_columns,
    )

    if turnover_zscore_column not in differential_zscores.columns:
        raise KeyError(
            f"Turnover z-score column not found: {turnover_zscore_column}"
        )

    differential_zscores[turnover_zscore_column] *= -1

    scored = pd.concat(
        [raw_differential.copy(), differential_zscores],
        axis=1,
    )

    domain_columns: list[str] = []

    for domain_name, domain_metric_columns in production_domains.items():
        missing_domain_metrics = [
            column
            for column in domain_metric_columns
            if column not in scored.columns
        ]
        if missing_domain_metrics:
            raise KeyError(
                f"Domain '{domain_name}' is missing metrics: "
                f"{missing_domain_metrics}"
            )

        domain_column = f"{domain_name}_production_differential"
        scored[domain_column] = scored[list(domain_metric_columns)].mean(
            axis=1,
            skipna=True,
        )
        domain_columns.append(domain_column)

    scored["production_domain_count_available"] = (
        scored[domain_columns].notna().sum(axis=1)
    )

    scored["on_court_production_differential"] = scored[domain_columns].mean(
        axis=1,
        skipna=True,
    )

    scored["on_court_production_differential_zscore"] = stats.zmap(
        scored["on_court_production_differential"],
        production_reference["on_court_production_differential"],
        ddof=0,
        nan_policy="omit",
    )

    return scored


def add_draft_compensation_score(
    scored_player_trade: pd.DataFrame,
    net_draft_counts: Mapping[str, int | float],
    outright_pick_net_columns: Sequence[str],
    outright_pick_weights: Sequence[float],
    draft_model_intercept: float,
) -> pd.DataFrame:
    """Add modeled outright-pick compensation and the balance residual."""
    net_columns = list(outright_pick_net_columns)
    weights = np.asarray(outright_pick_weights, dtype="float64")

    if len(net_columns) != len(weights):
        raise ValueError(
            "outright_pick_net_columns and outright_pick_weights must "
            "have the same length."
        )

    missing_pick_columns = [
        column for column in net_columns if column not in net_draft_counts
    ]
    if missing_pick_columns:
        raise KeyError(f"Missing pick counts: {missing_pick_columns}")

    draft_vector = np.asarray(
        [net_draft_counts[column] for column in net_columns],
        dtype="float64",
    )

    expected_production_differential = (
        float(draft_model_intercept) - draft_vector @ weights
    )

    result = scored_player_trade.copy()

    for column in net_columns:
        result[column] = float(net_draft_counts[column])

    result[
        "expected_production_differential_from_outright_picks"
    ] = expected_production_differential

    result["historical_balance_residual"] = (
        result["on_court_production_differential_zscore"]
        - expected_production_differential
    )

    return result


def score_one_for_one_trade(
    roster_data: pd.DataFrame,
    incoming_player_name: str,
    outgoing_player_name: str,
    net_draft_counts: Mapping[str, int | float],
    production_reference: pd.DataFrame,
    production_metric_columns: Sequence[str],
    production_domains: Mapping[str, Sequence[str]],
    turnover_zscore_column: str,
    outright_pick_net_columns: Sequence[str],
    outright_pick_weights: Sequence[float],
    draft_model_intercept: float,
    team_column: str = "team",
    player_name_column: str = "candidate_player",
) -> pd.DataFrame:
    """Score one team perspective of a one-player-for-one-player proposal."""
    incoming_player = get_roster_player(
        roster_data,
        incoming_player_name,
        player_name_column=player_name_column,
    )
    outgoing_player = get_roster_player(
        roster_data,
        outgoing_player_name,
        player_name_column=player_name_column,
    )

    if team_column not in incoming_player.index:
        raise KeyError(f"Roster is missing team column: {team_column}")

    if incoming_player[team_column] == outgoing_player[team_column]:
        raise ValueError(
            "Incoming and outgoing players must belong to different teams."
        )

    raw_differential = calculate_one_for_one_raw_differential(
        incoming_player=incoming_player,
        outgoing_player=outgoing_player,
        production_metric_columns=production_metric_columns,
    )

    scored_trade = score_player_production_differential(
        raw_differential=raw_differential,
        production_reference=production_reference,
        production_metric_columns=production_metric_columns,
        production_domains=production_domains,
        turnover_zscore_column=turnover_zscore_column,
    )

    scored_trade.insert(0, "incoming_player", incoming_player_name)
    scored_trade.insert(1, "incoming_team", incoming_player[team_column])
    scored_trade.insert(2, "outgoing_player", outgoing_player_name)
    scored_trade.insert(3, "outgoing_team", outgoing_player[team_column])

    return add_draft_compensation_score(
        scored_player_trade=scored_trade,
        net_draft_counts=net_draft_counts,
        outright_pick_net_columns=outright_pick_net_columns,
        outright_pick_weights=outright_pick_weights,
        draft_model_intercept=draft_model_intercept,
    )


def score_two_team_one_for_one_trade(
    roster_data: pd.DataFrame,
    team_a_player_name: str,
    team_b_player_name: str,
    team_a_draft_counts: Mapping[str, int | float],
    production_reference: pd.DataFrame,
    production_metric_columns: Sequence[str],
    production_domains: Mapping[str, Sequence[str]],
    turnover_zscore_column: str,
    outright_pick_net_columns: Sequence[str],
    outright_pick_weights: Sequence[float],
    draft_model_intercept: float,
    team_a_name: str | None = None,
    team_b_name: str | None = None,
    team_column: str = "team",
    player_name_column: str = "candidate_player",
) -> dict[str, Any]:
    """
    Score both perspectives and return a symmetric residual from Team A's
    perspective. Negative means Team A needs additional value.
    """
    normalized_team_a_counts = {
        column: float(team_a_draft_counts.get(column, 0))
        for column in outright_pick_net_columns
    }
    team_b_draft_counts = {
        column: -count for column, count in normalized_team_a_counts.items()
    }

    team_a_player = get_roster_player(
        roster_data,
        team_a_player_name,
        player_name_column=player_name_column,
    )
    team_b_player = get_roster_player(
        roster_data,
        team_b_player_name,
        player_name_column=player_name_column,
    )

    resolved_team_a_name = str(team_a_player[team_column])
    resolved_team_b_name = str(team_b_player[team_column])

    if team_a_name is not None and resolved_team_a_name != team_a_name:
        raise ValueError(
            f"{team_a_player_name} belongs to {resolved_team_a_name}, "
            f"not {team_a_name}."
        )
    if team_b_name is not None and resolved_team_b_name != team_b_name:
        raise ValueError(
            f"{team_b_player_name} belongs to {resolved_team_b_name}, "
            f"not {team_b_name}."
        )

    team_a_result = score_one_for_one_trade(
        roster_data=roster_data,
        incoming_player_name=team_b_player_name,
        outgoing_player_name=team_a_player_name,
        net_draft_counts=normalized_team_a_counts,
        production_reference=production_reference,
        production_metric_columns=production_metric_columns,
        production_domains=production_domains,
        turnover_zscore_column=turnover_zscore_column,
        outright_pick_net_columns=outright_pick_net_columns,
        outright_pick_weights=outright_pick_weights,
        draft_model_intercept=draft_model_intercept,
        team_column=team_column,
        player_name_column=player_name_column,
    )

    team_b_result = score_one_for_one_trade(
        roster_data=roster_data,
        incoming_player_name=team_a_player_name,
        outgoing_player_name=team_b_player_name,
        net_draft_counts=team_b_draft_counts,
        production_reference=production_reference,
        production_metric_columns=production_metric_columns,
        production_domains=production_domains,
        turnover_zscore_column=turnover_zscore_column,
        outright_pick_net_columns=outright_pick_net_columns,
        outright_pick_weights=outright_pick_weights,
        draft_model_intercept=draft_model_intercept,
        team_column=team_column,
        player_name_column=player_name_column,
    )

    team_a_residual = float(
        team_a_result.loc[0, "historical_balance_residual"]
    )
    team_b_residual = float(
        team_b_result.loc[0, "historical_balance_residual"]
    )

    team_a_symmetric_residual = (
        team_a_residual - team_b_residual
    ) / 2.0

    return {
        "team_a_name": resolved_team_a_name,
        "team_b_name": resolved_team_b_name,
        "team_a_player": team_a_player_name,
        "team_b_player": team_b_player_name,
        "team_a_result": team_a_result,
        "team_b_result": team_b_result,
        "team_a_residual": team_a_residual,
        "team_b_residual": team_b_residual,
        "team_a_symmetric_residual": team_a_symmetric_residual,
        "absolute_symmetric_residual": abs(team_a_symmetric_residual),
    }


def calculate_slot_adjusted_package_value(
    individual_values: Sequence[float] | pd.Series | np.ndarray,
    additional_player_slot_cost: float,
) -> float:
    """
    Calculate the slot-adjusted value of one outgoing player package.

    The highest-valued player contributes full value. Every additional player
    contributes only the amount above the selected rotation-slot cost.
    """
    slot_cost = float(additional_player_slot_cost)
    if not np.isfinite(slot_cost) or slot_cost < 0:
        raise ValueError(
            "additional_player_slot_cost must be a finite nonnegative number."
        )

    values = pd.to_numeric(
        pd.Series(list(individual_values), dtype="object"),
        errors="coerce",
    )

    if values.empty:
        return 0.0
    if values.isna().any():
        raise ValueError("Player package contains missing or invalid values.")
    if values.lt(0).any():
        raise ValueError("Individual player values cannot be negative.")

    ordered_values = np.sort(values.to_numpy(dtype="float64"))[::-1]
    leading_value = float(ordered_values[0])
    additional_contribution = float(
        np.clip(ordered_values[1:] - slot_cost, 0.0, None).sum()
    )
    return leading_value + additional_contribution



def calculate_player_match_profile(
    team_a_players: pd.DataFrame,
    team_b_players: pd.DataFrame,
    player_value_column: str,
    player_name_column: str = "candidate_player",
) -> dict[str, Any]:
    """
    Match players by descending individual production value.

    The highest-valued player on each side is paired first, followed by the
    second-highest player, and so on. The shorter package is padded with
    zero-valued placeholders. The resulting L1 distance measures how closely
    the two sides exchange comparable individual production before picks are
    considered.
    """
    required_columns = {player_name_column, player_value_column}
    for label, players in (("Team A", team_a_players), ("Team B", team_b_players)):
        if not isinstance(players, pd.DataFrame):
            raise TypeError(f"{label} players must be a pandas DataFrame.")
        missing_columns = sorted(required_columns - set(players.columns))
        if missing_columns:
            raise KeyError(
                f"{label} player package is missing columns: {missing_columns}"
            )

    def prepare(players: pd.DataFrame) -> pd.DataFrame:
        prepared = players[[player_name_column, player_value_column]].copy()
        prepared[player_name_column] = prepared[player_name_column].astype(str)
        prepared[player_value_column] = pd.to_numeric(
            prepared[player_value_column],
            errors="coerce",
        )
        if prepared[player_value_column].isna().any():
            invalid_players = prepared.loc[
                prepared[player_value_column].isna(),
                player_name_column,
            ].tolist()
            raise ValueError(
                f"Player matching contains missing values for: {invalid_players}"
            )
        if prepared[player_value_column].lt(0).any():
            raise ValueError("Individual player values cannot be negative.")
        return prepared.sort_values(
            [player_value_column, player_name_column],
            ascending=[False, True],
        ).reset_index(drop=True)

    team_a = prepare(team_a_players)
    team_b = prepare(team_b_players)
    pair_count = max(len(team_a), len(team_b))
    pair_records: list[dict[str, Any]] = []

    for pair_index in range(pair_count):
        team_a_present = pair_index < len(team_a)
        team_b_present = pair_index < len(team_b)

        team_a_name = (
            str(team_a.at[pair_index, player_name_column])
            if team_a_present
            else None
        )
        team_b_name = (
            str(team_b.at[pair_index, player_name_column])
            if team_b_present
            else None
        )
        team_a_value = (
            float(team_a.at[pair_index, player_value_column])
            if team_a_present
            else 0.0
        )
        team_b_value = (
            float(team_b.at[pair_index, player_value_column])
            if team_b_present
            else 0.0
        )
        absolute_gap = abs(team_a_value - team_b_value)
        is_unmatched = not (team_a_present and team_b_present)

        pair_records.append(
            {
                "pair_rank": pair_index + 1,
                "team_a_player": team_a_name,
                "team_a_player_value": team_a_value,
                "team_b_player": team_b_name,
                "team_b_player_value": team_b_value,
                "absolute_player_value_gap": absolute_gap,
                "is_unmatched_pair": is_unmatched,
            }
        )

    player_match_cost = float(
        sum(record["absolute_player_value_gap"] for record in pair_records)
    )
    unmatched_player_value = float(
        sum(
            record["absolute_player_value_gap"]
            for record in pair_records
            if record["is_unmatched_pair"]
        )
    )
    maximum_player_match_gap = float(
        max(
            (
                record["absolute_player_value_gap"]
                for record in pair_records
            ),
            default=0.0,
        )
    )

    return {
        "player_match_cost": player_match_cost,
        "unmatched_player_value": unmatched_player_value,
        "maximum_player_match_gap": maximum_player_match_gap,
        "matched_player_pair_count": int(
            sum(not record["is_unmatched_pair"] for record in pair_records)
        ),
        "unmatched_player_pair_count": int(
            sum(record["is_unmatched_pair"] for record in pair_records)
        ),
        "player_match_pairs": pair_records,
    }



def score_two_team_multi_player_trade(
    roster_data: pd.DataFrame,
    team_a_player_names: str | Sequence[str] | None,
    team_b_player_names: str | Sequence[str] | None,
    team_a_draft_counts: Mapping[str, int | float],
    player_value_column: str,
    outright_pick_net_columns: Sequence[str],
    outright_pick_weights: Sequence[float],
    team_a_name: str,
    team_b_name: str,
    additional_player_slot_cost: float,
    team_a_itemized_pick_adjustment: float | None = None,
    team_a_initial_pick_assets: pd.DataFrame | None = None,
    team_b_initial_pick_assets: pd.DataFrame | None = None,
    team_column: str = "team",
    player_name_column: str = "candidate_player",
) -> dict[str, Any]:
    """
    Score a multi-player proposal from Team A's perspective.

    Each side's outgoing player package is valued with the selected slot cost.
    Positive Team A draft counts mean Team A receives the picks.
    """
    team_a_names = _normalize_player_names(
        team_a_player_names,
        argument_name="team_a_player_names",
        allow_empty=True,
    )
    team_b_names = _normalize_player_names(
        team_b_player_names,
        argument_name="team_b_player_names",
        allow_empty=True,
    )

    overlap = sorted(set(team_a_names) & set(team_b_names))
    if overlap:
        raise ValueError(
            f"The same players appear on both sides of the trade: {overlap}"
        )

    required_columns = {
        team_column,
        player_name_column,
        player_value_column,
    }
    missing_columns = sorted(required_columns - set(roster_data.columns))
    if missing_columns:
        raise KeyError(f"Roster is missing columns: {missing_columns}")

    team_a_players = get_roster_players(
        roster_data,
        team_a_names,
        player_name_column=player_name_column,
        allow_empty=True,
    )
    team_b_players = get_roster_players(
        roster_data,
        team_b_names,
        player_name_column=player_name_column,
        allow_empty=True,
    )

    invalid_team_a = team_a_players.loc[
        ~team_a_players[team_column].astype(str).eq(str(team_a_name)),
        player_name_column,
    ].astype(str).tolist()
    invalid_team_b = team_b_players.loc[
        ~team_b_players[team_column].astype(str).eq(str(team_b_name)),
        player_name_column,
    ].astype(str).tolist()

    if invalid_team_a:
        raise ValueError(
            f"These players do not belong to {team_a_name}: {invalid_team_a}"
        )
    if invalid_team_b:
        raise ValueError(
            f"These players do not belong to {team_b_name}: {invalid_team_b}"
        )

    team_a_values = pd.to_numeric(
        team_a_players[player_value_column],
        errors="coerce",
    )
    team_b_values = pd.to_numeric(
        team_b_players[player_value_column],
        errors="coerce",
    )

    if team_a_values.isna().any():
        missing_value_players = team_a_players.loc[
            team_a_values.isna(),
            player_name_column,
        ].astype(str).tolist()
        raise ValueError(
            f"Missing or invalid player values for {missing_value_players}"
        )
    if team_b_values.isna().any():
        missing_value_players = team_b_players.loc[
            team_b_values.isna(),
            player_name_column,
        ].astype(str).tolist()
        raise ValueError(
            f"Missing or invalid player values for {missing_value_players}"
        )

    net_columns = list(outright_pick_net_columns)
    weights = np.asarray(outright_pick_weights, dtype="float64")
    if len(net_columns) != len(weights):
        raise ValueError(
            "outright_pick_net_columns and outright_pick_weights must "
            "have the same length."
        )

    normalized_team_a_counts = {
        column: float(team_a_draft_counts.get(column, 0))
        for column in net_columns
    }
    draft_vector = np.asarray(
        [normalized_team_a_counts[column] for column in net_columns],
        dtype="float64",
    )

    team_a_raw_player_value_sent = float(team_a_values.sum())
    team_b_raw_player_value_sent = float(team_b_values.sum())
    team_a_player_package_value_sent = calculate_slot_adjusted_package_value(
        team_a_values,
        additional_player_slot_cost=additional_player_slot_cost,
    )
    team_b_player_package_value_sent = calculate_slot_adjusted_package_value(
        team_b_values,
        additional_player_slot_cost=additional_player_slot_cost,
    )

    team_a_player_value_received = team_b_player_package_value_sent
    team_a_player_value_sent = team_a_player_package_value_sent
    team_a_player_differential = (
        team_a_player_value_received - team_a_player_value_sent
    )
    count_based_draft_adjustment = float(draft_vector @ weights)
    if team_a_itemized_pick_adjustment is None:
        team_a_draft_adjustment = count_based_draft_adjustment
        pick_adjustment_method = "tier_count_weights"
    else:
        team_a_draft_adjustment = float(team_a_itemized_pick_adjustment)
        if not np.isfinite(team_a_draft_adjustment):
            raise ValueError("team_a_itemized_pick_adjustment must be finite.")
        pick_adjustment_method = "item_level_adjusted_value"

    team_a_symmetric_residual = (
        team_a_player_differential + team_a_draft_adjustment
    )
    player_match_profile = calculate_player_match_profile(
        team_a_players=team_a_players,
        team_b_players=team_b_players,
        player_value_column=player_value_column,
        player_name_column=player_name_column,
    )

    return {
        "team_a_name": str(team_a_name),
        "team_b_name": str(team_b_name),
        "team_a_players_sent": team_a_names,
        "team_b_players_sent": team_b_names,
        "team_a_player_rows": team_a_players,
        "team_b_player_rows": team_b_players,
        "additional_player_slot_cost": float(additional_player_slot_cost),
        "team_a_raw_player_value_sent": team_a_raw_player_value_sent,
        "team_b_raw_player_value_sent": team_b_raw_player_value_sent,
        "team_a_player_package_value_sent": team_a_player_package_value_sent,
        "team_b_player_package_value_sent": team_b_player_package_value_sent,
        **player_match_profile,
        # Store the player and draft results displayed by the application.
        "team_a_player_value_sent": team_a_player_value_sent,
        "team_a_player_value_received": team_a_player_value_received,
        "team_a_player_differential": team_a_player_differential,
        "team_a_draft_counts": normalized_team_a_counts,
        "team_b_draft_counts": {
            column: -count
            for column, count in normalized_team_a_counts.items()
        },
        "team_a_draft_adjustment": team_a_draft_adjustment,
        "count_based_draft_adjustment": count_based_draft_adjustment,
        "pick_adjustment_method": pick_adjustment_method,
        "team_a_initial_pick_assets": (
            team_a_initial_pick_assets.copy()
            if isinstance(team_a_initial_pick_assets, pd.DataFrame)
            else pd.DataFrame()
        ),
        "team_b_initial_pick_assets": (
            team_b_initial_pick_assets.copy()
            if isinstance(team_b_initial_pick_assets, pd.DataFrame)
            else pd.DataFrame()
        ),
        "team_a_initial_pick_value_sent": (
            float(pd.to_numeric(
                team_a_initial_pick_assets.get("adjusted_value", pd.Series(dtype=float)),
                errors="coerce",
            ).sum())
            if isinstance(team_a_initial_pick_assets, pd.DataFrame)
            else 0.0
        ),
        "team_b_initial_pick_value_sent": (
            float(pd.to_numeric(
                team_b_initial_pick_assets.get("adjusted_value", pd.Series(dtype=float)),
                errors="coerce",
            ).sum())
            if isinstance(team_b_initial_pick_assets, pd.DataFrame)
            else 0.0
        ),
        "team_a_symmetric_residual": team_a_symmetric_residual,
        "team_b_symmetric_residual": -team_a_symmetric_residual,
        "absolute_symmetric_residual": abs(team_a_symmetric_residual),
    }



def generate_player_addition_alternatives(
    roster_data: pd.DataFrame,
    team_a_name: str,
    team_b_name: str,
    locked_team_a_player: str | Sequence[str],
    locked_team_b_player: str | Sequence[str],
    base_team_a_residual: float,
    player_value_column: str,
    additional_player_slot_cost: float,
    max_additional_players_per_team: int = 2,
    max_total_additional_players: int = 3,
    balance_tolerance: float = 0.05,
    team_column: str = "team",
    player_name_column: str = "candidate_player",
    roster_slot_column: str = "roster_slot",
    require_directional_improvement: bool = True,
) -> pd.DataFrame:
    """
    Search additional-player packages using full slot-adjusted package values.

    Players may be added to either or both sides. For each candidate, the
    complete locked-plus-additional package is recalculated; the model does not
    simply sum the additional players' individual values.
    """
    if max_additional_players_per_team < 0:
        raise ValueError("max_additional_players_per_team cannot be negative.")
    if max_total_additional_players < 1:
        raise ValueError("max_total_additional_players must be at least 1.")

    required_columns = {
        team_column,
        player_name_column,
        player_value_column,
    }
    missing_columns = sorted(required_columns - set(roster_data.columns))
    if missing_columns:
        raise KeyError(f"Roster is missing columns: {missing_columns}")

    selected_columns = [player_name_column, player_value_column]
    if roster_slot_column in roster_data.columns:
        selected_columns.insert(1, roster_slot_column)

    locked_team_a_players = _normalize_player_names(
        locked_team_a_player,
        argument_name="locked_team_a_player",
        allow_empty=True,
    )
    locked_team_b_players = _normalize_player_names(
        locked_team_b_player,
        argument_name="locked_team_b_player",
        allow_empty=True,
    )

    locked_team_a_rows = get_roster_players(
        roster_data,
        locked_team_a_players,
        player_name_column=player_name_column,
        allow_empty=True,
    )
    locked_team_b_rows = get_roster_players(
        roster_data,
        locked_team_b_players,
        player_name_column=player_name_column,
        allow_empty=True,
    )

    for rows, expected_team, label in [
        (locked_team_a_rows, team_a_name, "Team A"),
        (locked_team_b_rows, team_b_name, "Team B"),
    ]:
        invalid = rows.loc[
            ~rows[team_column].astype(str).eq(str(expected_team)),
            player_name_column,
        ].astype(str).tolist()
        if invalid:
            raise ValueError(
                f"{label} locked players do not belong to {expected_team}: "
                f"{invalid}"
            )

    locked_team_a_values = pd.to_numeric(
        locked_team_a_rows[player_value_column], errors="coerce"
    )
    locked_team_b_values = pd.to_numeric(
        locked_team_b_rows[player_value_column], errors="coerce"
    )
    if locked_team_a_values.isna().any() or locked_team_b_values.isna().any():
        raise ValueError("Locked player packages contain invalid player values.")

    base_team_a_package_value = calculate_slot_adjusted_package_value(
        locked_team_a_values,
        additional_player_slot_cost=additional_player_slot_cost,
    )
    base_team_b_package_value = calculate_slot_adjusted_package_value(
        locked_team_b_values,
        additional_player_slot_cost=additional_player_slot_cost,
    )
    base_player_differential = (
        base_team_b_package_value - base_team_a_package_value
    )
    base_player_match_profile = calculate_player_match_profile(
        team_a_players=locked_team_a_rows,
        team_b_players=locked_team_b_rows,
        player_value_column=player_value_column,
        player_name_column=player_name_column,
    )
    # This preserves any draft adjustment already present in a sequential mode.
    non_player_residual_component = (
        float(base_team_a_residual) - base_player_differential
    )

    team_a_candidates = roster_data.loc[
        roster_data[team_column].astype(str).eq(str(team_a_name))
        & ~roster_data[player_name_column].isin(locked_team_a_players),
        selected_columns,
    ].copy()
    team_b_candidates = roster_data.loc[
        roster_data[team_column].astype(str).eq(str(team_b_name))
        & ~roster_data[player_name_column].isin(locked_team_b_players),
        selected_columns,
    ].copy()

    for candidate_data in (team_a_candidates, team_b_candidates):
        candidate_data[player_value_column] = pd.to_numeric(
            candidate_data[player_value_column],
            errors="coerce",
        )

    team_a_candidates = team_a_candidates.loc[
        team_a_candidates[player_value_column].gt(0)
    ].reset_index(drop=True)
    team_b_candidates = team_b_candidates.loc[
        team_b_candidates[player_value_column].gt(0)
    ].reset_index(drop=True)

    candidate_records: list[dict[str, Any]] = []
    max_team_a_count = min(
        max_additional_players_per_team,
        len(team_a_candidates),
    )
    max_team_b_count = min(
        max_additional_players_per_team,
        len(team_b_candidates),
    )

    for team_a_count in range(max_team_a_count + 1):
        for team_b_count in range(max_team_b_count + 1):
            total_player_count = team_a_count + team_b_count
            if total_player_count == 0:
                continue
            if total_player_count > max_total_additional_players:
                continue

            for team_a_indices in combinations(
                team_a_candidates.index,
                team_a_count,
            ):
                team_a_package = team_a_candidates.loc[list(team_a_indices)]
                team_a_additional_values = pd.to_numeric(
                    team_a_package[player_value_column], errors="coerce"
                )

                for team_b_indices in combinations(
                    team_b_candidates.index,
                    team_b_count,
                ):
                    team_b_package = team_b_candidates.loc[list(team_b_indices)]
                    team_b_additional_values = pd.to_numeric(
                        team_b_package[player_value_column], errors="coerce"
                    )

                    full_team_a_values = pd.concat(
                        [locked_team_a_values, team_a_additional_values],
                        ignore_index=True,
                    )
                    full_team_b_values = pd.concat(
                        [locked_team_b_values, team_b_additional_values],
                        ignore_index=True,
                    )
                    full_team_a_rows = pd.concat(
                        [
                            locked_team_a_rows[selected_columns],
                            team_a_package[selected_columns],
                        ],
                        ignore_index=True,
                    )
                    full_team_b_rows = pd.concat(
                        [
                            locked_team_b_rows[selected_columns],
                            team_b_package[selected_columns],
                        ],
                        ignore_index=True,
                    )
                    player_match_profile = calculate_player_match_profile(
                        team_a_players=full_team_a_rows,
                        team_b_players=full_team_b_rows,
                        player_value_column=player_value_column,
                        player_name_column=player_name_column,
                    )

                    team_a_full_package_value = (
                        calculate_slot_adjusted_package_value(
                            full_team_a_values,
                            additional_player_slot_cost=(
                                additional_player_slot_cost
                            ),
                        )
                    )
                    team_b_full_package_value = (
                        calculate_slot_adjusted_package_value(
                            full_team_b_values,
                            additional_player_slot_cost=(
                                additional_player_slot_cost
                            ),
                        )
                    )

                    candidate_player_differential = (
                        team_b_full_package_value - team_a_full_package_value
                    )
                    net_team_a_adjustment = (
                        candidate_player_differential
                        - base_player_differential
                    )

                    if require_directional_improvement:
                        if base_team_a_residual < 0:
                            direction_is_valid = net_team_a_adjustment > 0
                        elif base_team_a_residual > 0:
                            direction_is_valid = net_team_a_adjustment < 0
                        else:
                            direction_is_valid = False
                        if not direction_is_valid:
                            continue

                    residual_after_players = (
                        candidate_player_differential
                        + non_player_residual_component
                    )
                    residual_improvement = (
                        abs(float(base_team_a_residual))
                        - abs(residual_after_players)
                    )
                    if require_directional_improvement and residual_improvement <= 0:
                        continue

                    team_a_incremental_package_value = (
                        team_a_full_package_value - base_team_a_package_value
                    )
                    team_b_incremental_package_value = (
                        team_b_full_package_value - base_team_b_package_value
                    )

                    candidate_records.append(
                        {
                            "team_a_additional_players_sent": (
                                team_a_package[player_name_column].tolist()
                            ),
                            "team_b_additional_players_sent": (
                                team_b_package[player_name_column].tolist()
                            ),
                            "team_a_additional_raw_value_sent": float(
                                team_a_additional_values.sum()
                            ),
                            "team_b_additional_raw_value_sent": float(
                                team_b_additional_values.sum()
                            ),
                            # Existing names now mean incremental package value.
                            "team_a_additional_value_sent": float(
                                team_a_incremental_package_value
                            ),
                            "team_b_additional_value_sent": float(
                                team_b_incremental_package_value
                            ),
                            "team_a_full_player_package_value_sent": float(
                                team_a_full_package_value
                            ),
                            "team_b_full_player_package_value_sent": float(
                                team_b_full_package_value
                            ),
                            "base_team_a_player_package_value_sent": float(
                                base_team_a_package_value
                            ),
                            "base_team_b_player_package_value_sent": float(
                                base_team_b_package_value
                            ),
                            **player_match_profile,
                            "base_player_match_cost": float(
                                base_player_match_profile["player_match_cost"]
                            ),
                            "player_match_improvement": float(
                                base_player_match_profile["player_match_cost"]
                                - player_match_profile["player_match_cost"]
                            ),
                            "candidate_player_differential": float(
                                candidate_player_differential
                            ),
                            "net_team_a_player_adjustment": float(
                                net_team_a_adjustment
                            ),
                            "team_a_additional_player_count": team_a_count,
                            "team_b_additional_player_count": team_b_count,
                            "total_additional_players": total_player_count,
                            "base_team_a_residual": float(base_team_a_residual),
                            "residual_after_players": float(
                                residual_after_players
                            ),
                            "absolute_residual_after_players": abs(
                                residual_after_players
                            ),
                            "residual_improvement": float(residual_improvement),
                            "both_teams_add_players": (
                                team_a_count > 0 and team_b_count > 0
                            ),
                            "crosses_balance_point": (
                                np.sign(residual_after_players)
                                != np.sign(base_team_a_residual)
                            ),
                            "within_balance_tolerance": (
                                abs(residual_after_players)
                                <= balance_tolerance
                            ),
                        }
                    )

    return pd.DataFrame(candidate_records)


def rank_player_addition_alternatives(
    alternatives: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank within-tolerance packages by fewer players first. Rank packages still
    outside tolerance by the closest remaining residual first.
    """
    if alternatives.empty:
        return alternatives.copy()

    required_columns = {
        "within_balance_tolerance",
        "total_additional_players",
        "absolute_residual_after_players",
        "residual_improvement",
    }
    missing_columns = sorted(required_columns - set(alternatives.columns))
    if missing_columns:
        raise KeyError(
            f"Player alternatives are missing columns: {missing_columns}"
        )

    within_tolerance = alternatives.loc[
        alternatives["within_balance_tolerance"]
    ].sort_values(
        [
            "total_additional_players",
            "absolute_residual_after_players",
            "residual_improvement",
        ],
        ascending=[True, True, False],
    )

    outside_tolerance = alternatives.loc[
        ~alternatives["within_balance_tolerance"]
    ].sort_values(
        [
            "absolute_residual_after_players",
            "total_additional_players",
            "residual_improvement",
        ],
        ascending=[True, True, False],
    )

    ranked = pd.concat(
        [within_tolerance, outside_tolerance],
        ignore_index=True,
    )
    ranked.insert(
        0,
        "recommendation_rank",
        np.arange(1, len(ranked) + 1),
    )
    return ranked



def apply_player_addition_phase(
    starting_residual: float,
    roster_data: pd.DataFrame,
    team_a_name: str,
    team_b_name: str,
    locked_team_a_player: str | Sequence[str],
    locked_team_b_player: str | Sequence[str],
    player_value_column: str,
    additional_player_slot_cost: float,
    max_additional_players_per_team: int = 2,
    max_total_additional_players: int = 3,
    balance_tolerance: float = 0.05,
    team_column: str = "team",
    player_name_column: str = "candidate_player",
    roster_slot_column: str = "roster_slot",
) -> dict[str, Any]:
    """Run, rank, and select the best slot-adjusted player package."""
    starting_residual = float(starting_residual)

    if abs(starting_residual) <= balance_tolerance:
        return {
            "player_adjustment_applied": False,
            "reason": "Trade is already within balance tolerance.",
            "alternatives": pd.DataFrame(),
            "selected_player_package": None,
            "starting_residual": starting_residual,
            "player_adjustment": 0.0,
            "final_residual": starting_residual,
            "within_balance_tolerance": True,
        }

    alternatives = generate_player_addition_alternatives(
        roster_data=roster_data,
        team_a_name=team_a_name,
        team_b_name=team_b_name,
        locked_team_a_player=locked_team_a_player,
        locked_team_b_player=locked_team_b_player,
        base_team_a_residual=starting_residual,
        player_value_column=player_value_column,
        additional_player_slot_cost=additional_player_slot_cost,
        max_additional_players_per_team=max_additional_players_per_team,
        max_total_additional_players=max_total_additional_players,
        balance_tolerance=balance_tolerance,
        team_column=team_column,
        player_name_column=player_name_column,
        roster_slot_column=roster_slot_column,
    )
    alternatives = rank_player_addition_alternatives(alternatives)

    if alternatives.empty:
        return {
            "player_adjustment_applied": False,
            "reason": "No player-addition package improves the trade balance.",
            "alternatives": alternatives,
            "selected_player_package": None,
            "starting_residual": starting_residual,
            "player_adjustment": 0.0,
            "final_residual": starting_residual,
            "within_balance_tolerance": (
                abs(starting_residual) <= balance_tolerance
            ),
        }

    selected_package = alternatives.iloc[0].copy()
    final_residual = float(selected_package["residual_after_players"])

    return {
        "player_adjustment_applied": True,
        "reason": "Slot-adjusted player package improves the trade balance.",
        "alternatives": alternatives,
        "selected_player_package": selected_package,
        "starting_residual": starting_residual,
        "player_adjustment": final_residual - starting_residual,
        "final_residual": final_residual,
        "within_balance_tolerance": (
            abs(final_residual) <= balance_tolerance
        ),
    }


def validate_pick_inventory(
    pick_inventory: pd.DataFrame,
    outright_pick_hierarchy: Sequence[str],
) -> None:
    """Validate the one-row-per-pick, item-valued inventory."""
    required_columns = {
        "pick_id",
        "owning_team",
        "draft_year",
        "round",
        "tier",
        "base_value",
        "floor_tier",
        "floor_value",
        "tier_confidence",
        "tier_decay",
        "adjusted_value",
        "currently_owned",
        "available_for_trade",
    }
    missing_columns = sorted(required_columns - set(pick_inventory.columns))
    if missing_columns:
        raise KeyError(f"Pick inventory is missing columns: {missing_columns}")

    if pick_inventory["pick_id"].isna().any():
        raise ValueError("pick_id cannot be missing.")
    if not pick_inventory["pick_id"].is_unique:
        raise ValueError("pick_id must be unique.")

    supported_tiers = set(outright_pick_hierarchy)
    unsupported_tiers = sorted(
        set(pick_inventory["tier"].dropna()) - supported_tiers
    )
    if unsupported_tiers:
        raise ValueError(
            f"Pick inventory contains unsupported tiers: {unsupported_tiers}"
        )

    unsupported_floor_tiers = sorted(
        set(pick_inventory["floor_tier"].dropna()) - supported_tiers
    )
    if unsupported_floor_tiers:
        raise ValueError(
            "Pick inventory contains unsupported floor tiers: "
            f"{unsupported_floor_tiers}"
        )

    numeric_columns = [
        "draft_year",
        "round",
        "base_value",
        "floor_value",
        "tier_confidence",
        "tier_decay",
        "adjusted_value",
    ]
    numeric = pick_inventory[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if numeric.isna().any().any():
        bad_columns = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(
            "Pick inventory contains missing or nonnumeric values in: "
            f"{bad_columns}"
        )

    if not numeric["round"].isin([1, 2]).all():
        raise ValueError("Pick round must be 1 or 2.")
    if numeric["base_value"].lt(0).any():
        raise ValueError("Pick base_value cannot be negative.")
    if numeric["floor_value"].lt(0).any():
        raise ValueError("Pick floor_value cannot be negative.")
    if numeric["adjusted_value"].lt(0).any():
        raise ValueError("Pick adjusted_value cannot be negative.")
    if not numeric["tier_confidence"].between(0, 1).all():
        raise ValueError("tier_confidence must be between 0 and 1.")
    if not numeric["tier_decay"].between(0, 1).all():
        raise ValueError("tier_decay must be between 0 and 1.")

    if not np.allclose(
        numeric["tier_confidence"] + numeric["tier_decay"],
        1.0,
        atol=1e-9,
    ):
        raise ValueError("tier_confidence + tier_decay must equal 1.")

    lower_bound = np.minimum(
        numeric["base_value"],
        numeric["floor_value"],
    )
    upper_bound = np.maximum(
        numeric["base_value"],
        numeric["floor_value"],
    )
    if (numeric["adjusted_value"] < lower_bound - 1e-9).any():
        raise ValueError("adjusted_value cannot fall below the one-tier floor.")
    if (numeric["adjusted_value"] > upper_bound + 1e-9).any():
        raise ValueError("adjusted_value cannot exceed base_value.")



def get_available_pick_inventory(
    pick_inventory: pd.DataFrame,
    team_name: str,
    outright_pick_hierarchy: Sequence[str],
    excluded_pick_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return specific eligible assets and available counts by tier."""
    validate_pick_inventory(
        pick_inventory=pick_inventory,
        outright_pick_hierarchy=outright_pick_hierarchy,
    )

    excluded = set(str(value) for value in (excluded_pick_ids or []))

    currently_owned = pick_inventory["currently_owned"].fillna(False).astype(bool)
    available_for_trade = (
        pick_inventory["available_for_trade"].fillna(False).astype(bool)
    )

    available_picks = pick_inventory.loc[
        pick_inventory["owning_team"].astype(str).eq(str(team_name))
        & currently_owned
        & available_for_trade
        & pick_inventory["tier"].isin(outright_pick_hierarchy)
        & ~pick_inventory["pick_id"].astype(str).isin(excluded)
    ].copy()

    available_picks["adjusted_value"] = pd.to_numeric(
        available_picks["adjusted_value"],
        errors="coerce",
    )
    available_picks = available_picks.sort_values(
        ["draft_year", "round", "pick_id"]
    ).reset_index(drop=True)

    available_counts = (
        available_picks["tier"]
        .value_counts()
        .reindex(outright_pick_hierarchy, fill_value=0)
        .astype(int)
        .to_dict()
    )

    return available_picks, available_counts


def resolve_initial_pick_assets(
    pick_inventory: pd.DataFrame,
    team_name: str,
    selected_pick_ids: Sequence[str] | None,
    outright_pick_hierarchy: Sequence[str],
) -> pd.DataFrame:
    """Validate and return specific initial-trade picks in requested order."""
    selected_ids = [
        str(value) for value in (selected_pick_ids or []) if str(value).strip()
    ]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError(
            f"Initial pick selection for {team_name} contains duplicates."
        )

    available, _ = get_available_pick_inventory(
        pick_inventory=pick_inventory,
        team_name=team_name,
        outright_pick_hierarchy=outright_pick_hierarchy,
    )
    if not selected_ids:
        return available.iloc[0:0].copy()

    available_lookup = available.set_index(
        available["pick_id"].astype(str),
        drop=False,
    )
    missing = [pick_id for pick_id in selected_ids if pick_id not in available_lookup.index]
    if missing:
        raise ValueError(
            f"Selected initial picks are not available for {team_name}: {missing}"
        )

    resolved = available_lookup.loc[selected_ids].copy()
    return resolved.reset_index(drop=True)


def calculate_initial_pick_adjustment(
    team_a_initial_pick_assets: pd.DataFrame | None,
    team_b_initial_pick_assets: pd.DataFrame | None,
) -> float:
    """Return Team A's item-level value received minus value sent."""
    def package_value(assets: pd.DataFrame | None) -> float:
        if assets is None or assets.empty:
            return 0.0
        if "adjusted_value" not in assets.columns:
            raise KeyError("Initial pick assets are missing adjusted_value.")
        values = pd.to_numeric(assets["adjusted_value"], errors="coerce")
        if values.isna().any() or values.lt(0).any():
            raise ValueError("Initial pick assets contain invalid adjusted values.")
        return float(values.sum())

    team_a_value_sent = package_value(team_a_initial_pick_assets)
    team_b_value_sent = package_value(team_b_initial_pick_assets)
    return team_b_value_sent - team_a_value_sent


def build_initial_draft_counts(
    team_a_initial_pick_assets: pd.DataFrame | None,
    team_b_initial_pick_assets: pd.DataFrame | None,
    outright_pick_hierarchy: Sequence[str],
    outright_pick_net_columns: Sequence[str],
) -> dict[str, float]:
    """Build net tier counts: positive means Team A receives picks."""
    hierarchy = list(outright_pick_hierarchy)
    net_columns = list(outright_pick_net_columns)
    if len(hierarchy) != len(net_columns):
        raise ValueError("Pick hierarchy and net columns must have equal length.")
    tier_to_column = dict(zip(hierarchy, net_columns))
    counts = {column: 0.0 for column in net_columns}

    for assets, sign in (
        (team_a_initial_pick_assets, -1.0),
        (team_b_initial_pick_assets, 1.0),
    ):
        if assets is None or assets.empty:
            continue
        if "tier" not in assets.columns:
            raise KeyError("Initial pick assets are missing tier.")
        for tier in assets["tier"].astype(str):
            if tier not in tier_to_column:
                raise ValueError(f"Unsupported initial pick tier: {tier}")
            counts[tier_to_column[tier]] += sign
    return counts


def recommend_inventory_constrained_pick_package(
    target_value: float,
    available_picks: pd.DataFrame,
    outright_pick_hierarchy: Sequence[str],
    max_total_picks: int = 5,
    balance_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Choose the best specific owned-pick package using item-level values."""
    target_value = abs(float(target_value))

    required_columns = {"pick_id", "tier", "adjusted_value"}
    missing_columns = sorted(required_columns - set(available_picks.columns))
    if missing_columns:
        raise KeyError(
            f"Available pick inventory is missing columns: {missing_columns}"
        )

    if max_total_picks < 1:
        raise ValueError("max_total_picks must be at least 1.")

    candidate_assets = available_picks.copy().reset_index(drop=True)
    candidate_assets["adjusted_value"] = pd.to_numeric(
        candidate_assets["adjusted_value"],
        errors="coerce",
    )
    candidate_assets = candidate_assets.loc[
        candidate_assets["adjusted_value"].notna()
        & candidate_assets["adjusted_value"].ge(0)
    ].copy()

    candidate_packages: list[dict[str, Any]] = []
    maximum_package_size = min(max_total_picks, len(candidate_assets))

    for package_size in range(1, maximum_package_size + 1):
        for selected_indices in combinations(
            candidate_assets.index,
            package_size,
        ):
            selected_assets = candidate_assets.loc[
                list(selected_indices)
            ].copy()
            package_value = float(selected_assets["adjusted_value"].sum())
            absolute_error = abs(target_value - package_value)

            if absolute_error >= target_value:
                continue

            tier_counts = (
                selected_assets["tier"]
                .value_counts()
                .reindex(outright_pick_hierarchy, fill_value=0)
                .astype(int)
                .to_dict()
            )

            candidate_packages.append(
                {
                    "selected_pick_assets": selected_assets,
                    "selected_pick_ids": (
                        selected_assets["pick_id"].astype(str).tolist()
                    ),
                    "package_value": package_value,
                    "absolute_error": absolute_error,
                    "total_picks": package_size,
                    "within_tolerance": (
                        absolute_error <= balance_tolerance
                    ),
                    **{
                        f"{tier}_count": int(tier_counts[tier])
                        for tier in outright_pick_hierarchy
                    },
                }
            )

    if not candidate_packages:
        return {
            "pick_package_available": False,
            "reason": "No owned pick package improves the remaining residual.",
            "recommendation": pd.DataFrame(),
            "selected_pick_assets": pd.DataFrame(),
            "selected_pick_ids": [],
            "package_value": 0.0,
            "remaining_gap": target_value,
            "absolute_error": target_value,
            "within_tolerance": False,
            "total_picks": 0,
        }

    candidate_data = pd.DataFrame(candidate_packages)
    tier_count_columns = [
        f"{tier}_count" for tier in reversed(list(outright_pick_hierarchy))
    ]

    tolerance_candidates = candidate_data.loc[
        candidate_data["within_tolerance"]
    ]

    if not tolerance_candidates.empty:
        ranked_packages = tolerance_candidates.sort_values(
            [
                "total_picks",
                "absolute_error",
                *tier_count_columns,
            ],
            ascending=[
                True,
                True,
                *([False] * len(tier_count_columns)),
            ],
        )
    else:
        ranked_packages = candidate_data.sort_values(
            [
                "absolute_error",
                "total_picks",
                *tier_count_columns,
            ],
            ascending=[
                True,
                True,
                *([False] * len(tier_count_columns)),
            ],
        )

    best_package = ranked_packages.iloc[0]
    selected_assets = best_package["selected_pick_assets"].copy()

    recommendation = (
        selected_assets.groupby("tier", as_index=False)
        .agg(
            count=("pick_id", "size"),
            total_value=("adjusted_value", "sum"),
        )
    )
    recommendation["average_value_per_pick"] = (
        recommendation["total_value"] / recommendation["count"]
    )

    return {
        "pick_package_available": True,
        "reason": "Owned pick package identified.",
        "recommendation": recommendation,
        "selected_pick_assets": selected_assets,
        "selected_pick_ids": best_package["selected_pick_ids"],
        "package_value": float(best_package["package_value"]),
        "remaining_gap": float(target_value - best_package["package_value"]),
        "absolute_error": float(best_package["absolute_error"]),
        "within_tolerance": bool(best_package["within_tolerance"]),
        "total_picks": int(best_package["total_picks"]),
    }



def enumerate_inventory_constrained_pick_packages(
    starting_residual: float,
    team_a_name: str,
    team_b_name: str,
    pick_inventory: pd.DataFrame,
    pick_value_table: pd.DataFrame,
    outright_pick_hierarchy: Sequence[str],
    outright_pick_net_columns: Sequence[str],
    max_total_picks: int = 5,
    balance_tolerance: float = 0.05,
    excluded_pick_ids: Sequence[str] | None = None,
    include_zero_pick_option: bool = True,
) -> pd.DataFrame:
    """Enumerate every specific owned-pick package that improves a residual."""
    del pick_value_table  # Base tier values are already embedded in inventory.

    starting_residual = float(starting_residual)
    starting_absolute_residual = abs(starting_residual)

    if max_total_picks < 0:
        raise ValueError("max_total_picks cannot be negative.")

    net_columns = list(outright_pick_net_columns)
    empty_counts = create_empty_draft_counts(net_columns)
    tier_to_net_column = dict(
        zip(outright_pick_hierarchy, outright_pick_net_columns)
    )

    if starting_residual < 0:
        sending_team = team_b_name
        receiving_team = team_a_name
        adjustment_sign = 1
        team_a_count_sign = 1
    elif starting_residual > 0:
        sending_team = team_a_name
        receiving_team = team_b_name
        adjustment_sign = -1
        team_a_count_sign = -1
    else:
        sending_team = None
        receiving_team = None
        adjustment_sign = 0
        team_a_count_sign = 0

    records: list[dict[str, Any]] = []

    if include_zero_pick_option:
        records.append(
            {
                "pick_sending_team": None,
                "pick_receiving_team": None,
                "selected_pick_ids": [],
                "selected_pick_assets": pd.DataFrame(),
                "team_a_draft_counts": empty_counts.copy(),
                "team_b_draft_counts": empty_counts.copy(),
                "pick_package_value": 0.0,
                "pick_adjustment": 0.0,
                "total_picks": 0,
                "residual_before_picks": starting_residual,
                "final_residual": starting_residual,
                "absolute_final_residual": starting_absolute_residual,
                "residual_improvement": 0.0,
                "within_balance_tolerance": (
                    starting_absolute_residual <= balance_tolerance
                ),
                **{f"{tier}_count": 0 for tier in outright_pick_hierarchy},
            }
        )

    if starting_residual == 0 or max_total_picks == 0:
        return pd.DataFrame(records)

    available_picks, _ = get_available_pick_inventory(
        pick_inventory=pick_inventory,
        team_name=str(sending_team),
        outright_pick_hierarchy=outright_pick_hierarchy,
        excluded_pick_ids=excluded_pick_ids,
    )

    maximum_package_size = min(max_total_picks, len(available_picks))
    for package_size in range(1, maximum_package_size + 1):
        for selected_indices in combinations(
            available_picks.index,
            package_size,
        ):
            selected_assets = available_picks.loc[
                list(selected_indices)
            ].copy()
            package_value = float(selected_assets["adjusted_value"].sum())
            pick_adjustment = adjustment_sign * package_value
            final_residual = starting_residual + pick_adjustment
            improvement = starting_absolute_residual - abs(final_residual)

            if improvement <= 0:
                continue

            tier_counts = (
                selected_assets["tier"]
                .value_counts()
                .reindex(outright_pick_hierarchy, fill_value=0)
                .astype(int)
                .to_dict()
            )
            team_a_draft_counts = empty_counts.copy()
            for tier, count in tier_counts.items():
                net_column = tier_to_net_column[tier]
                team_a_draft_counts[net_column] = (
                    team_a_count_sign * int(count)
                )

            records.append(
                {
                    "pick_sending_team": sending_team,
                    "pick_receiving_team": receiving_team,
                    "selected_pick_ids": (
                        selected_assets["pick_id"].astype(str).tolist()
                    ),
                    "selected_pick_assets": selected_assets,
                    "team_a_draft_counts": team_a_draft_counts,
                    "team_b_draft_counts": {
                        column: -count
                        for column, count in team_a_draft_counts.items()
                    },
                    "pick_package_value": package_value,
                    "pick_adjustment": pick_adjustment,
                    "total_picks": package_size,
                    "residual_before_picks": starting_residual,
                    "final_residual": final_residual,
                    "absolute_final_residual": abs(final_residual),
                    "residual_improvement": improvement,
                    "within_balance_tolerance": (
                        abs(final_residual) <= balance_tolerance
                    ),
                    **{
                        f"{tier}_count": int(tier_counts[tier])
                        for tier in outright_pick_hierarchy
                    },
                }
            )

    return pd.DataFrame(records)



def apply_pick_top_up(
    starting_residual: float,
    team_a_name: str,
    team_b_name: str,
    pick_inventory: pd.DataFrame,
    pick_value_table: pd.DataFrame,
    outright_pick_hierarchy: Sequence[str],
    outright_pick_net_columns: Sequence[str],
    max_total_picks: int = 5,
    balance_tolerance: float = 0.05,
    excluded_pick_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Apply the best available specific owned-pick package."""
    del pick_value_table  # Item-level adjusted values are stored in inventory.

    starting_residual = float(starting_residual)
    starting_absolute_residual = abs(starting_residual)
    empty_draft_counts = create_empty_draft_counts(
        outright_pick_net_columns
    )

    empty_result: dict[str, Any] = {
        "pick_top_up_applied": False,
        "pick_sending_team": None,
        "pick_receiving_team": None,
        "pick_direction": None,
        "recommended_pick_counts": pd.DataFrame(),
        "selected_pick_assets": pd.DataFrame(),
        "team_a_draft_counts": empty_draft_counts.copy(),
        "team_b_draft_counts": empty_draft_counts.copy(),
        "starting_residual": starting_residual,
        "pick_adjustment": 0.0,
        "final_residual": starting_residual,
        "absolute_final_residual": starting_absolute_residual,
        "within_balance_tolerance": (
            starting_absolute_residual <= balance_tolerance
        ),
    }

    if starting_absolute_residual <= balance_tolerance:
        return {
            **empty_result,
            "reason": "Trade is already within balance tolerance.",
        }

    if starting_residual < 0:
        pick_sending_team = team_b_name
        pick_receiving_team = team_a_name
        adjustment_sign = 1
        team_a_net_count_sign = 1
        pick_direction = f"{team_a_name} receives"
    else:
        pick_sending_team = team_a_name
        pick_receiving_team = team_b_name
        adjustment_sign = -1
        team_a_net_count_sign = -1
        pick_direction = f"{team_a_name} sends"

    available_picks, available_pick_counts = get_available_pick_inventory(
        pick_inventory=pick_inventory,
        team_name=pick_sending_team,
        outright_pick_hierarchy=outright_pick_hierarchy,
        excluded_pick_ids=excluded_pick_ids,
    )

    pick_recommendation = recommend_inventory_constrained_pick_package(
        target_value=starting_absolute_residual,
        available_picks=available_picks,
        outright_pick_hierarchy=outright_pick_hierarchy,
        max_total_picks=max_total_picks,
        balance_tolerance=balance_tolerance,
    )

    if not pick_recommendation["pick_package_available"]:
        return {
            **empty_result,
            "reason": pick_recommendation["reason"],
            "pick_sending_team": pick_sending_team,
            "pick_receiving_team": pick_receiving_team,
            "available_pick_counts": available_pick_counts,
        }

    selected_pick_assets = pick_recommendation[
        "selected_pick_assets"
    ].copy()
    package_value = float(pick_recommendation["package_value"])
    proposed_final_residual = (
        starting_residual + adjustment_sign * package_value
    )

    if abs(proposed_final_residual) >= starting_absolute_residual:
        return {
            **empty_result,
            "reason": (
                "The available pick package does not improve the trade balance."
            ),
            "pick_sending_team": pick_sending_team,
            "pick_receiving_team": pick_receiving_team,
            "available_pick_counts": available_pick_counts,
        }

    tier_to_net_column = dict(
        zip(outright_pick_hierarchy, outright_pick_net_columns)
    )
    tier_counts = (
        selected_pick_assets["tier"]
        .value_counts()
        .reindex(outright_pick_hierarchy, fill_value=0)
        .astype(int)
    )
    team_a_draft_counts = empty_draft_counts.copy()
    for tier, count in tier_counts.items():
        team_a_draft_counts[tier_to_net_column[tier]] = (
            team_a_net_count_sign * int(count)
        )
    team_b_draft_counts = {
        column: -count for column, count in team_a_draft_counts.items()
    }

    recommended_pick_counts = pick_recommendation["recommendation"].copy()
    recommended_pick_counts["sending_team"] = pick_sending_team
    recommended_pick_counts["receiving_team"] = pick_receiving_team

    return {
        "pick_top_up_applied": True,
        "reason": "Owned pick package improves the remaining residual.",
        "pick_sending_team": pick_sending_team,
        "pick_receiving_team": pick_receiving_team,
        "pick_direction": pick_direction,
        "recommended_pick_counts": recommended_pick_counts,
        "selected_pick_assets": selected_pick_assets,
        "available_pick_counts": available_pick_counts,
        "team_a_draft_counts": team_a_draft_counts,
        "team_b_draft_counts": team_b_draft_counts,
        "starting_residual": starting_residual,
        "pick_adjustment": adjustment_sign * package_value,
        "final_residual": proposed_final_residual,
        "absolute_final_residual": abs(proposed_final_residual),
        "within_balance_tolerance": (
            abs(proposed_final_residual) <= balance_tolerance
        ),
    }


def reserve_selected_picks(
    pick_inventory: pd.DataFrame,
    selected_pick_assets: pd.DataFrame,
    inplace: bool = False,
) -> pd.DataFrame:
    """Mark selected assets unavailable for future recommendations."""
    result = pick_inventory if inplace else pick_inventory.copy()

    if selected_pick_assets is None or selected_pick_assets.empty:
        return result

    if "pick_id" not in selected_pick_assets.columns:
        raise KeyError("selected_pick_assets is missing 'pick_id'.")

    selected_ids = set(selected_pick_assets["pick_id"].dropna())
    result.loc[
        result["pick_id"].isin(selected_ids),
        "available_for_trade",
    ] = False

    return result



def generate_hybrid_trade_options(
    base_team_a_residual: float,
    roster_data: pd.DataFrame,
    pick_inventory: pd.DataFrame,
    team_a_name: str,
    team_b_name: str,
    locked_team_a_player: str | Sequence[str],
    locked_team_b_player: str | Sequence[str],
    player_value_column: str,
    additional_player_slot_cost: float,
    pick_value_table: pd.DataFrame,
    outright_pick_hierarchy: Sequence[str],
    outright_pick_net_columns: Sequence[str],
    max_additional_players_per_team: int = 2,
    max_total_additional_players: int = 3,
    max_total_picks: int = 5,
    balance_tolerance: float = 0.05,
    excluded_pick_ids: Sequence[str] | None = None,
    team_column: str = "team",
    player_name_column: str = "candidate_player",
    roster_slot_column: str = "roster_slot",
) -> pd.DataFrame:
    """Enumerate player, pick, and hybrid slot-adjusted recommendations."""
    initial_residual = float(base_team_a_residual)

    locked_team_a_rows = get_roster_players(
        roster_data,
        locked_team_a_player,
        player_name_column=player_name_column,
        allow_empty=True,
    )
    locked_team_b_rows = get_roster_players(
        roster_data,
        locked_team_b_player,
        player_name_column=player_name_column,
        allow_empty=True,
    )
    base_team_a_package_value = calculate_slot_adjusted_package_value(
        pd.to_numeric(
            locked_team_a_rows[player_value_column], errors="coerce"
        ),
        additional_player_slot_cost=additional_player_slot_cost,
    )
    base_team_b_package_value = calculate_slot_adjusted_package_value(
        pd.to_numeric(
            locked_team_b_rows[player_value_column], errors="coerce"
        ),
        additional_player_slot_cost=additional_player_slot_cost,
    )
    base_player_match_profile = calculate_player_match_profile(
        team_a_players=locked_team_a_rows,
        team_b_players=locked_team_b_rows,
        player_value_column=player_value_column,
        player_name_column=player_name_column,
    )

    player_options = generate_player_addition_alternatives(
        roster_data=roster_data,
        team_a_name=team_a_name,
        team_b_name=team_b_name,
        locked_team_a_player=locked_team_a_player,
        locked_team_b_player=locked_team_b_player,
        base_team_a_residual=initial_residual,
        player_value_column=player_value_column,
        additional_player_slot_cost=additional_player_slot_cost,
        max_additional_players_per_team=max_additional_players_per_team,
        max_total_additional_players=max_total_additional_players,
        balance_tolerance=balance_tolerance,
        team_column=team_column,
        player_name_column=player_name_column,
        roster_slot_column=roster_slot_column,
        require_directional_improvement=False,
    )

    no_player_option = pd.DataFrame(
        [
            {
                "team_a_additional_players_sent": [],
                "team_b_additional_players_sent": [],
                "team_a_additional_raw_value_sent": 0.0,
                "team_b_additional_raw_value_sent": 0.0,
                "team_a_additional_value_sent": 0.0,
                "team_b_additional_value_sent": 0.0,
                "team_a_full_player_package_value_sent": (
                    base_team_a_package_value
                ),
                "team_b_full_player_package_value_sent": (
                    base_team_b_package_value
                ),
                "base_team_a_player_package_value_sent": (
                    base_team_a_package_value
                ),
                "base_team_b_player_package_value_sent": (
                    base_team_b_package_value
                ),
                **base_player_match_profile,
                "base_player_match_cost": float(
                    base_player_match_profile["player_match_cost"]
                ),
                "player_match_improvement": 0.0,
                "candidate_player_differential": (
                    base_team_b_package_value - base_team_a_package_value
                ),
                "net_team_a_player_adjustment": 0.0,
                "team_a_additional_player_count": 0,
                "team_b_additional_player_count": 0,
                "total_additional_players": 0,
                "base_team_a_residual": initial_residual,
                "residual_after_players": initial_residual,
                "absolute_residual_after_players": abs(initial_residual),
                "residual_improvement": 0.0,
                "both_teams_add_players": False,
                "crosses_balance_point": False,
                "within_balance_tolerance": (
                    abs(initial_residual) <= balance_tolerance
                ),
            }
        ]
    )

    player_options = pd.concat(
        [no_player_option, player_options],
        ignore_index=True,
        sort=False,
    )

    hybrid_records: list[dict[str, Any]] = []
    for player_option in player_options.to_dict(orient="records"):
        residual_after_players = float(
            player_option["residual_after_players"]
        )

        pick_options = enumerate_inventory_constrained_pick_packages(
            starting_residual=residual_after_players,
            team_a_name=team_a_name,
            team_b_name=team_b_name,
            pick_inventory=pick_inventory,
            pick_value_table=pick_value_table,
            outright_pick_hierarchy=outright_pick_hierarchy,
            outright_pick_net_columns=outright_pick_net_columns,
            max_total_picks=max_total_picks,
            balance_tolerance=balance_tolerance,
            excluded_pick_ids=excluded_pick_ids,
            include_zero_pick_option=True,
        )
        if pick_options.empty:
            continue

        for pick_option in pick_options.to_dict(orient="records"):
            total_players = int(player_option["total_additional_players"])
            total_picks = int(pick_option["total_picks"])
            final_residual = float(pick_option["final_residual"])

            if total_players > 0 and total_picks > 0:
                option_type = "players_and_picks"
            elif total_players > 0:
                option_type = "players_only"
            elif total_picks > 0:
                option_type = "picks_only"
            else:
                option_type = "no_adjustment"

            hybrid_records.append(
                {
                    "option_type": option_type,
                    "team_a_additional_players_sent": player_option[
                        "team_a_additional_players_sent"
                    ],
                    "team_b_additional_players_sent": player_option[
                        "team_b_additional_players_sent"
                    ],
                    "team_a_additional_raw_value_sent": float(
                        player_option.get("team_a_additional_raw_value_sent", 0)
                    ),
                    "team_b_additional_raw_value_sent": float(
                        player_option.get("team_b_additional_raw_value_sent", 0)
                    ),
                    "team_a_additional_value_sent": float(
                        player_option["team_a_additional_value_sent"]
                    ),
                    "team_b_additional_value_sent": float(
                        player_option["team_b_additional_value_sent"]
                    ),
                    "team_a_full_player_package_value_sent": float(
                        player_option[
                            "team_a_full_player_package_value_sent"
                        ]
                    ),
                    "team_b_full_player_package_value_sent": float(
                        player_option[
                            "team_b_full_player_package_value_sent"
                        ]
                    ),
                    "player_match_cost": float(
                        player_option["player_match_cost"]
                    ),
                    "unmatched_player_value": float(
                        player_option["unmatched_player_value"]
                    ),
                    "maximum_player_match_gap": float(
                        player_option["maximum_player_match_gap"]
                    ),
                    "matched_player_pair_count": int(
                        player_option["matched_player_pair_count"]
                    ),
                    "unmatched_player_pair_count": int(
                        player_option["unmatched_player_pair_count"]
                    ),
                    "player_match_pairs": player_option[
                        "player_match_pairs"
                    ],
                    "base_player_match_cost": float(
                        player_option["base_player_match_cost"]
                    ),
                    "player_match_improvement": float(
                        player_option["player_match_improvement"]
                    ),
                    "net_team_a_player_adjustment": float(
                        player_option["net_team_a_player_adjustment"]
                    ),
                    "team_a_additional_player_count": int(
                        player_option["team_a_additional_player_count"]
                    ),
                    "team_b_additional_player_count": int(
                        player_option["team_b_additional_player_count"]
                    ),
                    "total_additional_players": total_players,
                    "residual_after_players": residual_after_players,
                    "pick_sending_team": pick_option["pick_sending_team"],
                    "pick_receiving_team": pick_option["pick_receiving_team"],
                    "selected_pick_ids": pick_option["selected_pick_ids"],
                    "selected_pick_assets": pick_option[
                        "selected_pick_assets"
                    ],
                    "team_a_draft_counts": pick_option[
                        "team_a_draft_counts"
                    ],
                    "team_b_draft_counts": pick_option[
                        "team_b_draft_counts"
                    ],
                    "pick_package_value": float(
                        pick_option["pick_package_value"]
                    ),
                    "pick_adjustment": float(
                        pick_option["pick_adjustment"]
                    ),
                    "total_picks": total_picks,
                    "total_adjustment_assets": total_players + total_picks,
                    "initial_residual": initial_residual,
                    "final_residual": final_residual,
                    "absolute_final_residual": abs(final_residual),
                    "total_residual_improvement": (
                        abs(initial_residual) - abs(final_residual)
                    ),
                    "within_balance_tolerance": (
                        abs(final_residual) <= balance_tolerance
                    ),
                    **{
                        f"{tier}_count": int(
                            pick_option.get(f"{tier}_count", 0)
                        )
                        for tier in outright_pick_hierarchy
                    },
                }
            )

    hybrid_options = pd.DataFrame(hybrid_records)
    if hybrid_options.empty:
        return hybrid_options

    return hybrid_options.loc[
        hybrid_options["option_type"].eq("no_adjustment")
        | hybrid_options["total_residual_improvement"].gt(0)
    ].reset_index(drop=True)



def _decode_catalogue_list(value: Any) -> list[Any]:
    """Decode a JSON-list field stored in the package catalogue."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, np.ndarray, pd.Series)):
        return list(value)
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("Catalogue list fields must contain JSON arrays.")
    return parsed


def _calculate_player_match_profile_from_lists(
    team_a_names: Sequence[str],
    team_a_values: Sequence[float],
    team_b_names: Sequence[str],
    team_b_values: Sequence[float],
) -> dict[str, Any]:
    """List-based equivalent of calculate_player_match_profile."""
    team_a = sorted(
        zip([str(value) for value in team_a_names], map(float, team_a_values)),
        key=lambda item: (-item[1], item[0]),
    )
    team_b = sorted(
        zip([str(value) for value in team_b_names], map(float, team_b_values)),
        key=lambda item: (-item[1], item[0]),
    )

    pair_records: list[dict[str, Any]] = []
    for pair_index in range(max(len(team_a), len(team_b))):
        team_a_present = pair_index < len(team_a)
        team_b_present = pair_index < len(team_b)
        team_a_name = team_a[pair_index][0] if team_a_present else None
        team_b_name = team_b[pair_index][0] if team_b_present else None
        team_a_value = team_a[pair_index][1] if team_a_present else 0.0
        team_b_value = team_b[pair_index][1] if team_b_present else 0.0
        absolute_gap = abs(team_a_value - team_b_value)
        is_unmatched = not (team_a_present and team_b_present)
        pair_records.append(
            {
                "pair_rank": pair_index + 1,
                "team_a_player": team_a_name,
                "team_a_player_value": float(team_a_value),
                "team_b_player": team_b_name,
                "team_b_player_value": float(team_b_value),
                "absolute_player_value_gap": float(absolute_gap),
                "is_unmatched_pair": is_unmatched,
            }
        )

    return {
        "player_match_cost": float(
            sum(record["absolute_player_value_gap"] for record in pair_records)
        ),
        "unmatched_player_value": float(
            sum(
                record["absolute_player_value_gap"]
                for record in pair_records
                if record["is_unmatched_pair"]
            )
        ),
        "maximum_player_match_gap": float(
            max(
                (
                    record["absolute_player_value_gap"]
                    for record in pair_records
                ),
                default=0.0,
            )
        ),
        "matched_player_pair_count": int(
            sum(not record["is_unmatched_pair"] for record in pair_records)
        ),
        "unmatched_player_pair_count": int(
            sum(record["is_unmatched_pair"] for record in pair_records)
        ),
        "player_match_pairs": pair_records,
    }


def generate_hybrid_trade_options_from_catalogue(
    base_team_a_residual: float,
    roster_data: pd.DataFrame,
    pick_inventory: pd.DataFrame,
    hybrid_package_catalogue: pd.DataFrame,
    team_a_name: str,
    team_b_name: str,
    locked_team_a_player: str | Sequence[str],
    locked_team_b_player: str | Sequence[str],
    player_value_column: str,
    additional_player_slot_cost: float,
    outright_pick_hierarchy: Sequence[str],
    outright_pick_net_columns: Sequence[str],
    max_additional_players_per_team: int = 2,
    max_total_additional_players: int = 3,
    max_total_picks: int = 5,
    balance_tolerance: float = 0.05,
    excluded_pick_ids: Sequence[str] | None = None,
    ranking_mode: str = "closest_first",
    max_returned_options: int = 250,
    team_column: str = "team",
    player_name_column: str = "candidate_player",
) -> pd.DataFrame:
    """Enumerate the same hybrid options as the original runtime engine.

    Player and pick combinations come from the precomputed catalogue, but the
    locked-plus-additional player values, residuals, improvement filters, pick
    direction, and final option ranking inputs are calculated with the same
    formulas used by :func:`generate_hybrid_trade_options`.

    No nearest-value shortlist or approximate completion step is used. Every
    valid catalogue completion is evaluated with the original formulas. After
    exact scoring, only the highest-ranked options needed for display are kept
    in memory; this does not affect the selected recommendation.
    """
    required_catalogue_columns = {
        "team",
        "player_package_id",
        "pick_package_id",
        "player_names_json",
        "player_count",
        "pick_ids_json",
        "pick_count",
    }
    missing_catalogue = sorted(
        required_catalogue_columns - set(hybrid_package_catalogue.columns)
    )
    if missing_catalogue:
        raise KeyError(
            f"Hybrid package catalogue is missing: {missing_catalogue}"
        )
    valid_ranking_modes = {
        "closest_first",
        "simplest_within_tolerance",
        "player_matching",
    }
    if ranking_mode not in valid_ranking_modes:
        raise ValueError(
            f"ranking_mode must be one of {sorted(valid_ranking_modes)}"
        )
    if max_returned_options < 1:
        raise ValueError("max_returned_options must be at least 1.")

    initial_residual = float(base_team_a_residual)
    locked_team_a_players = _normalize_player_names(
        locked_team_a_player,
        argument_name="locked_team_a_player",
        allow_empty=True,
    )
    locked_team_b_players = _normalize_player_names(
        locked_team_b_player,
        argument_name="locked_team_b_player",
        allow_empty=True,
    )
    locked_team_a_rows = get_roster_players(
        roster_data,
        locked_team_a_players,
        player_name_column=player_name_column,
        allow_empty=True,
    )
    locked_team_b_rows = get_roster_players(
        roster_data,
        locked_team_b_players,
        player_name_column=player_name_column,
        allow_empty=True,
    )

    locked_team_a_values = pd.to_numeric(
        locked_team_a_rows[player_value_column], errors="coerce"
    ).astype("float64").tolist()
    locked_team_b_values = pd.to_numeric(
        locked_team_b_rows[player_value_column], errors="coerce"
    ).astype("float64").tolist()
    if any(
        not np.isfinite(value)
        for value in locked_team_a_values + locked_team_b_values
    ):
        raise ValueError("Locked player packages contain invalid player values.")

    base_team_a_package_value = calculate_slot_adjusted_package_value(
        locked_team_a_values,
        additional_player_slot_cost=additional_player_slot_cost,
    )
    base_team_b_package_value = calculate_slot_adjusted_package_value(
        locked_team_b_values,
        additional_player_slot_cost=additional_player_slot_cost,
    )
    base_player_differential = (
        base_team_b_package_value - base_team_a_package_value
    )
    non_player_residual_component = initial_residual - base_player_differential
    base_player_match_profile = _calculate_player_match_profile_from_lists(
        locked_team_a_players,
        locked_team_a_values,
        locked_team_b_players,
        locked_team_b_values,
    )

    def current_candidate_lookup(
        team_name: str,
        locked_names: Sequence[str],
    ) -> tuple[dict[str, float], dict[str, int]]:
        candidates = roster_data.loc[
            roster_data[team_column].astype(str).eq(str(team_name))
            & ~roster_data[player_name_column].astype(str).isin(
                [str(value) for value in locked_names]
            ),
            [player_name_column, player_value_column],
        ].copy()
        candidates[player_value_column] = pd.to_numeric(
            candidates[player_value_column], errors="coerce"
        )
        candidates = candidates.loc[
            candidates[player_value_column].gt(0)
        ].reset_index(drop=True)
        names = candidates[player_name_column].astype(str).tolist()
        return (
            dict(zip(names, candidates[player_value_column].astype(float))),
            {name: index for index, name in enumerate(names)},
        )

    def prepare_player_packages(
        team_name: str,
        locked_names: Sequence[str],
    ) -> list[dict[str, Any]]:
        value_lookup, order_lookup = current_candidate_lookup(
            team_name,
            locked_names,
        )
        rows = (
            hybrid_package_catalogue.loc[
                hybrid_package_catalogue["team"].astype(str).eq(str(team_name))
                & pd.to_numeric(
                    hybrid_package_catalogue["player_count"], errors="coerce"
                ).le(max_additional_players_per_team)
            ]
            .drop_duplicates("player_package_id")
            .copy()
        )
        prepared: list[dict[str, Any]] = []
        for row in rows.to_dict(orient="records"):
            names = [
                str(value)
                for value in _decode_catalogue_list(
                    row["player_names_json"]
                )
            ]
            if any(name not in value_lookup for name in names):
                continue
            ordered_names = sorted(names, key=order_lookup.__getitem__)
            values = [float(value_lookup[name]) for name in ordered_names]
            order_key = tuple(order_lookup[name] for name in ordered_names)
            prepared.append(
                {
                    "package_id": str(row["player_package_id"]),
                    "names": ordered_names,
                    "values": values,
                    "count": len(ordered_names),
                    "raw_value": float(sum(values)),
                    "order_key": order_key,
                }
            )
        prepared.sort(key=lambda item: (item["count"], item["order_key"]))
        return prepared

    excluded_ids = {str(value) for value in (excluded_pick_ids or [])}

    def prepare_pick_packages(team_name: str) -> list[dict[str, Any]]:
        available_picks, _ = get_available_pick_inventory(
            pick_inventory=pick_inventory,
            team_name=team_name,
            outright_pick_hierarchy=outright_pick_hierarchy,
            excluded_pick_ids=excluded_ids,
        )
        available_picks = available_picks.copy()
        available_picks["pick_id"] = available_picks["pick_id"].astype(str)
        available_lookup = available_picks.set_index("pick_id", drop=False)
        order_lookup = {
            pick_id: index
            for index, pick_id in enumerate(
                available_picks["pick_id"].astype(str).tolist()
            )
        }

        rows = (
            hybrid_package_catalogue.loc[
                hybrid_package_catalogue["team"].astype(str).eq(str(team_name))
                & pd.to_numeric(
                    hybrid_package_catalogue["player_count"], errors="coerce"
                ).eq(0)
                & pd.to_numeric(
                    hybrid_package_catalogue["pick_count"], errors="coerce"
                ).le(max_total_picks)
            ]
            .drop_duplicates("pick_package_id")
            .copy()
        )
        prepared: list[dict[str, Any]] = []
        for row in rows.to_dict(orient="records"):
            pick_ids = [
                str(value)
                for value in _decode_catalogue_list(row["pick_ids_json"])
            ]
            if any(pick_id not in available_lookup.index for pick_id in pick_ids):
                continue
            ordered_ids = sorted(pick_ids, key=order_lookup.__getitem__)
            selected_assets = (
                available_lookup.loc[ordered_ids].copy().reset_index(drop=True)
                if ordered_ids
                else available_picks.iloc[0:0].copy()
            )
            tier_counts = (
                selected_assets["tier"]
                .value_counts()
                .reindex(outright_pick_hierarchy, fill_value=0)
                .astype(int)
                .to_dict()
            )
            prepared.append(
                {
                    "package_id": str(row["pick_package_id"]),
                    "pick_ids": ordered_ids,
                    "selected_pick_assets": selected_assets,
                    "count": len(ordered_ids),
                    "value": float(
                        pd.to_numeric(
                            selected_assets.get(
                                "adjusted_value", pd.Series(dtype=float)
                            ),
                            errors="coerce",
                        ).sum()
                    ),
                    "tier_counts": tier_counts,
                    "order_key": tuple(
                        order_lookup[pick_id] for pick_id in ordered_ids
                    ),
                }
            )
        prepared.sort(key=lambda item: (item["count"], item["order_key"]))
        return prepared

    def prepare_full_player_packages(
        packages: list[dict[str, Any]],
        locked_names: list[str],
        locked_values: list[float],
        base_value: float,
    ) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for package in packages:
            full_names = locked_names + package["names"]
            full_values = locked_values + package["values"]
            full_value = calculate_slot_adjusted_package_value(
                full_values,
                additional_player_slot_cost=additional_player_slot_cost,
            )
            prepared.append(
                {
                    **package,
                    "full_names": full_names,
                    "full_values": full_values,
                    "full_value": float(full_value),
                    "incremental_value": float(full_value - base_value),
                }
            )
        return prepared

    team_a_player_packages = prepare_full_player_packages(
        prepare_player_packages(team_a_name, locked_team_a_players),
        locked_team_a_players,
        locked_team_a_values,
        base_team_a_package_value,
    )
    team_b_player_packages = prepare_full_player_packages(
        prepare_player_packages(team_b_name, locked_team_b_players),
        locked_team_b_players,
        locked_team_b_values,
        base_team_b_package_value,
    )
    team_a_pick_packages = prepare_pick_packages(team_a_name)
    team_b_pick_packages = prepare_pick_packages(team_b_name)

    empty_draft_counts = create_empty_draft_counts(
        outright_pick_net_columns
    )
    tier_to_net_column = dict(
        zip(outright_pick_hierarchy, outright_pick_net_columns)
    )

    retained_candidates: list[tuple[tuple[Any, ...], int, dict[str, Any]]] = []
    enumeration_order = 0
    trim_threshold = max(1000, max_returned_options * 4)

    def option_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
        within = bool(record["within_balance_tolerance"])
        if ranking_mode == "closest_first":
            return (
                not within,
                float(record["absolute_final_residual"]),
                int(record["total_adjustment_assets"]),
                int(record["total_additional_players"]),
                int(record["total_picks"]),
                -float(record["total_residual_improvement"]),
            )
        if ranking_mode == "simplest_within_tolerance":
            if within:
                return (
                    0,
                    int(record["total_additional_players"]),
                    int(record["total_picks"]),
                    float(record["absolute_final_residual"]),
                    -float(record["total_residual_improvement"]),
                )
            return (
                1,
                float(record["absolute_final_residual"]),
                int(record["total_adjustment_assets"]),
                int(record["total_additional_players"]),
                int(record["total_picks"]),
            )
        if within:
            return (
                0,
                float(record["player_match_cost"]),
                float(record["unmatched_player_value"]),
                float(record["maximum_player_match_gap"]),
                float(record["pick_package_value"]),
                int(record["total_adjustment_assets"]),
                int(record["total_additional_players"]),
                float(record["absolute_final_residual"]),
            )
        return (
            1,
            float(record["absolute_final_residual"]),
            float(record["player_match_cost"]),
            float(record["unmatched_player_value"]),
            float(record["maximum_player_match_gap"]),
            int(record["total_adjustment_assets"]),
        )

    def retain_candidate(record: dict[str, Any]) -> None:
        nonlocal retained_candidates, enumeration_order
        if (
            record["option_type"] != "no_adjustment"
            and float(record["total_residual_improvement"]) <= 0
        ):
            return
        record["_catalogue_enumeration_order"] = enumeration_order
        retained_candidates.append(
            (option_sort_key(record), enumeration_order, record)
        )
        enumeration_order += 1
        if len(retained_candidates) >= trim_threshold:
            retained_candidates = sorted(
                retained_candidates,
                key=lambda item: (item[0], item[1]),
            )[:max_returned_options]

    for team_a_package in team_a_player_packages:
        for team_b_package in team_b_player_packages:
            total_players = int(
                team_a_package["count"] + team_b_package["count"]
            )
            if total_players > max_total_additional_players:
                continue

            candidate_player_differential = (
                team_b_package["full_value"] - team_a_package["full_value"]
            )
            net_team_a_player_adjustment = (
                candidate_player_differential - base_player_differential
            )
            residual_after_players = (
                candidate_player_differential
                + non_player_residual_component
            )
            player_match_profile = _calculate_player_match_profile_from_lists(
                team_a_package["full_names"],
                team_a_package["full_values"],
                team_b_package["full_names"],
                team_b_package["full_values"],
            )

            if residual_after_players < 0:
                pick_packages = team_b_pick_packages
                pick_sending_team = str(team_b_name)
                pick_receiving_team = str(team_a_name)
                adjustment_sign = 1.0
                team_a_count_sign = 1
            elif residual_after_players > 0:
                pick_packages = team_a_pick_packages
                pick_sending_team = str(team_a_name)
                pick_receiving_team = str(team_b_name)
                adjustment_sign = -1.0
                team_a_count_sign = -1
            else:
                pick_packages = [
                    package for package in team_a_pick_packages
                    if package["count"] == 0
                ]
                pick_sending_team = None
                pick_receiving_team = None
                adjustment_sign = 0.0
                team_a_count_sign = 0

            for pick_package in pick_packages:
                total_picks = int(pick_package["count"])
                package_value = float(pick_package["value"])
                pick_adjustment = adjustment_sign * package_value
                final_residual = residual_after_players + pick_adjustment
                pick_stage_improvement = (
                    abs(residual_after_players) - abs(final_residual)
                )
                if total_picks > 0 and pick_stage_improvement <= 0:
                    continue

                team_a_draft_counts = empty_draft_counts.copy()
                for tier, count in pick_package["tier_counts"].items():
                    team_a_draft_counts[tier_to_net_column[tier]] = (
                        team_a_count_sign * int(count)
                    )
                team_b_draft_counts = {
                    column: -count
                    for column, count in team_a_draft_counts.items()
                }

                if total_players > 0 and total_picks > 0:
                    option_type = "players_and_picks"
                elif total_players > 0:
                    option_type = "players_only"
                elif total_picks > 0:
                    option_type = "picks_only"
                else:
                    option_type = "no_adjustment"

                record = {
                        "option_type": option_type,
                        "team_a_additional_players_sent": list(
                            team_a_package["names"]
                        ),
                        "team_b_additional_players_sent": list(
                            team_b_package["names"]
                        ),
                        "team_a_additional_raw_value_sent": float(
                            team_a_package["raw_value"]
                        ),
                        "team_b_additional_raw_value_sent": float(
                            team_b_package["raw_value"]
                        ),
                        "team_a_additional_value_sent": float(
                            team_a_package["incremental_value"]
                        ),
                        "team_b_additional_value_sent": float(
                            team_b_package["incremental_value"]
                        ),
                        "team_a_full_player_package_value_sent": float(
                            team_a_package["full_value"]
                        ),
                        "team_b_full_player_package_value_sent": float(
                            team_b_package["full_value"]
                        ),
                        **player_match_profile,
                        "base_player_match_cost": float(
                            base_player_match_profile["player_match_cost"]
                        ),
                        "player_match_improvement": float(
                            base_player_match_profile["player_match_cost"]
                            - player_match_profile["player_match_cost"]
                        ),
                        "net_team_a_player_adjustment": float(
                            net_team_a_player_adjustment
                        ),
                        "team_a_additional_player_count": int(
                            team_a_package["count"]
                        ),
                        "team_b_additional_player_count": int(
                            team_b_package["count"]
                        ),
                        "total_additional_players": total_players,
                        "residual_after_players": float(
                            residual_after_players
                        ),
                        "pick_sending_team": (
                            pick_sending_team if total_picks > 0 else None
                        ),
                        "pick_receiving_team": (
                            pick_receiving_team if total_picks > 0 else None
                        ),
                        "selected_pick_ids": list(
                            pick_package["pick_ids"]
                        ),
                        "selected_pick_assets": None,
                        "team_a_draft_counts": team_a_draft_counts,
                        "team_b_draft_counts": team_b_draft_counts,
                        "pick_package_value": package_value,
                        "pick_adjustment": pick_adjustment,
                        "total_picks": total_picks,
                        "total_adjustment_assets": (
                            total_players + total_picks
                        ),
                        "initial_residual": initial_residual,
                        "final_residual": float(final_residual),
                        "absolute_final_residual": abs(final_residual),
                        "total_residual_improvement": (
                            abs(initial_residual) - abs(final_residual)
                        ),
                        "within_balance_tolerance": (
                            abs(final_residual) <= balance_tolerance
                        ),
                        **{
                            f"{tier}_count": int(
                                pick_package["tier_counts"].get(tier, 0)
                            )
                            for tier in outright_pick_hierarchy
                        },
                    }
                retain_candidate(record)

    if not retained_candidates:
        return pd.DataFrame()
    retained_candidates = sorted(
        retained_candidates,
        key=lambda item: (item[0], item[1]),
    )[:max_returned_options]
    return pd.DataFrame(
        [item[2] for item in retained_candidates]
    ).reset_index(drop=True)

def rank_hybrid_trade_options(
    options: pd.DataFrame,
    ranking_mode: str = "closest_first",
) -> pd.DataFrame:
    """Rank hybrid recommendations and add a one-based recommendation rank."""
    if options.empty:
        return options.copy()

    valid_modes = {
        "closest_first",
        "simplest_within_tolerance",
        "player_matching",
    }
    if ranking_mode not in valid_modes:
        raise ValueError(
            f"ranking_mode must be one of {sorted(valid_modes)}"
        )

    required_columns = {
        "within_balance_tolerance",
        "absolute_final_residual",
        "total_adjustment_assets",
        "total_additional_players",
        "total_picks",
        "total_residual_improvement",
        "pick_package_value",
    }
    if ranking_mode == "player_matching":
        required_columns |= {
            "player_match_cost",
            "unmatched_player_value",
            "maximum_player_match_gap",
        }

    missing_columns = sorted(required_columns - set(options.columns))
    if missing_columns:
        raise KeyError(f"Hybrid options are missing columns: {missing_columns}")

    tie_break_columns = (
        ["_catalogue_enumeration_order"]
        if "_catalogue_enumeration_order" in options.columns
        else []
    )

    if ranking_mode == "closest_first":
        ranked = options.sort_values(
            [
                "within_balance_tolerance",
                "absolute_final_residual",
                "total_adjustment_assets",
                "total_additional_players",
                "total_picks",
                "total_residual_improvement",
                *tie_break_columns,
            ],
            ascending=[
                False, True, True, True, True, False,
                *([True] * len(tie_break_columns)),
            ],
        )
    elif ranking_mode == "simplest_within_tolerance":
        within = options.loc[
            options["within_balance_tolerance"]
        ].sort_values(
            [
                "total_additional_players",
                "total_picks",
                "absolute_final_residual",
                "total_residual_improvement",
                *tie_break_columns,
            ],
            ascending=[
                True, True, True, False,
                *([True] * len(tie_break_columns)),
            ],
        )
        outside = options.loc[
            ~options["within_balance_tolerance"]
        ].sort_values(
            [
                "absolute_final_residual",
                "total_adjustment_assets",
                "total_additional_players",
                "total_picks",
                *tie_break_columns,
            ],
            ascending=[
                True, True, True, True,
                *([True] * len(tie_break_columns)),
            ],
        )
        ranked = pd.concat([within, outside], ignore_index=True)
    else:
        # Picks do not affect the matching metrics. Within the arithmetic
        # tolerance, first choose the player package that most closely matches
        # individual production rank-for-rank, then use the least pick value
        # and fewest assets necessary to finish the trade.
        within = options.loc[
            options["within_balance_tolerance"]
        ].sort_values(
            [
                "player_match_cost",
                "unmatched_player_value",
                "maximum_player_match_gap",
                "pick_package_value",
                "total_adjustment_assets",
                "total_additional_players",
                "absolute_final_residual",
                *tie_break_columns,
            ],
            ascending=[
                True, True, True, True, True, True, True,
                *([True] * len(tie_break_columns)),
            ],
        )
        outside = options.loc[
            ~options["within_balance_tolerance"]
        ].sort_values(
            [
                "absolute_final_residual",
                "player_match_cost",
                "unmatched_player_value",
                "maximum_player_match_gap",
                "total_adjustment_assets",
                *tie_break_columns,
            ],
            ascending=[
                True, True, True, True, True,
                *([True] * len(tie_break_columns)),
            ],
        )
        ranked = pd.concat([within, outside], ignore_index=True)

    ranked = ranked.reset_index(drop=True)
    ranked.insert(
        0,
        "recommendation_rank",
        np.arange(1, len(ranked) + 1),
    )
    return ranked



def recommend_trade_adjustments(
    base_team_a_residual: float,
    roster_data: pd.DataFrame,
    pick_inventory: pd.DataFrame,
    team_a_name: str,
    team_b_name: str,
    locked_team_a_player: str | Sequence[str],
    locked_team_b_player: str | Sequence[str],
    player_value_column: str,
    additional_player_slot_cost: float,
    pick_value_table: pd.DataFrame,
    outright_pick_hierarchy: Sequence[str],
    outright_pick_net_columns: Sequence[str],
    priority_mode: str = "best_fit",
    max_additional_players_per_team: int = 2,
    max_total_additional_players: int = 3,
    max_total_picks: int = 5,
    balance_tolerance: float = 0.05,
    excluded_pick_ids: Sequence[str] | None = None,
    hybrid_ranking_mode: str = "closest_first",
    hybrid_package_catalogue: pd.DataFrame | None = None,
    team_column: str = "team",
    player_name_column: str = "candidate_player",
    roster_slot_column: str = "roster_slot",
) -> dict[str, Any]:
    """Recommend slot-adjusted trade additions using the selected mode."""
    valid_priority_modes = {
        "players_first",
        "picks_first",
        "best_fit",
        "player_matching",
    }
    if priority_mode not in valid_priority_modes:
        raise ValueError(
            f"priority_mode must be one of {sorted(valid_priority_modes)}"
        )

    locked_team_a_players = _normalize_player_names(
        locked_team_a_player,
        argument_name="locked_team_a_player",
        allow_empty=True,
    )
    locked_team_b_players = _normalize_player_names(
        locked_team_b_player,
        argument_name="locked_team_b_player",
        allow_empty=True,
    )

    team_a_player_rows = get_roster_players(
        roster_data,
        locked_team_a_players,
        player_name_column=player_name_column,
        allow_empty=True,
    )
    team_b_player_rows = get_roster_players(
        roster_data,
        locked_team_b_players,
        player_name_column=player_name_column,
        allow_empty=True,
    )

    invalid_team_a = team_a_player_rows.loc[
        ~team_a_player_rows[team_column].astype(str).eq(str(team_a_name)),
        player_name_column,
    ].astype(str).tolist()
    invalid_team_b = team_b_player_rows.loc[
        ~team_b_player_rows[team_column].astype(str).eq(str(team_b_name)),
        player_name_column,
    ].astype(str).tolist()
    if invalid_team_a:
        raise ValueError(
            f"These players do not belong to {team_a_name}: {invalid_team_a}"
        )
    if invalid_team_b:
        raise ValueError(
            f"These players do not belong to {team_b_name}: {invalid_team_b}"
        )

    initial_residual = float(base_team_a_residual)

    if priority_mode in {"best_fit", "player_matching"}:
        if hybrid_package_catalogue is None:
            hybrid_options = generate_hybrid_trade_options(
                base_team_a_residual=initial_residual,
                roster_data=roster_data,
                pick_inventory=pick_inventory,
                team_a_name=team_a_name,
                team_b_name=team_b_name,
                locked_team_a_player=locked_team_a_players,
                locked_team_b_player=locked_team_b_players,
                player_value_column=player_value_column,
                additional_player_slot_cost=additional_player_slot_cost,
                pick_value_table=pick_value_table,
                outright_pick_hierarchy=outright_pick_hierarchy,
                outright_pick_net_columns=outright_pick_net_columns,
                max_additional_players_per_team=(
                    max_additional_players_per_team
                ),
                max_total_additional_players=max_total_additional_players,
                max_total_picks=max_total_picks,
                balance_tolerance=balance_tolerance,
                excluded_pick_ids=excluded_pick_ids,
                team_column=team_column,
                player_name_column=player_name_column,
                roster_slot_column=roster_slot_column,
            )
        else:
            hybrid_options = generate_hybrid_trade_options_from_catalogue(
                base_team_a_residual=initial_residual,
                roster_data=roster_data,
                pick_inventory=pick_inventory,
                hybrid_package_catalogue=hybrid_package_catalogue,
                team_a_name=team_a_name,
                team_b_name=team_b_name,
                locked_team_a_player=locked_team_a_players,
                locked_team_b_player=locked_team_b_players,
                player_value_column=player_value_column,
                additional_player_slot_cost=additional_player_slot_cost,
                outright_pick_hierarchy=outright_pick_hierarchy,
                outright_pick_net_columns=outright_pick_net_columns,
                max_additional_players_per_team=(
                    max_additional_players_per_team
                ),
                max_total_additional_players=max_total_additional_players,
                max_total_picks=max_total_picks,
                balance_tolerance=balance_tolerance,
                excluded_pick_ids=excluded_pick_ids,
                ranking_mode=(
                    "player_matching"
                    if priority_mode == "player_matching"
                    else hybrid_ranking_mode
                ),
                team_column=team_column,
                player_name_column=player_name_column,
            )
        selected_hybrid_ranking_mode = (
            "player_matching"
            if priority_mode == "player_matching"
            else hybrid_ranking_mode
        )
        ranked_options = rank_hybrid_trade_options(
            hybrid_options,
            ranking_mode=selected_hybrid_ranking_mode,
        )

        if ranked_options.empty:
            selected_option = None
            final_residual = initial_residual
        else:
            selected_option = ranked_options.iloc[0].copy()
            selected_pick_ids = [
                str(value)
                for value in (selected_option.get("selected_pick_ids", []) or [])
            ]
            if selected_pick_ids:
                indexed_inventory = pick_inventory.copy()
                indexed_inventory["pick_id"] = (
                    indexed_inventory["pick_id"].astype(str)
                )
                indexed_inventory = (
                    indexed_inventory.drop_duplicates("pick_id")
                    .set_index("pick_id", drop=False)
                )
                selected_option["selected_pick_assets"] = (
                    indexed_inventory.loc[selected_pick_ids]
                    .copy()
                    .reset_index(drop=True)
                )
            else:
                selected_option["selected_pick_assets"] = (
                    pick_inventory.iloc[0:0].copy()
                )
            final_residual = float(selected_option["final_residual"])

        return {
            "priority_mode": priority_mode,
            "initial_residual": initial_residual,
            "additional_player_slot_cost": float(
                additional_player_slot_cost
            ),
            "locked_team_a_players": locked_team_a_players,
            "locked_team_b_players": locked_team_b_players,
            "hybrid_ranking_mode": selected_hybrid_ranking_mode,
            "hybrid_options": ranked_options,
            "selected_hybrid_option": selected_option,
            "player_result": None,
            "pick_result": None,
            "phase_audit": pd.DataFrame(
                [
                    {
                        "phase": priority_mode,
                        "executed": True,
                        "adjustment_applied": (
                            selected_option is not None
                            and selected_option["option_type"]
                            != "no_adjustment"
                        ),
                        "reason": (
                            (
                                "Ranked balanced options by player-to-player "
                                "production matching before pick value."
                                if priority_mode == "player_matching"
                                else "Ranked all slot-adjusted player, pick, "
                                "and hybrid options."
                            )
                            if selected_option is not None
                            else "No recommendation options were available."
                        ),
                        "residual_before": initial_residual,
                        "adjustment": final_residual - initial_residual,
                        "residual_after": final_residual,
                    }
                ]
            ),
            "final_residual": final_residual,
            "absolute_final_residual": abs(final_residual),
            "within_balance_tolerance": (
                abs(final_residual) <= balance_tolerance
            ),
        }

    current_residual = initial_residual
    player_result: dict[str, Any] | None = None
    pick_result: dict[str, Any] | None = None
    phase_records: list[dict[str, Any]] = []
    phase_order = (
        ["players", "picks"]
        if priority_mode == "players_first"
        else ["picks", "players"]
    )

    for phase_name in phase_order:
        residual_before_phase = current_residual
        if abs(current_residual) <= balance_tolerance:
            phase_records.append(
                {
                    "phase": phase_name,
                    "executed": False,
                    "adjustment_applied": False,
                    "reason": "Trade already reached balance tolerance.",
                    "residual_before": residual_before_phase,
                    "adjustment": 0.0,
                    "residual_after": current_residual,
                }
            )
            continue

        if phase_name == "players":
            player_result = apply_player_addition_phase(
                starting_residual=current_residual,
                roster_data=roster_data,
                team_a_name=team_a_name,
                team_b_name=team_b_name,
                locked_team_a_player=locked_team_a_players,
                locked_team_b_player=locked_team_b_players,
                player_value_column=player_value_column,
                additional_player_slot_cost=additional_player_slot_cost,
                max_additional_players_per_team=(
                    max_additional_players_per_team
                ),
                max_total_additional_players=(
                    max_total_additional_players
                ),
                balance_tolerance=balance_tolerance,
                team_column=team_column,
                player_name_column=player_name_column,
                roster_slot_column=roster_slot_column,
            )
            current_residual = float(player_result["final_residual"])
            phase_records.append(
                {
                    "phase": "players",
                    "executed": True,
                    "adjustment_applied": player_result[
                        "player_adjustment_applied"
                    ],
                    "reason": player_result["reason"],
                    "residual_before": residual_before_phase,
                    "adjustment": player_result["player_adjustment"],
                    "residual_after": current_residual,
                }
            )
        else:
            pick_result = apply_pick_top_up(
                starting_residual=current_residual,
                team_a_name=team_a_name,
                team_b_name=team_b_name,
                pick_inventory=pick_inventory,
                pick_value_table=pick_value_table,
                outright_pick_hierarchy=outright_pick_hierarchy,
                outright_pick_net_columns=outright_pick_net_columns,
                max_total_picks=max_total_picks,
                balance_tolerance=balance_tolerance,
                excluded_pick_ids=excluded_pick_ids,
            )
            current_residual = float(pick_result["final_residual"])
            phase_records.append(
                {
                    "phase": "picks",
                    "executed": True,
                    "adjustment_applied": pick_result[
                        "pick_top_up_applied"
                    ],
                    "reason": pick_result["reason"],
                    "residual_before": residual_before_phase,
                    "adjustment": pick_result["pick_adjustment"],
                    "residual_after": current_residual,
                }
            )

    return {
        "priority_mode": priority_mode,
        "phase_order": phase_order,
        "initial_residual": initial_residual,
        "additional_player_slot_cost": float(additional_player_slot_cost),
        "locked_team_a_players": locked_team_a_players,
        "locked_team_b_players": locked_team_b_players,
        "player_result": player_result,
        "pick_result": pick_result,
        "hybrid_options": pd.DataFrame(),
        "selected_hybrid_option": None,
        "phase_audit": pd.DataFrame(phase_records),
        "final_residual": current_residual,
        "absolute_final_residual": abs(current_residual),
        "within_balance_tolerance": (
            abs(current_residual) <= balance_tolerance
        ),
    }



def balance_trade(
    roster_data: pd.DataFrame,
    pick_inventory: pd.DataFrame,
    team_a_name: str,
    team_b_name: str,
    team_a_player_names: str | Sequence[str] | None,
    team_b_player_names: str | Sequence[str] | None,
    player_value_column: str,
    additional_player_slot_cost: float,
    pick_value_table: pd.DataFrame,
    outright_pick_hierarchy: Sequence[str],
    outright_pick_net_columns: Sequence[str],
    outright_pick_weights: Sequence[float],
    team_a_draft_counts: Mapping[str, int | float] | None = None,
    team_a_initial_pick_ids: Sequence[str] | None = None,
    team_b_initial_pick_ids: Sequence[str] | None = None,
    priority_mode: str = "best_fit",
    max_additional_players_per_team: int = 2,
    max_total_additional_players: int = 3,
    max_total_picks: int = 5,
    balance_tolerance: float = 0.05,
    excluded_pick_ids: Sequence[str] | None = None,
    hybrid_ranking_mode: str = "closest_first",
    hybrid_package_catalogue: pd.DataFrame | None = None,
    team_column: str = "team",
    player_name_column: str = "candidate_player",
    roster_slot_column: str = "roster_slot",
) -> dict[str, Any]:
    """Score an initial trade, including specific initial picks, and balance it."""
    team_a_names = _normalize_player_names(
        team_a_player_names,
        argument_name="team_a_player_names",
        allow_empty=True,
    )
    team_b_names = _normalize_player_names(
        team_b_player_names,
        argument_name="team_b_player_names",
        allow_empty=True,
    )
    team_a_initial_pick_assets = resolve_initial_pick_assets(
        pick_inventory=pick_inventory,
        team_name=team_a_name,
        selected_pick_ids=team_a_initial_pick_ids,
        outright_pick_hierarchy=outright_pick_hierarchy,
    )
    team_b_initial_pick_assets = resolve_initial_pick_assets(
        pick_inventory=pick_inventory,
        team_name=team_b_name,
        selected_pick_ids=team_b_initial_pick_ids,
        outright_pick_hierarchy=outright_pick_hierarchy,
    )

    if not team_a_names and team_a_initial_pick_assets.empty:
        raise ValueError("Team A must send at least one player or draft pick.")
    if not team_b_names and team_b_initial_pick_assets.empty:
        raise ValueError("Team B must send at least one player or draft pick.")

    itemized_initial_counts = build_initial_draft_counts(
        team_a_initial_pick_assets=team_a_initial_pick_assets,
        team_b_initial_pick_assets=team_b_initial_pick_assets,
        outright_pick_hierarchy=outright_pick_hierarchy,
        outright_pick_net_columns=outright_pick_net_columns,
    )
    if team_a_draft_counts is None:
        initial_draft_counts = itemized_initial_counts
    else:
        explicit_counts = {
            column: float(team_a_draft_counts.get(column, 0))
            for column in outright_pick_net_columns
        }
        if (
            (not team_a_initial_pick_assets.empty or not team_b_initial_pick_assets.empty)
            and any(abs(value) > 1e-12 for value in explicit_counts.values())
        ):
            raise ValueError(
                "Use either specific initial pick IDs or aggregate draft counts, not both."
            )
        initial_draft_counts = explicit_counts

    itemized_pick_adjustment = (
        calculate_initial_pick_adjustment(
            team_a_initial_pick_assets,
            team_b_initial_pick_assets,
        )
        if (not team_a_initial_pick_assets.empty or not team_b_initial_pick_assets.empty)
        else None
    )

    initial_score = score_two_team_multi_player_trade(
        roster_data=roster_data,
        team_a_player_names=team_a_names,
        team_b_player_names=team_b_names,
        team_a_draft_counts=initial_draft_counts,
        player_value_column=player_value_column,
        outright_pick_net_columns=outright_pick_net_columns,
        outright_pick_weights=outright_pick_weights,
        team_a_name=team_a_name,
        team_b_name=team_b_name,
        additional_player_slot_cost=additional_player_slot_cost,
        team_a_itemized_pick_adjustment=itemized_pick_adjustment,
        team_a_initial_pick_assets=team_a_initial_pick_assets,
        team_b_initial_pick_assets=team_b_initial_pick_assets,
        team_column=team_column,
        player_name_column=player_name_column,
    )

    initial_pick_ids = (
        team_a_initial_pick_assets["pick_id"].astype(str).tolist()
        + team_b_initial_pick_assets["pick_id"].astype(str).tolist()
    )
    combined_excluded_pick_ids = list(dict.fromkeys(
        [str(value) for value in (excluded_pick_ids or [])] + initial_pick_ids
    ))

    recommendation = recommend_trade_adjustments(
        base_team_a_residual=initial_score[
            "team_a_symmetric_residual"
        ],
        roster_data=roster_data,
        pick_inventory=pick_inventory,
        team_a_name=team_a_name,
        team_b_name=team_b_name,
        locked_team_a_player=team_a_names,
        locked_team_b_player=team_b_names,
        player_value_column=player_value_column,
        additional_player_slot_cost=additional_player_slot_cost,
        pick_value_table=pick_value_table,
        outright_pick_hierarchy=outright_pick_hierarchy,
        outright_pick_net_columns=outright_pick_net_columns,
        priority_mode=priority_mode,
        max_additional_players_per_team=(
            max_additional_players_per_team
        ),
        max_total_additional_players=max_total_additional_players,
        max_total_picks=max_total_picks,
        balance_tolerance=balance_tolerance,
        excluded_pick_ids=combined_excluded_pick_ids,
        hybrid_ranking_mode=hybrid_ranking_mode,
        hybrid_package_catalogue=hybrid_package_catalogue,
        team_column=team_column,
        player_name_column=player_name_column,
        roster_slot_column=roster_slot_column,
    )

    return {
        "initial_score": initial_score,
        "recommendation": recommendation,
        "priority_mode": priority_mode,
        "team_a_initial_pick_assets": team_a_initial_pick_assets,
        "team_b_initial_pick_assets": team_b_initial_pick_assets,
        "initial_pick_adjustment": float(
            itemized_pick_adjustment or 0.0
        ),
        "additional_player_slot_cost": float(additional_player_slot_cost),
        "initial_residual": initial_score[
            "team_a_symmetric_residual"
        ],
        "final_residual": recommendation["final_residual"],
        "within_balance_tolerance": recommendation[
            "within_balance_tolerance"
        ],
    }
