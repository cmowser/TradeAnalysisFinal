"""Resolve traded player names to NBA person IDs.

The lookup is driven by the manually reviewed
``season_Player_Name_Comparison.csv`` file. It deliberately does not remove
Jr./Sr./II/III/IV when determining identity.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping

import pandas as pd

# Default lookups point to the reviewed reference and filtered box scores.
BASE_PATH = Path(__file__).resolve().parents[1]


_EXCEL_SEASON_REPAIRS = {
    "Jan-00": "2000-01",
    "Feb-01": "2001-02",
    "Mar-02": "2002-03",
    "Apr-03": "2003-04",
    "May-04": "2004-05",
    "Jun-05": "2005-06",
    "Jul-06": "2006-07",
    "Aug-07": "2007-08",
    "Sep-08": "2008-09",
    "Oct-09": "2009-10",
    "Nov-10": "2010-11",
    "Dec-11": "2011-12",
}

# Reviewed identities whose normalized box-score display name is shared by
# multiple people in the same season. These IDs were verified using the name
# annotation and the teams involved in the transaction.
_SPECIAL_PERSON_IDS = {
    "George Johnson (Lee)": 77148,
    "George Johnson (Thomas)": 77149,
    "Charles Smith (Daniel)": 293,
    "Marcus Williams (D.)": 200766,
    "Chuck Williams / Chuckie Williams": 78547,
}

_SEASON_SPECIFIC_PERSON_IDS = {
    # The Hawks-to-Cavaliers player is Edward Lee Johnson Jr. (77144), not
    # the Sacramento/Phoenix scorer whose box-score display name is identical.
    ("1985-86", "Eddie Johnson"): 77144
}

_NON_PLAYER_ASSET = re.compile(
    r"(?:\bpicks?\b|\bcash\b|considerations?|trad(?:e|ed)[ -]player exception|"
    r"trade exception|cap exception|exemption|compensation|cap room|draft choice|"
    r"option to swap|right to swap|^\$|^#|protected|protection|less favorable|"
    r"more favorable|not exercised|extinguished)",
    re.IGNORECASE,
)
_NON_PLAYER_SENTENCE = re.compile(
    r"(?:agreed? to|agreement not to|waived rights?|relinquished right|"
    r"player to be named|^plus |^who |^else |^then |^thereafter |^top \d|"
    r"^or \d|^\d{4}(?:\s|$)|^\(|^not per|^middle of|^lesser of)",
    re.IGNORECASE,
)
_FOOTNOTE = re.compile(r"\s*\((?:a|b|c|E\.|R\.|C\.|T\.|S\.)\)\s*", re.IGNORECASE)


def _canonical_season(value: Any) -> str:
    value = str(value)
    return _EXCEL_SEASON_REPAIRS.get(value, value)


def _season_start(season: str) -> int:
    return int(_canonical_season(season).split("-", 1)[0])


def _comparison_period(row: Mapping[str, Any]) -> tuple[str, str]:
    """Return (comparison season, segment) using the same rules as the audit."""
    season = _canonical_season(row.get("Season", ""))
    if season and season.lower() != "offseason" and re.fullmatch(r"\d{4}-\d{2}", season):
        return season, "regular_season"

    date = pd.Timestamp(row["Date"])
    return f"{date.year}-{str(date.year + 1)[-2:]}", "offseason"


def _extract_player_names(asset_value: Any) -> list[str]:
    # Remove pick, cash, and rights language before attempting name matching.
    """Extract player-name assets using the rules used to create the review."""
    if pd.isna(asset_value):
        return []

    names: list[str] = []
    for token in str(asset_value).split(","):
        raw = token.strip()
        if not raw or _NON_PLAYER_ASSET.search(raw) or _NON_PLAYER_SENTENCE.search(raw):
            continue

        name = re.sub(r"(?i)^(?:rights to\s+)+", "", raw).strip()
        name = re.sub(r"(?i)^restricted free agent\s+", "", name).strip()
        name = re.sub(r"\s*\(changed from.*$", "", name).strip()
        name = _FOOTNOTE.sub("", name).strip()

        if (
            not name
            or re.search(r"\d", name)
            or len(name) > 100
            or re.search(r"\b(?:team|roster|free agent|coach|director|select|draft|void trade)\b", name, re.IGNORECASE)
        ):
            continue
        names.append(name)
    return names


class TransactionPlayerIDMatcher:
    """Cached resolver for transaction-row player names and NBA person IDs."""

    def __init__(
        self,
        comparison_csv: str | Path = BASE_PATH / "data" / "reference" / "season_Player_Name_Comparison.csv",
        player_stats_parquet: str | Path = BASE_PATH / "data" / "interim" / "filtered_Player_Stats.parquet",
    ) -> None:
        # Only manually approved normalization rows are eligible for matching.
        comparison = pd.read_csv(comparison_csv)
        comparison["Season"] = comparison["Season"].map(_canonical_season)
        comparison["manual_match"] = comparison["manual_match"].fillna("").str.lower()

        unapproved = comparison[
            comparison["match_status"].eq("normalization_needed") & ~comparison["manual_match"].isin({"yes", "approved", "true", "1"})
        ]
        if not unapproved.empty:
            raise ValueError("The comparison file contains unapproved normalization rows: " f"{len(unapproved)}")

        self._comparison = {(r.Season, r.season_segment, r.transaction_player_name): r for r in comparison.itertuples(index=False)}

        stats = pd.read_parquet(player_stats_parquet, columns=["firstName", "lastName", "personId", "game_Date"])
        stats = stats.dropna(subset=["personId", "game_Date"]).copy()
        stats["game_Date"] = pd.to_datetime(stats["game_Date"])
        stats["Season"] = stats["game_Date"].map(
            lambda d: (f"{d.year - 1}-{str(d.year)[-2:]}" if d.month < 7 else f"{d.year}-{str(d.year + 1)[-2:]}")
        )
        stats["box_score_name"] = (stats["firstName"].fillna("").str.strip() + " " + stats["lastName"].fillna("").str.strip()).str.strip()
        stats["personId"] = stats["personId"].astype(int)
        self._appearances = stats[["Season", "box_score_name", "personId"]].drop_duplicates()

    @staticmethod
    def _reviewed_box_names(review_row: Any) -> list[str]:
        if pd.isna(review_row.box_score_player_name):
            return []
        return [name.strip() for name in str(review_row.box_score_player_name).split(" | ") if name.strip()]

    def _resolve_one(self, transaction_name: str, season: str, segment: str) -> dict[str, Any]:
        review = self._comparison.get((season, segment, transaction_name))
        if review is None:
            return {
                "player_id": None,
                "match_status": "not_in_review_file",
                "box_score_player_name": None,
                "most_recent_season": None,
                "model_value": 0,
            }

        box_names = self._reviewed_box_names(review)
        status = review.match_status

        if status == "unresolved_name" or not box_names:
            return {"player_id": None, "match_status": status, "box_score_player_name": None, "most_recent_season": None, "model_value": 0}

        reviewed_id = _SEASON_SPECIFIC_PERSON_IDS.get((season, transaction_name), _SPECIAL_PERSON_IDS.get(transaction_name))
        if reviewed_id is not None:
            player_id = reviewed_id
            appearances = self._appearances[self._appearances["personId"].eq(player_id)]
        else:
            appearances = self._appearances[self._appearances["box_score_name"].isin(box_names)]
            in_season = appearances[appearances["Season"].eq(season)]
            ids = set(in_season["personId"])
            player_id = next(iter(ids)) if len(ids) == 1 else None

        if status in {"exact_match", "normalization_needed"}:
            # Refuse to merge careers when a reviewed display name is still
            # ambiguous within the relevant season.
            return {
                "player_id": player_id,
                "match_status": status if player_id is not None else "ambiguous_person_id",
                "box_score_player_name": " | ".join(box_names),
                "most_recent_season": season if player_id is not None else None,
                "model_value": None,
            }

        # No box score in the transaction season: only use an NBA ID if this
        # person appeared in an earlier season. Never look forward.
        prior = appearances[appearances["Season"].map(_season_start) < _season_start(season)]
        if prior.empty:
            return {
                "player_id": None,
                "match_status": "no_prior_nba_appearance",
                "box_score_player_name": " | ".join(box_names),
                "most_recent_season": None,
                "model_value": 0,
            }

        latest = max(prior["Season"], key=_season_start)
        latest_ids = set(prior.loc[prior["Season"].eq(latest), "personId"])
        prior_id = next(iter(latest_ids)) if len(latest_ids) == 1 else None
        return {
            "player_id": prior_id,
            "match_status": ("prior_nba_appearance" if prior_id is not None else "ambiguous_person_id"),
            "box_score_player_name": " | ".join(box_names),
            "most_recent_season": latest if prior_id is not None else None,
            "model_value": 0,
        }

    def match_transaction(self, transaction_row: Mapping[str, Any], *, include_details: bool = False) -> dict[str, dict[str, Any]]:
        """Return acquired/relinquished player-name-to-ID mappings.

        Set ``include_details=True`` to return each player's status, reviewed
        box-score name, most recent applicable season, and zero-value flag.
        """
        season, segment = _comparison_period(transaction_row)
        result: dict[str, dict[str, Any]] = {"Acquired": {}, "Relinquished": {}}

        for side in result:
            for name in _extract_player_names(transaction_row.get(side)):
                # A plausible-looking token that is absent from the reviewed
                # player-season file is an asset/team fragment, not a player.
                if (season, segment, name) not in self._comparison:
                    continue
                detail = self._resolve_one(name, season, segment)
                result[side][name] = detail if include_details else detail["player_id"]
        return result


_DEFAULT_MATCHER: TransactionPlayerIDMatcher | None = None


def match_transaction_players(transaction_row: Mapping[str, Any], *, include_details: bool = False) -> dict[str, dict[str, Any]]:
    """Convenience function that lazily loads and then reuses the lookup data."""
    global _DEFAULT_MATCHER
    if _DEFAULT_MATCHER is None:
        _DEFAULT_MATCHER = TransactionPlayerIDMatcher()
    return _DEFAULT_MATCHER.match_transaction(transaction_row, include_details=include_details)
