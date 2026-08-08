"""Extract draft compensation from the cleaned NBA transaction assets.

The parser is intentionally conservative. Every extracted record retains its
original text, a parse-confidence flag, and a review flag. Realized outcomes
(eventual pick number/player) are separated from information known on the
transaction date to avoid look-ahead leakage.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

# Normalize written round and quantity words before parsing asset text.
ROUND_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8}
COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

_ROUND_RE = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth)" r"(?:[- ]round| round)\b", re.IGNORECASE)
_PICK_RE = re.compile(r"\b(?:draft\s+)?picks?\b|\b(?:future\s+)?draft considerations?\b", re.IGNORECASE)
_SWAP_RE = re.compile(r"\b(?:right|rights|option)\s+to\s+swap\b|\bswap\s+(?:rights?|option)\b", re.IGNORECASE)
_DRAFT_RIGHTS_RE = re.compile(r"^rights\s+to\s+(.+)$", re.IGNORECASE)
_NON_DRAFT_RIGHT_RE = re.compile(
    r"right of first refusal|rights? to (?:sign|select|free agent|restricted free agent|" r"GM\b|coach\b)|waive[sd]? rights?", re.IGNORECASE
)
_PROTECTION_RE = re.compile(
    r"protect(?:ed|ion)?|lottery|top\s*\d+|#\d+[-–]\d+|unprotected|"
    r"less favorable|more favorable|least favorable|most favorable|"
    r"if conveyed|if not conveyed|otherwise|else|extinguish|conditional",
    re.IGNORECASE,
)
_NOT_EXERCISED_RE = re.compile(r"not exercised|not conveyed|extinguished|never conveyed", re.IGNORECASE)
_REALIZED_RE = re.compile(r"(?:\b(19\d{2}|20\d{2})\s+)?#(\d{1,3})-([^()]+?)(?=\)|$)", re.IGNORECASE)
_FROM_TEAM_RE = re.compile(r"\bfrom\s+([A-Za-z0-9 .'-]+?)(?=,|\)|$)", re.IGNORECASE)


def split_top_level(value: Any) -> list[str]:
    # Split asset lists without breaking text contained in parentheses.
    """Split comma-separated assets without splitting parenthetical clauses."""
    if pd.isna(value):
        return []
    text = str(value)
    parts: list[str] = []
    start = 0
    depth = 0
    for i, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            part = text[start:i].strip()
            if part:
                parts.append(part)
            start = i + 1
    part = text[start:].strip()
    if part:
        parts.append(part)
    return parts


def _years_known_at_trade(text: str, transaction_year: int) -> list[int]:
    """Return plausible draft years, excluding years only in realized outcomes."""
    scrubbed = _REALIZED_RE.sub("", text)
    years = sorted({int(y) for y in re.findall(r"\b(?:19\d{2}|20\d{2})\b", scrubbed)})
    # A historical date sometimes appears in amendment language. Draft assets
    # overwhelmingly refer to the current or a future draft.
    return [y for y in years if y >= transaction_year - 1]


def _remove_realized_outcomes(text: str) -> str:
    """Remove eventual #pick-player annotations from transaction-date text."""
    cleaned = _REALIZED_RE.sub("", text)
    cleaned = re.sub(r"\(\s*(?:19\d{2}|20\d{2})?\s*\)", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _pick_count(text: str, years: list[int]) -> int | None:
    match = re.search(
        r"\b(one|two|three|four|five|[1-5])\s+" r"(?:(?:conditional|future)\s+)?(?:first|second|third|fourth|draft)", text, re.IGNORECASE
    )
    if match:
        token = match.group(1).lower()
        return COUNT_WORDS.get(token, int(token) if token.isdigit() else None)
    if len(years) > 1 and re.search(r"\bpicks\b", text, re.IGNORECASE):
        return len(years)
    return 1


def _rounds(text: str) -> list[int | None]:
    rounds = list(dict.fromkeys(ROUND_WORDS[m.lower()] for m in _ROUND_RE.findall(text)))
    return rounds or [None]


def parse_asset_component(component: str, *, transaction_date: Any) -> list[dict[str, Any]]:
    # Preserve only information known on the transaction date.
    """Parse one top-level asset component into zero or more draft records."""
    raw = component.strip()
    if not raw or _NON_DRAFT_RIGHT_RE.search(raw):
        return []

    is_swap = bool(_SWAP_RE.search(raw))
    rights_match = _DRAFT_RIGHTS_RE.match(raw)
    is_pick = bool(_PICK_RE.search(raw) or _ROUND_RE.search(raw))

    if rights_match and not is_swap and not is_pick:
        player = re.sub(r"(?i)^(?:rights to\s+)+", "", rights_match.group(1)).strip()
        return [
            {
                "asset_type": "draft_rights",
                "draft_round": None,
                "draft_years": "[]",
                "pick_count": 1,
                "protection_text": None,
                "source_team_text": None,
                "rights_player_name": player,
                "realized_year": None,
                "realized_pick_number": None,
                "realized_player_name": None,
                "not_exercised_or_extinguished": False,
                "parse_confidence": "high",
                "needs_manual_review": False,
                "known_at_transaction_text": raw,
                "raw_asset_text": raw,
            }
        ]

    if not (is_pick or is_swap):
        return []

    date = pd.Timestamp(transaction_date)
    years = _years_known_at_trade(raw, date.year)
    count = _pick_count(raw, years)
    realized = _REALIZED_RE.search(raw)
    from_team = _FROM_TEAM_RE.search(raw)
    known_text = _remove_realized_outcomes(raw)
    protection = known_text if _PROTECTION_RE.search(known_text) else None
    generic_pick = bool(re.search(r"\bdraft picks?(?:\(s\))?\b|\bdraft pick\(s\)", raw, re.I))
    rounds = _rounds(raw)

    output = []
    for draft_round in rounds:
        confidence = "high"
        review = False
        if draft_round is None or not years:
            confidence = "medium"
        if generic_pick or len(rounds) > 1 or "?" in raw:
            review = True
        output.append(
            {
                "asset_type": "pick_swap" if is_swap else "draft_pick",
                "draft_round": draft_round,
                "draft_years": json.dumps(years),
                "pick_count": count,
                "protection_text": protection,
                "source_team_text": from_team.group(1).strip() if from_team else None,
                "rights_player_name": None,
                "realized_year": int(realized.group(1)) if realized and realized.group(1) else None,
                "realized_pick_number": int(realized.group(2)) if realized else None,
                "realized_player_name": realized.group(3).strip() if realized else None,
                "not_exercised_or_extinguished": bool(_NOT_EXERCISED_RE.search(raw)),
                "parse_confidence": confidence,
                "needs_manual_review": review,
                "known_at_transaction_text": known_text,
                "raw_asset_text": raw,
            }
        )
    return output


def extract_transaction_draft_assets(transactions: pd.DataFrame) -> pd.DataFrame:
    """Create one row per extracted draft asset and transaction-team side."""
    rows: list[dict[str, Any]] = []
    for row in transactions.itertuples(index=False):
        source_index = getattr(row, "index")
        for side in ("Acquired", "Relinquished"):
            value = getattr(row, side)
            for component_number, component in enumerate(split_top_level(value), start=1):
                parsed = parse_asset_component(component, transaction_date=row.Date)
                for subasset_number, asset in enumerate(parsed, start=1):
                    rows.append(
                        {
                            "source_transaction_index": int(source_index),
                            "transaction_date": row.Date,
                            "transaction_season": row.Season,
                            "transaction_team": row.Team,
                            "transaction_team_id": row.teamID,
                            "transaction_notes": row.Notes,
                            "transaction_side": side,
                            "component_number": component_number,
                            "subasset_number": subasset_number,
                            **asset,
                        }
                    )
    return pd.DataFrame(rows)


def summarize_draft_assets(transactions: pd.DataFrame, draft_assets: pd.DataFrame) -> pd.DataFrame:
    """Create transaction-team-row draft counts suitable for feature joins."""
    base = transactions[["index", "Date", "Team", "teamID"]].rename(
        columns={
            "index": "source_transaction_index",
            "Date": "transaction_date",
            "Team": "transaction_team",
            "teamID": "transaction_team_id",
        }
    )
    if draft_assets.empty:
        return base

    work = draft_assets.copy()
    work["direction"] = work["transaction_side"].str.lower()

    # Multiple rows from one component represent possible round outcomes, not
    # necessarily multiple assets. Collapse them before counting package size.
    component = work.groupby(["source_transaction_index", "direction", "component_number"], as_index=False).agg(
        asset_type=("asset_type", "first"),
        pick_count=("pick_count", "max"),
        round_option_count=("draft_round", lambda x: x.dropna().nunique()),
        possible_first=("draft_round", lambda x: int((x == 1).any())),
        possible_second=("draft_round", lambda x: int((x == 2).any())),
        possible_later=("draft_round", lambda x: int((x >= 3).any())),
        has_known_round=("draft_round", lambda x: int(x.notna().any())),
        needs_review=("needs_manual_review", "max"),
    )
    component["is_swap"] = component["asset_type"].eq("pick_swap").astype(int)
    component["is_rights"] = component["asset_type"].eq("draft_rights").astype(int)
    component["is_pick"] = component["asset_type"].eq("draft_pick").astype(int)
    component["definite_round"] = component["round_option_count"].eq(1)
    units = component["pick_count"].fillna(1)
    component["definite_first_units"] = (units * component["possible_first"] * component["definite_round"] * component["is_pick"]).astype(
        int
    )
    component["definite_second_units"] = (units * component["possible_second"] * component["definite_round"] * component["is_pick"]).astype(
        int
    )
    component["definite_later_units"] = (units * component["possible_later"] * component["definite_round"] * component["is_pick"]).astype(
        int
    )
    component["unknown_round"] = (component["is_pick"].eq(1) & component["has_known_round"].eq(0)).astype(int)

    pieces = []
    for direction, group in component.groupby("direction"):
        summary = (
            group.groupby("source_transaction_index")
            .agg(
                **{
                    f"draft_assets_{direction}": ("asset_type", "size"),
                    f"definite_first_round_units_{direction}": ("definite_first_units", "sum"),
                    f"definite_second_round_units_{direction}": ("definite_second_units", "sum"),
                    f"definite_later_round_units_{direction}": ("definite_later_units", "sum"),
                    f"possible_first_round_assets_{direction}": ("possible_first", "sum"),
                    f"possible_second_round_assets_{direction}": ("possible_second", "sum"),
                    f"possible_later_round_assets_{direction}": ("possible_later", "sum"),
                    f"unknown_round_assets_{direction}": ("unknown_round", "sum"),
                    f"pick_swaps_{direction}": ("is_swap", "sum"),
                    f"draft_rights_{direction}": ("is_rights", "sum"),
                    f"draft_assets_needing_review_{direction}": ("needs_review", "sum"),
                }
            )
            .reset_index()
        )
        pieces.append(summary)

    result = base.copy()
    for piece in pieces:
        result = result.merge(piece, on="source_transaction_index", how="left")
    numeric = [c for c in result if c not in base.columns]
    result[numeric] = result[numeric].fillna(0).astype(int)
    return result


def build_draft_outputs(
    transactions_path: str | Path = "transactions_with_WL.csv",
    player_features_path: str | Path = "transaction_player_features.parquet",
    output_directory: str | Path = ".",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract, summarize, and join draft assets to processed player rows."""
    output_directory = Path(output_directory)
    transactions = pd.read_csv(transactions_path)
    assets = extract_transaction_draft_assets(transactions)
    summary = summarize_draft_assets(transactions, assets)
    player_features = pd.read_parquet(player_features_path)
    joined = player_features.merge(
        summary.drop(columns=["transaction_date", "transaction_team", "transaction_team_id"]),
        on="source_transaction_index",
        how="left",
        validate="many_to_one",
    )

    # Raw source text was useful during parser development; the retained
    # known-at-transaction text is sufficient for the modeling dataset.
    assets_output = assets.drop(columns=["raw_asset_text"])
    assets_output.to_csv(output_directory / "transaction_draft_assets.csv", index=False)
    assets_output.to_parquet(output_directory / "transaction_draft_assets.parquet", index=False)
    summary.to_parquet(output_directory / "transaction_draft_summary.parquet", index=False)
    joined.to_parquet(output_directory / "transaction_player_features_with_draft.parquet", index=False)
    return assets_output, summary, joined


if __name__ == "__main__":
    extracted, summarized, player_joined = build_draft_outputs()
    print(f"Extracted draft-asset rows: {len(extracted):,}")
    print(f"Transaction summary rows: {len(summarized):,}")
    print(f"Joined player-feature rows: {len(player_joined):,}")
