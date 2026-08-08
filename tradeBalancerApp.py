from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st

import tradeBalancer as tb


BASE_PATH = Path(__file__).resolve().parent
PROCESSED_PATH = BASE_PATH / "data" / "processed"

ROSTER_VALUES_PATH = (
    PROCESSED_PATH / "current_roster_player_values_efficiency_load_role_scaled_base_regularized.parquet"
)
PICK_INVENTORY_PATH = PROCESSED_PATH / "pick_Inventory.csv"
HYBRID_PACKAGE_CATALOGUE_PATH = (
    PROCESSED_PATH / "team_hybrid_package_catalogue.parquet"
)
PICK_VALUES_PATH = (
    PROCESSED_PATH / "anchored_efficiency_load_role_scaled_base_regularized_pick_values.parquet"
)
MODEL_BUNDLE_PATH = (
    PROCESSED_PATH / "draft_weight_model_bundle_efficiency_load_role_scaled_base_regularized.json"
)
APP_WRITEUPS_PATH = BASE_PATH / "APP_WRITEUPS.md"

TEAM_COLUMN = "END_TEAM_FULL_NAME"
PLAYER_NAME_COLUMN = "PLAYER_NAME"
PLAYER_VALUE_COLUMN = "PLAYER_PRODUCTION_VALUE"
ELIGIBLE_COLUMN = "PRODUCTION_ELIGIBLE"
REASON_COLUMN = "PRODUCTION_INELIGIBILITY_REASON"
BALANCE_TOLERANCE = 0.05


def load_app_writeups(path: Path) -> list[tuple[str, str]]:
    """Load level-two Markdown sections for the app's project notes."""
    sections = []
    title = None
    body = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if title is not None:
                sections.append((title, "\n".join(body).strip()))
            title = line[3:].strip()
            body = []
        elif title is not None:
            body.append(line)

    if title is not None:
        sections.append((title, "\n".join(body).strip()))

    return sections


@st.cache_data
def load_trade_data():
    roster_data = pd.read_parquet(ROSTER_VALUES_PATH)
    pick_inventory = pd.read_csv(PICK_INVENTORY_PATH)
    hybrid_package_catalogue = pd.read_parquet(
        HYBRID_PACKAGE_CATALOGUE_PATH
    )
    pick_value_table = pd.read_parquet(PICK_VALUES_PATH)

    with MODEL_BUNDLE_PATH.open("r", encoding="utf-8") as file:
        model_bundle = json.load(file)

    outright_pick_hierarchy = list(
        model_bundle["outright_pick_hierarchy"]
    )
    outright_pick_net_columns = list(
        model_bundle["outright_pick_net_columns"]
    )

    pick_lookup = pick_value_table.set_index("tier")
    outright_pick_weights = (
        pick_lookup.loc[outright_pick_hierarchy, "estimated_value"]
        .astype("float64")
        .to_numpy()
    )

    # Calculate each pick's value from its tier and projection confidence.
    tier_value_lookup = (
        pick_value_table.drop_duplicates("tier")
        .set_index("tier")["estimated_value"]
        .astype("float64")
        .to_dict()
    )
    pick_inventory = pick_inventory.copy()
    pick_inventory["base_value"] = (
        pick_inventory["tier"].map(tier_value_lookup)
    )
    pick_inventory["floor_value"] = (
        pick_inventory["floor_tier"].map(tier_value_lookup)
    )
    pick_inventory["tier_confidence"] = pd.to_numeric(
        pick_inventory["tier_confidence"], errors="coerce"
    )
    pick_inventory["tier_decay"] = (
        1.0 - pick_inventory["tier_confidence"]
    )
    pick_inventory["adjusted_value"] = (
        pick_inventory["tier_confidence"]
        * pick_inventory["base_value"]
        + pick_inventory["tier_decay"]
        * pick_inventory["floor_value"]
    )
    pick_inventory["adjusted_value"] = np.maximum(
        pick_inventory["adjusted_value"],
        pick_inventory["floor_value"],
    )

    additional_player_slot_cost = float(
        model_bundle["additional_player_slot_cost"]
    )
    return (
        roster_data,
        pick_inventory,
        hybrid_package_catalogue,
        pick_value_table,
        outright_pick_hierarchy,
        outright_pick_net_columns,
        outright_pick_weights,
        additional_player_slot_cost,
        model_bundle,
    )


def as_list(value):
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return [
            str(item) for item in value
            if not (isinstance(item, float) and np.isnan(item))
        ]
    return [str(value)]



def extract_pick_assets(value):
    """Normalize selected pick output into displayable asset dictionaries."""
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return []
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return [value.to_dict()]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)):
        records = []
        for item in value:
            if isinstance(item, dict):
                records.append(item)
            else:
                records.append({"pick_id": str(item)})
        return records
    if isinstance(value, float) and np.isnan(value):
        return []
    return [{"pick_id": str(value)}]



def format_pick_option(pick_id, inventory):
    """Create a compact label for a specific pick dropdown option."""
    matches = inventory.loc[
        inventory["pick_id"].astype(str).eq(str(pick_id))
    ]
    if matches.empty:
        return str(pick_id)
    asset = matches.iloc[0]
    year = int(float(asset["draft_year"]))
    round_number = int(float(asset["round"]))
    tier = str(asset["tier"]).replace("projected_", "").replace("_", " ").title()
    value = float(asset["adjusted_value"])
    confidence = 100 * float(asset["tier_confidence"])
    return (
        f"{year} | Round {round_number} | {tier} "
        f"({confidence:.0f}% confidence, value {value:.3f})"
    )


def extract_balanced_proposal(trade_result):
    initial_score = trade_result["initial_score"]
    recommendation = trade_result["recommendation"]
    mode = trade_result["priority_mode"]

    team_a_extra = []
    team_b_extra = []
    pick_sending_team = None
    selected_pick_assets = []
    team_a_initial_pick_assets = extract_pick_assets(
        trade_result.get("team_a_initial_pick_assets")
    )
    team_b_initial_pick_assets = extract_pick_assets(
        trade_result.get("team_b_initial_pick_assets")
    )
    option_type = "initial_trade_only"
    team_a_package_value = float(
        initial_score["team_a_player_package_value_sent"]
    )
    team_b_package_value = float(
        initial_score["team_b_player_package_value_sent"]
    )
    pick_package_value = 0.0
    player_match_cost = float(initial_score.get("player_match_cost", 0.0))
    unmatched_player_value = float(
        initial_score.get("unmatched_player_value", 0.0)
    )
    maximum_player_match_gap = float(
        initial_score.get("maximum_player_match_gap", 0.0)
    )
    player_match_pairs = initial_score.get("player_match_pairs", [])

    if mode in {"best_fit", "player_matching"}:
        selected = recommendation.get("selected_hybrid_option")
        if selected is not None:
            option_type = str(selected.get("option_type", "no_adjustment"))
            team_a_extra = as_list(
                selected.get("team_a_additional_players_sent")
            )
            team_b_extra = as_list(
                selected.get("team_b_additional_players_sent")
            )
            team_a_package_value = float(
                selected.get(
                    "team_a_full_player_package_value_sent",
                    team_a_package_value,
                )
            )
            team_b_package_value = float(
                selected.get(
                    "team_b_full_player_package_value_sent",
                    team_b_package_value,
                )
            )
            pick_sending_team = selected.get("pick_sending_team")
            pick_package_value = float(selected.get("pick_package_value", 0.0))
            player_match_cost = float(
                selected.get("player_match_cost", player_match_cost)
            )
            unmatched_player_value = float(
                selected.get(
                    "unmatched_player_value", unmatched_player_value
                )
            )
            maximum_player_match_gap = float(
                selected.get(
                    "maximum_player_match_gap", maximum_player_match_gap
                )
            )
            player_match_pairs = selected.get(
                "player_match_pairs", player_match_pairs
            )
            selected_pick_assets = extract_pick_assets(
                selected.get("selected_pick_assets")
            )
            if not selected_pick_assets:
                selected_pick_assets = extract_pick_assets(
                    selected.get("selected_pick_ids")
                )
    else:
        player_result = recommendation.get("player_result")
        if (
            player_result is not None
            and player_result.get("player_adjustment_applied", False)
        ):
            selected_players = player_result.get("selected_player_package")
            if selected_players is not None:
                team_a_extra = as_list(
                    selected_players.get("team_a_additional_players_sent")
                )
                team_b_extra = as_list(
                    selected_players.get("team_b_additional_players_sent")
                )
                team_a_package_value = float(
                    selected_players.get(
                        "team_a_full_player_package_value_sent",
                        team_a_package_value,
                    )
                )
                team_b_package_value = float(
                    selected_players.get(
                        "team_b_full_player_package_value_sent",
                        team_b_package_value,
                    )
                )
                player_match_cost = float(
                    selected_players.get(
                        "player_match_cost", player_match_cost
                    )
                )
                unmatched_player_value = float(
                    selected_players.get(
                        "unmatched_player_value", unmatched_player_value
                    )
                )
                maximum_player_match_gap = float(
                    selected_players.get(
                        "maximum_player_match_gap",
                        maximum_player_match_gap,
                    )
                )
                player_match_pairs = selected_players.get(
                    "player_match_pairs", player_match_pairs
                )

        pick_result = recommendation.get("pick_result")
        if (
            pick_result is not None
            and pick_result.get("pick_top_up_applied", False)
        ):
            pick_sending_team = pick_result.get("pick_sending_team")
            pick_package_value = abs(float(pick_result.get("pick_adjustment", 0.0)))
            selected_pick_assets = extract_pick_assets(
                pick_result.get("selected_pick_assets")
            )

        has_players = bool(team_a_extra or team_b_extra)
        has_picks = bool(selected_pick_assets)
        if has_players and has_picks:
            option_type = "players_and_picks"
        elif has_players:
            option_type = "players_only"
        elif has_picks:
            option_type = "picks_only"

    team_a_name = initial_score["team_a_name"]
    team_b_name = initial_score["team_b_name"]

    return {
        "team_a_name": team_a_name,
        "team_b_name": team_b_name,
        "team_a_players": (
            as_list(initial_score["team_a_players_sent"]) + team_a_extra
        ),
        "team_b_players": (
            as_list(initial_score["team_b_players_sent"]) + team_b_extra
        ),
        "team_a_picks": (
            team_a_initial_pick_assets
            + (selected_pick_assets if pick_sending_team == team_a_name else [])
        ),
        "team_b_picks": (
            team_b_initial_pick_assets
            + (selected_pick_assets if pick_sending_team == team_b_name else [])
        ),
        "team_a_player_package_value": team_a_package_value,
        "team_b_player_package_value": team_b_package_value,
        "pick_sending_team": pick_sending_team,
        "pick_package_value": pick_package_value,
        "team_a_pick_package_value": float(
            sum(float(asset.get("adjusted_value", 0.0) or 0.0)
                for asset in team_a_initial_pick_assets)
            + (pick_package_value if pick_sending_team == team_a_name else 0.0)
        ),
        "team_b_pick_package_value": float(
            sum(float(asset.get("adjusted_value", 0.0) or 0.0)
                for asset in team_b_initial_pick_assets)
            + (pick_package_value if pick_sending_team == team_b_name else 0.0)
        ),
        "player_match_cost": player_match_cost,
        "unmatched_player_value": unmatched_player_value,
        "maximum_player_match_gap": maximum_player_match_gap,
        "player_match_pairs": player_match_pairs,
        "option_type": option_type,
    }



def render_team_package(
    team_name,
    players,
    picks,
    player_package_value=None,
    pick_package_value=0.0,
):
    st.markdown(f"### {team_name} sends")
    for player in players:
        st.write(f"• {player}")

    if player_package_value is not None:
        st.caption(
            "Slot-adjusted on-court production value: "
            f"{float(player_package_value):.3f}"
        )

    if not picks:
        return

    st.markdown("**Draft picks**")
    for asset in picks:
        if not isinstance(asset, dict):
            st.write(f"• {asset}")
            continue

        pick_id = str(asset.get("pick_id", "Draft pick"))
        draft_year = asset.get("draft_year")
        round_number = asset.get("round")
        tier = str(asset.get("tier", "")).replace(
            "projected_", ""
        ).replace("_", " ")
        confidence = pd.to_numeric(
            pd.Series([asset.get("tier_confidence")]),
            errors="coerce",
        ).iloc[0]
        adjusted_value = pd.to_numeric(
            pd.Series([asset.get("adjusted_value")]),
            errors="coerce",
        ).iloc[0]

        descriptor_parts = []
        if pd.notna(draft_year):
            descriptor_parts.append(str(int(float(draft_year))))
        if pd.notna(round_number):
            descriptor_parts.append(f"Round {int(float(round_number))}")
        if tier:
            descriptor_parts.append(tier.title())

        descriptor = " | ".join(descriptor_parts) or pick_id
        details = []
        if pd.notna(confidence):
            details.append(f"confidence {100 * float(confidence):.0f}%")
        if pd.notna(adjusted_value):
            details.append(f"value {float(adjusted_value):.3f}")

        suffix = f" ({', '.join(details)})" if details else ""
        st.write(f"• {descriptor}{suffix}")

    st.caption(
        f"Total adjusted pick value: {float(pick_package_value):.3f}"
    )



def render_player_matches(
    player_match_pairs,
    team_a_name,
    team_b_name,
    title="Player-to-player production matching",
):
    if player_match_pairs is None:
        return
    if isinstance(player_match_pairs, pd.DataFrame):
        match_data = player_match_pairs.copy()
    else:
        try:
            match_data = pd.DataFrame(list(player_match_pairs))
        except TypeError:
            return
    if match_data.empty:
        return

    display = match_data.rename(
        columns={
            "pair_rank": "Match",
            "team_a_player": team_a_name,
            "team_a_player_value": f"{team_a_name} value",
            "team_b_player": team_b_name,
            "team_b_player_value": f"{team_b_name} value",
            "absolute_player_value_gap": "Absolute gap",
            "is_unmatched_pair": "Unmatched",
        }
    )
    display_columns = [
        "Match",
        team_a_name,
        f"{team_a_name} value",
        team_b_name,
        f"{team_b_name} value",
        "Absolute gap",
        "Unmatched",
    ]
    display_columns = [
        column for column in display_columns if column in display.columns
    ]
    with st.expander(title):
        st.dataframe(
            display[display_columns],
            width="stretch",
            hide_index=True,
        )


def render_ranked_options(trade_result):
    recommendation = trade_result["recommendation"]
    if trade_result["priority_mode"] not in {"best_fit", "player_matching"}:
        phase_audit = recommendation.get("phase_audit")
        if isinstance(phase_audit, pd.DataFrame) and not phase_audit.empty:
            with st.expander("Balancing phases"):
                st.dataframe(
                    phase_audit,
                    width="stretch",
                    hide_index=True,
                )
        return

    options = recommendation.get("hybrid_options")
    if not isinstance(options, pd.DataFrame) or options.empty:
        return

    display_columns = [
        "recommendation_rank",
        "option_type",
        "team_a_additional_players_sent",
        "team_b_additional_players_sent",
        "pick_sending_team",
        "selected_pick_ids",
        "team_a_full_player_package_value_sent",
        "team_b_full_player_package_value_sent",
        "player_match_cost",
        "unmatched_player_value",
        "maximum_player_match_gap",
        "pick_package_value",
        "total_additional_players",
        "total_picks",
        "final_residual",
        "within_balance_tolerance",
    ]
    display_columns = [
        column for column in display_columns if column in options.columns
    ]
    ranked_options = options[display_columns].head(20).copy()
    for column in (
        "team_a_additional_players_sent",
        "team_b_additional_players_sent",
        "selected_pick_ids",
    ):
        if column in ranked_options.columns:
            ranked_options[column] = ranked_options[column].apply(
                lambda value: ", ".join(as_list(value))
            )
    with st.expander("Ranked balancing alternatives"):
        st.dataframe(
            ranked_options,
            width="stretch",
            hide_index=True,
        )


def render_fairness_scale(
    residual,
    team_a_name,
    team_b_name,
    tolerance,
    scale_limit=1.5,
):
    # Keep the visual marker within the scale
    clamped_residual = max(
        -scale_limit,
        min(scale_limit, residual)
    )

    # Convert residual to a 0–100% position
    marker_position = ((scale_limit - clamped_residual) / (2 * scale_limit)) * 100

    # Convert tolerance zone to percentages
    tolerance_start = (
        (-tolerance + scale_limit)
        / (2 * scale_limit)
    ) * 100

    tolerance_width = (
        (2 * tolerance)
        / (2 * scale_limit)
    ) * 100

    st.html(f"""
    <div class="fairness-scale">

        <div class="fairness-labels">
            <span>{team_a_name.upper()} ADVANTAGE</span>
            <span>BALANCED</span>
            <span>{team_b_name.upper()} ADVANTAGE</span>
        </div>

        <div class="fairness-track">

            <div
                class="fairness-tolerance"
                style="
                    left: {tolerance_start}%;
                    width: {tolerance_width}%;
                "
            ></div>

            <div class="fairness-center"></div>

            <div
                class="fairness-marker"
                style="left: {marker_position}%;"
            ></div>

        </div>

        <div class="fairness-values">
            <span>+{scale_limit:.2f}</span>
            <span>0.00</span>
            <span>-{scale_limit:.2f}</span>
        </div>

        <div class="fairness-result">
            <span style="left: {marker_position}%;">
                {residual:+.3f}
            </span>
        </div>

    </div>
    """)

st.set_page_config(page_title="NBA On-Court Production Trade Balancer", layout="wide", page_icon="BKN_Primary.svg")
st.markdown("""
<style>
.block-container {
    max-width: 1500px;
    padding-top: 4rem;
    padding-bottom: 3rem;
}

button[kind="primary"] {
    background-color: #F0F0F0 !important;
    color: #000000 !important;
    border: 1px solid #FFFFFF !important;
}

button[kind="primary"]:hover {
    background-color: #E6E6E6 !important;
    color: #000000 !important;
    border-color: #E6E6E6 !important;
}

button[kind="primary"] p {
    color: #000000 !important;
}

/* Match the reset control to the team-selection dropdowns. */
.st-key-reset_trade_button button {
    background-color: #262730 !important;
    color: #FFFFFF !important;
    border-color: #3D3E47 !important;
}

.st-key-reset_trade_button button:hover {
    background-color: #31323D !important;
    border-color: #555660 !important;
}

/* Initial evaluation outer card */
.st-key-initial_evaluation_card {
    background-color: #000000;
    border: 1px solid #333333;
    border-top: 3px solid #FFFFFF;
    border-radius: 6px;
    padding: 20px 22px;
}

/* Individual evaluation metric cards */
.st-key-eval_team_a,
.st-key-eval_residual,
.st-key-eval_team_b {
    background-color: #151515;
    border: 1px solid #333333;
    border-radius: 6px;
    padding: 12px 16px;
}


/* Fairness scale */
.fairness-scale {
    margin-top: 24px;
    margin-bottom: 18px;
    padding: 0 6px;
}

.fairness-labels {
    display: flex;
    justify-content: space-between;
    color: #A0A0A0;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 10px;
}

.fairness-track {
    position: relative;
    height: 8px;
    background-color: #333333;
    border-radius: 4px;
}

/* Gray region around zero = acceptable tolerance */
.fairness-tolerance {
    position: absolute;
    height: 100%;
    background-color: #777777;
}

/* Zero line */
.fairness-center {
    position: absolute;
    left: 50%;
    top: -5px;
    width: 2px;
    height: 18px;
    background-color: #FFFFFF;
    transform: translateX(-50%);
}

/* Current trade */
.fairness-marker {
    position: absolute;
    top: 50%;
    width: 16px;
    height: 16px;
    background-color: #FFFFFF;
    border: 3px solid #000000;
    outline: 1px solid #FFFFFF;
    border-radius: 50%;
    transform: translate(-50%, -50%);
}

.fairness-values {
    display: flex;
    justify-content: space-between;
    color: #777777;
    font-size: 0.70rem;
    margin-top: 7px;
}

.fairness-result {
    position: relative;
    height: 26px;
    margin-top: 3px;
}

.fairness-result span {
    position: absolute;
    transform: translateX(-50%);
    font-weight: 700;
    white-space: nowrap;
}

/* Project-information tabs */
.st-key-project_tabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
}

.st-key-project_tabs [data-baseweb="tab"] {
    background-color: #151515;
    border: 1px solid #333333;
    border-radius: 6px 6px 0 0;
    padding: 0.45rem 0.9rem;
}

.st-key-project_tabs [data-baseweb="tab"] p {
    color: #D0D0D0 !important;
    font-weight: 600;
}

.st-key-project_tabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: #F0F0F0;
    border-color: #F0F0F0;
}

.st-key-project_tabs [data-baseweb="tab"][aria-selected="true"] p {
    color: #000000 !important;
}

.st-key-project_tabs [class*="st-key-project_section_"] {
    background-color: #111111;
    border: 1px solid #333333;
    border-top: 3px solid #FFFFFF;
    border-radius: 6px;
    padding: 18px 22px 22px;
    margin-top: 0.75rem;
}

.st-key-project_tabs [class*="st-key-project_section_"] h3 {
    color: #FFFFFF !important;
    margin-top: 0;
    padding-top: 0;
}

</style>
""", unsafe_allow_html=True)


st.html("""
<style>

.st-key-header_container {
    border: 2px solid #FFFFFF;
    background-color: #111111;
    border-radius: 8px;
    padding: 12px 20px;
    
}

/* General card */
.dashboard-card {
    background-color: #000000;
    border: 1px solid #333333;
    border-radius: 6px;
    padding: 18px 20px;
}

/* Streamlit containers that we explicitly key */
.st-key-team_a_card,
.st-key-team_b_card {
    background-color: #000000;
    border: 1px solid #333333;
    border-top: 3px solid #FFFFFF;
    border-radius: 6px;
    padding: 18px 20px;
}

/* Evaluation metric cards */
.st-key-eval_a,
.st-key-eval_balance,
.st-key-eval_b {
    background-color: #000000;
    border: 1px solid #333333;
    border-radius: 6px;
    padding: 12px 18px;
}

/* Final proposal */
.st-key-balanced_trade_card {
    background-color: #000000;
    border: 1px solid #333333;
    border-top: 3px solid #FFFFFF;
    border-radius: 6px;
    padding: 20px 22px;
}

.st-key-balanced_team_a,
.st-key-balanced_team_b {
    background-color: #151515;
    border: 1px solid #333333;
    border-radius: 6px;
    padding: 16px 18px;
}

.st-key-balance_status {
    background-color: #151515;
    border: 1px solid #555555;
    border-radius: 6px;
    padding: 12px 16px;
    text-align: center;
}

/* Force text inside the inverted result card to black */
.st-key-balanced_proposal p,
.st-key-balanced_proposal h1,
.st-key-balanced_proposal h2,
.st-key-balanced_proposal h3,
.st-key-balanced_proposal span {
    color: #000000 !important;
}

div[role="radiogroup"]
label[data-baseweb="radio"] > div:first-child {
    outline: 1px solid #777777 !important;
    outline-offset: -1px;
    border-radius: 50% !important;
}

div[role="radiogroup"]
label[data-baseweb="radio"]:has(input:checked) > div:first-child {
    outline: 1px solid #FFFFFF !important;
    outline-offset: -1px;
    border-radius: 50% !important;
}

.st-key-trade_warning {
    background-color: #151515;
    border: 1px solid #444444;
    border-left: 4px solid #FFFFFF;
    border-radius: 6px;
    padding: 14px 16px;
}

[data-testid="stAppViewContainer"] {
    background-color: #171717;

    background-image:
        radial-gradient(
            ellipse at center,
            rgba(0,0,0,0.95) 0px,
            rgba(0,0,0,0.95) 1.3px,
            transparent 1.5px
        ),
        radial-gradient(
            ellipse at center,
            rgba(0,0,0,0.95) 0px,
            rgba(0,0,0,0.95) 1.3px,
            transparent 1.5px
        );

    background-size: 10px 12px;
    background-position:
        0px 0px,
        5px 6px;
}

/* Expander container */
[data-testid="stExpander"] {
    background-color: #000000 !important;
    border: 1px solid #333333 !important;
    border-radius: 6px !important;
}

/* Expander header */
[data-testid="stExpander"] summary {
    background-color: #000000 !important;
}

/* Expanded content */
[data-testid="stExpander"] details {
    background-color: #000000 !important;
}

</style>
""")

with st.container(
    horizontal=True,
    vertical_alignment="center",
    horizontal_alignment="left",
    gap="small",
    border=True,
    key="header_container"
):
    st.image("BKN_Primary.svg", width=120)
    st.title("NBA On-Court Production Trade Balancer")


st.caption("On-court production model: current-season production is adjusted for role and sample reliability. " \
"Draft picks are valued by the amount of on-court production historically exchanged for comparable draft compensation. " \
"This is not complete NBA market value. Current pick ownership, swaps, protections, and Stepien restrictions are out of scope. " \
"2025-26 end-of-season roster snapshot.  Waived/released players included in rosters as waivers not available in dataset.")


try:
    (
        rosters,
        pick_inventory,
        hybrid_package_catalogue,
        pick_value_table,
        outright_pick_hierarchy,
        outright_pick_net_columns,
        outright_pick_weights,
        additional_player_slot_cost,
        model_bundle,
    ) = load_trade_data()
except Exception as error:
    st.error(f"Unable to initialize trade balancer: {error}")
    st.code(f"Imported tradeBalancer module:\n{Path(tb.__file__).resolve()}")
    st.stop()

reliability_constant = float(
    model_bundle["reliability_shrinkage_constant"]
)
role_full_mpg = float(
    model_bundle["role_capacity_full_minutes_per_game"]
)
role_exponent = float(model_bundle["role_capacity_exponent"])
pick_calibration = model_bundle.get("pick_scale_calibration", {})
regularization_strength = float(
    pick_calibration.get("selected_regularization_strength", np.nan)
)
selected_pick_scale = float(
    model_bundle.get("selected_pick_curve_scale", np.nan)
)





teams = sorted(
    rosters[TEAM_COLUMN].dropna().astype(str).unique().tolist()
)
if len(teams) < 2:
    st.error("The roster file must contain at least two teams.")
    st.stop()

# Clear trade inputs and stored results without reloading cached model data.
def reset_trade():
    st.session_state["team_a"] = teams[0]
    st.session_state["team_b"] = teams[1]
    st.session_state["team_a_players"] = []
    st.session_state["team_b_players"] = []
    st.session_state["team_a_picks"] = []
    st.session_state["team_b_picks"] = []
    st.session_state["balancing_approach"] = "Player matching"

    result_keys = [
        "evaluation_result",
        "evaluation_signature",
        "trade_result",
        "balance_signature",
    ]
    for key in result_keys:
        st.session_state.pop(key, None)



team_col_a, team_col_b = st.columns(2, gap="small")

# Create persistent container references
with team_col_a:
    team_a_card = st.container(key="team_a_card")

with team_col_b:
    team_b_card = st.container(key="team_b_card")

with team_a_card:
    st.markdown("### TEAM A")

    team_a_name = st.selectbox("Team", teams, index=0, key="team_a",)

team_b_options = [team for team in teams if team != team_a_name]

with team_b_card:
    st.markdown("### TEAM B")

    team_b_name = st.selectbox("Team", team_b_options, index=0, key="team_b",)



selected_pick_inventory = pick_inventory.loc[
    pick_inventory["owning_team"].astype(str).isin(
        [team_a_name, team_b_name]
    )
].copy()

selected_team_rows = rosters.loc[
    rosters[TEAM_COLUMN].astype(str).isin([team_a_name, team_b_name])
].copy()

ineligible_players = selected_team_rows.loc[
    ~selected_team_rows[ELIGIBLE_COLUMN].fillna(False)
].copy()
if not ineligible_players.empty:
    with st.expander(
        "Players excluded from production scoring "
        f"for the selected teams ({len(ineligible_players)})"
    ):
        st.caption(
            "These players do not meet the 100-minute, positive-game, "
            "positive-possession, and complete finalized-production-input "
            "requirements."
        )
        st.dataframe(
            ineligible_players.rename(
                columns={
                    PLAYER_NAME_COLUMN: "Player",
                    TEAM_COLUMN: "Team",
                    REASON_COLUMN: "Reason",
                }
            )[["Player", "Team", "Reason"]],
            width="stretch",
            hide_index=True,
        )

eligible_rows = selected_team_rows.loc[
    selected_team_rows[ELIGIBLE_COLUMN].fillna(False)
    & pd.to_numeric(
        selected_team_rows[PLAYER_VALUE_COLUMN], errors="coerce"
    ).notna()
].copy()

team_a_roster = sorted(
    eligible_rows.loc[
        eligible_rows[TEAM_COLUMN].astype(str).eq(team_a_name),
        PLAYER_NAME_COLUMN,
    ].astype(str).unique().tolist()
)
team_b_roster = sorted(
    eligible_rows.loc[
        eligible_rows[TEAM_COLUMN].astype(str).eq(team_b_name),
        PLAYER_NAME_COLUMN,
    ].astype(str).unique().tolist()
)
if not team_a_roster or not team_b_roster:
    st.warning(
        "One selected team has no production-eligible players. "
        "A picks-only package can still be evaluated for that team."
    )

available_initial_picks = selected_pick_inventory.loc[
    selected_pick_inventory["currently_owned"].fillna(False).astype(bool)
    & selected_pick_inventory["available_for_trade"].fillna(False).astype(bool)
].copy()
team_a_pick_inventory = available_initial_picks.loc[
    available_initial_picks["owning_team"].astype(str).eq(team_a_name)
].sort_values(["draft_year", "round", "pick_id"])
team_b_pick_inventory = available_initial_picks.loc[
    available_initial_picks["owning_team"].astype(str).eq(team_b_name)
].sort_values(["draft_year", "round", "pick_id"])

with team_a_card:

    team_a_players = st.multiselect("Players in proposed trade", team_a_roster, key="team_a_players",)

    team_a_initial_pick_ids = st.multiselect(
        "Draft picks in proposed trade",
        team_a_pick_inventory["pick_id"]
        .astype(str)
        .tolist(),
        format_func=lambda pick_id: format_pick_option(
            pick_id,
            team_a_pick_inventory,
        ),
        key="team_a_picks",
    )


with team_b_card:

    team_b_players = st.multiselect("Players in proposed trade", team_b_roster, key="team_b_players",)

    team_b_initial_pick_ids = st.multiselect(
        "Draft picks in proposed trade",
        team_b_pick_inventory["pick_id"]
        .astype(str)
        .tolist(),
        format_func=lambda pick_id: format_pick_option(
            pick_id,
            team_b_pick_inventory,
        ),
        key="team_b_picks",
    )

with st.container(key="eval_balance"):

    mode_label = st.radio(
        "Balancing approach",
        ["Player matching", "Best fit", "Picks first"],
        horizontal=True,
        key="balancing_approach",
    )

    mode_map = {
        "Player matching": "player_matching",
        "Best fit": "best_fit",
        "Picks first": "picks_first",
    }

    if mode_label == "Player matching":
        st.caption(
            "Ranks within-tolerance trades by the closest "
            "player-to-player production matches."
        )
    elif mode_label == "Best fit":
        st.caption(
            "Ranks all improving player-and-pick combinations by the "
            "smallest final imbalance."
        )
    else:
        st.caption(
            "Uses available picks before considering additional players."
        )


    team_a_has_assets = bool(
        team_a_players or team_a_initial_pick_ids
    )
    team_b_has_assets = bool(
        team_b_players or team_b_initial_pick_ids
    )

    can_balance = team_a_has_assets and team_b_has_assets

    trade_signature = (
        team_a_name,
        team_b_name,
        tuple(team_a_players),
        tuple(team_b_players),
        tuple(team_a_initial_pick_ids),
        tuple(team_b_initial_pick_ids),
    )

    balance_signature = (
        *trade_signature,
        mode_map[mode_label],
    )


    evaluate_col, balance_col = st.columns(2)

    with evaluate_col:
        evaluate_clicked = st.button(
            "Evaluate trade",
            disabled=not can_balance,
            width="stretch",
        )

    with balance_col:
        balance_clicked = st.button(
            "Balance trade",
            disabled=not can_balance,
            type="primary",
            width="stretch",
        )

    st.button("Reset trade", on_click=reset_trade, width="stretch", key="reset_trade_button")

if evaluate_clicked:
    try:
        team_a_initial_pick_assets = tb.resolve_initial_pick_assets(
            pick_inventory=pick_inventory,
            team_name=team_a_name,
            selected_pick_ids=team_a_initial_pick_ids,
            outright_pick_hierarchy=outright_pick_hierarchy,
        )
        team_b_initial_pick_assets = tb.resolve_initial_pick_assets(
            pick_inventory=pick_inventory,
            team_name=team_b_name,
            selected_pick_ids=team_b_initial_pick_ids,
            outright_pick_hierarchy=outright_pick_hierarchy,
        )
        initial_draft_counts = tb.build_initial_draft_counts(
            team_a_initial_pick_assets=team_a_initial_pick_assets,
            team_b_initial_pick_assets=team_b_initial_pick_assets,
            outright_pick_hierarchy=outright_pick_hierarchy,
            outright_pick_net_columns=outright_pick_net_columns,
        )
        initial_pick_adjustment = tb.calculate_initial_pick_adjustment(
            team_a_initial_pick_assets,
            team_b_initial_pick_assets,
        )
        st.session_state["evaluation_result"] = (
            tb.score_two_team_multi_player_trade(
                roster_data=rosters,
                team_a_player_names=team_a_players,
                team_b_player_names=team_b_players,
                team_a_draft_counts=initial_draft_counts,
                player_value_column=PLAYER_VALUE_COLUMN,
                outright_pick_net_columns=outright_pick_net_columns,
                outright_pick_weights=outright_pick_weights,
                team_a_name=team_a_name,
                team_b_name=team_b_name,
                additional_player_slot_cost=additional_player_slot_cost,
                team_a_itemized_pick_adjustment=initial_pick_adjustment,
                team_a_initial_pick_assets=team_a_initial_pick_assets,
                team_b_initial_pick_assets=team_b_initial_pick_assets,
                team_column=TEAM_COLUMN,
                player_name_column=PLAYER_NAME_COLUMN,
            )
        )
        st.session_state["evaluation_signature"] = trade_signature
    except Exception as error:
        st.exception(error)

if balance_clicked:
    try:
        with st.spinner("Evaluating trade options..."):
            trade_result = tb.balance_trade(
                roster_data=rosters,
                pick_inventory=pick_inventory,
                team_a_name=team_a_name,
                team_b_name=team_b_name,
                team_a_player_names=team_a_players,
                team_b_player_names=team_b_players,
                player_value_column=PLAYER_VALUE_COLUMN,
                additional_player_slot_cost=additional_player_slot_cost,
                pick_value_table=pick_value_table,
                outright_pick_hierarchy=outright_pick_hierarchy,
                outright_pick_net_columns=outright_pick_net_columns,
                outright_pick_weights=outright_pick_weights,
                team_a_initial_pick_ids=team_a_initial_pick_ids,
                team_b_initial_pick_ids=team_b_initial_pick_ids,
                priority_mode=mode_map[mode_label],
                balance_tolerance=BALANCE_TOLERANCE,
                hybrid_package_catalogue=hybrid_package_catalogue,
                team_column=TEAM_COLUMN,
                player_name_column=PLAYER_NAME_COLUMN,
            )
            st.session_state["trade_result"] = trade_result
            st.session_state["balance_signature"] = balance_signature
            st.session_state["evaluation_result"] = trade_result[
                "initial_score"
            ]
            st.session_state["evaluation_signature"] = trade_signature
    except Exception as error:
        st.exception(error)


evaluation_result = st.session_state.get("evaluation_result")
evaluation_signature = st.session_state.get("evaluation_signature")

if evaluation_result is not None and evaluation_signature == trade_signature:
    residual = float(evaluation_result["team_a_symmetric_residual"])
    absolute_residual = abs(residual)
    favored_team = (
        "Neither: Balanced"
        if absolute_residual <= BALANCE_TOLERANCE
        else (team_a_name if residual > 0 else team_b_name)
    )

    team_a_player_value = float(evaluation_result["team_a_player_package_value_sent"])
    team_b_player_value = float(evaluation_result["team_b_player_package_value_sent"])
    team_a_pick_value = float(evaluation_result.get("team_a_initial_pick_value_sent", 0.0))
    team_b_pick_value = float(evaluation_result.get("team_b_initial_pick_value_sent", 0.0))
    net_pick_value = float(evaluation_result.get("team_a_draft_adjustment", 0.0))
    player_match_cost = float(evaluation_result.get("player_match_cost", 0.0))
    unmatched_player_value = float(evaluation_result.get("unmatched_player_value", 0.0))

    st.divider()

    with st.container(key="initial_evaluation_card"):
        st.markdown("## ON-COURT PRODUCTION EVALUATION")

        metric_a, metric_residual, metric_b = st.columns(3, gap="small")

        with metric_a:
            with st.container(key="eval_team_a"):
                st.metric(
                    f"{team_a_name} production value",
                    f"{team_a_player_value:.3f}",
                )

        with metric_residual:
            with st.container(key="eval_residual"):
                st.metric("Production-value imbalance", f"{residual:+.3f}")

        with metric_b:
            with st.container(key="eval_team_b"):
                st.metric(
                    f"{team_b_name} production value",
                    f"{team_b_player_value:.3f}",
                )

        render_fairness_scale(
            residual=residual,
            team_a_name=team_a_name,
            team_b_name=team_b_name,
            tolerance=BALANCE_TOLERANCE,
            scale_limit=1.5,
        )

        if absolute_residual <= BALANCE_TOLERANCE:
            with st.container(key="trade_warning"):
                            st.write("The proposal is within the model's on-court production tolerance.")
        else:
            with st.container(key="trade_warning"):
                st.write(f"{favored_team} receives {absolute_residual:.3f} more production-equivalent value."
    )

    

trade_result = st.session_state.get("trade_result")
stored_balance_signature = st.session_state.get("balance_signature")
if trade_result is not None and stored_balance_signature == balance_signature:
    proposal = extract_balanced_proposal(trade_result)

    initial_residual = float(trade_result["initial_residual"])
    final_residual = float(trade_result["final_residual"])
    improvement = abs(initial_residual) - abs(final_residual)
    within_tolerance = trade_result["within_balance_tolerance"]

    st.divider()

    with st.container(key="balanced_trade_card"):
        st.markdown("## ON-COURT BALANCED TRADE PROPOSAL")

        result_col_a, result_col_b = st.columns(2, gap="small")

        with result_col_a:
            with st.container(key="balanced_team_a"):
                render_team_package(
                    proposal["team_a_name"],
                    proposal["team_a_players"],
                    proposal["team_a_picks"],
                    player_package_value=proposal["team_a_player_package_value"],
                    pick_package_value=proposal["team_a_pick_package_value"],
                )

        with result_col_b:
            with st.container(key="balanced_team_b"):
                render_team_package(
                    proposal["team_b_name"],
                    proposal["team_b_players"],
                    proposal["team_b_picks"],
                    player_package_value=proposal["team_b_player_package_value"],
                    pick_package_value=proposal["team_b_pick_package_value"],
                )

        st.markdown("#### FINAL ON-COURT PRODUCTION BALANCE")

        render_fairness_scale(
            residual=final_residual,
            team_a_name=proposal["team_a_name"],
            team_b_name=proposal["team_b_name"],
            tolerance=BALANCE_TOLERANCE,
            scale_limit=1.5,
        )

        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Initial production balance", f"{initial_residual:+.3f}")
        metric_b.metric("Final production balance", f"{final_residual:+.3f}")
        metric_c.metric("Production-balance improvement", f"{improvement:.3f}")

        with st.container(key="balance_status"):
            st.markdown(
                "**✓ WITHIN ON-COURT PRODUCTION TOLERANCE**"
                if within_tolerance
                else "**OUTSIDE ON-COURT PRODUCTION TOLERANCE**"
            )

        st.markdown("#### MATCHING DETAILS")

        match_a, match_b, match_c = st.columns(3)
        match_a.metric("Player-match cost", f"{proposal['player_match_cost']:.3f}")
        match_b.metric("Unmatched player value", f"{proposal['unmatched_player_value']:.3f}")
        match_c.metric("Largest player-match gap", f"{proposal['maximum_player_match_gap']:.3f}")

        render_player_matches(
            proposal["player_match_pairs"],
            proposal["team_a_name"],
            proposal["team_b_name"],
            title="Selected player-to-player matches",
        )

        st.caption(
            f"Selected option: {proposal['option_type']} | "
            f"Mode: {trade_result['priority_mode']}"
        )

    render_ranked_options(trade_result)


st.divider()
st.markdown("## ABOUT THIS PROJECT")

parameter_rows = pd.DataFrame(
    [
        {"parameter": "Minimum season minutes", "value": model_bundle["minimum_season_minutes"]},
        {"parameter": "Reliability shrinkage constant", "value": reliability_constant},
        {"parameter": "Role-capacity full MPG", "value": role_full_mpg},
        {"parameter": "Role-capacity exponent", "value": role_exponent},
        {"parameter": "Additional-player slot cost", "value": additional_player_slot_cost},
        {"parameter": "Selected pick-scale regularization", "value": regularization_strength},
        {"parameter": "Selected top-five pick scale", "value": selected_pick_scale},
    ]
)

try:
    app_writeups = load_app_writeups(APP_WRITEUPS_PATH)
except OSError as error:
    st.warning(f"Project notes could not be loaded: {error}")
else:
    tab_labels = {
        "Project Overview": "Overview",
        "How the Math Works": "Math",
        "How to Interpret Results and Why Production Value Is Not Trade Value": "Interpreting Results",
        "Data and Historical Scope": "Data & Scope",
    }
    with st.container(key="project_tabs"):
        section_tabs = st.tabs([tab_labels.get(title, title) for title, _ in app_writeups])
        for index, (section_tab, (section_title, section_body)) in enumerate(zip(section_tabs, app_writeups)):
            with section_tab:
                with st.container(key=f"project_section_{index}"):
                    st.markdown(f"### {section_title}")
                    st.markdown(section_body)

                    if section_title == "How the Math Works":
                        with st.expander("Model parameters"):
                            st.dataframe(parameter_rows, width="stretch", hide_index=True)
                            st.dataframe(
                                pick_value_table[["tier", "estimated_value"]],
                                width="stretch",
                                hide_index=True,
                            )
    
