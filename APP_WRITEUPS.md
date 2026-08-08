# Application Writeups

## Project Overview

This project is a production-balance tool for exploring hypothetical NBA trades. It places current-season player production and projected draft compensation on a common scale, evaluates the assets exchanged by each team, and can recommend additional players or picks that move an imbalanced proposal closer to balance.

The project's greatest potential value would come from expanding this framework into a full-scale trade-balancing system. That would require quantifying the factors currently outside its scope, such as contracts, age, team control, health, positional fit, prospect upside, salary-cap effects, and detailed draft-pick conditions, and combining them with the existing production and draft-compensation measures.

Player value combines scoring load, scoring efficiency, playmaking, rebounding, defensive events, and plus-minus. Small statistical samples and limited roles are discounted. Draft-pick values are estimated from completed historical trades in which draft compensation offset measurable player production.

The application supports three balancing approaches:

- **Player Matching:** prioritizes similarly valued players before adding picks.
- **Best Fit:** searches player, pick, and combined packages for the closest balance.
- **Picks First:** attempts to resolve the imbalance with picks before adding players.

### In scope

- Current-season on-court production for players with at least 100 minutes
- First- and second-round picks grouped into five projected tiers
- Single-player and multi-player packages
- Evaluation of a proposed trade without modifying it
- Suggested packages of up to two additional players per team, three additional players overall, and five additional picks

### Out of scope

- Contracts and salary matching
- Player age, injuries, upside, and team control
- Positional need, roster fit, and coaching context
- Luxury-tax and other financial considerations
- Pick protections, swaps, and Stepien Rule restrictions
- Verified live pick ownership
- Predictions of future wins
- Determining whether a trade is realistic or advisable

### Strengths

The model uses only information that would have been available at the transaction date when calibrating historical values. It places players and picks on a transparent common scale and explicitly adjusts for sample size, demonstrated role, and the diminishing value of adding multiple players to one package.

Its precomputed package catalogue allows the application to search hundreds of thousands of possible player-and-pick combinations quickly.

### Limitations

On-court production is only one part of NBA trade value. Two equally productive players can have dramatically different markets because of age, contract, health, upside, or team context. Draft tiers are projections rather than guarantees, and historical trades do not provide one objectively correct answer for every hypothetical proposal.

The application should therefore be treated as a decision-support and exploration tool, not a complete trade-value authority.

## Methodology

The project began by cleaning NBA transaction data from 1976 through 2019 and matching traded-player names to historical box scores. Player identities were manually reviewed where normalization was required, with suffixes such as Jr. and Sr. preserved to prevent different careers from being combined.

For every usable transaction, player statistics were calculated using only information available before the transaction date. These point-in-time snapshots included season, career, and recent production. The original project used those features to predict post-trade win-percentage change, but XGBoost produced almost no improvement over an average-change baseline. The project therefore pivoted from outcome prediction to measuring the value exchanged within completed trades.

Player production was standardized against the historical population and combined into scoring load, scoring efficiency, playmaking, rebounding and defensive events, and plus-minus impact. Reliability and role adjustments were added to prevent small-sample, low-minute performances from being valued like established production. Multi-player packages were then adjusted for the diminishing value of additional roster slots.

Draft picks were classified into five projected tiers and calibrated against historical trades in which draft compensation offset measurable player production. The final application applies the resulting player and pick values to the 2025-26 roster snapshot, evaluates hypothetical trades, and searches a precomputed catalogue for player and pick packages that reduce the on-court production imbalance.

The completed application was checked using scenarios ranging from star-level exchanges to marginal rotation-player trades. These tests were used to confirm that the calculations and balancing recommendations behaved reasonably across different parts of the production-value distribution.

## How the Math Works

### 1. Standardize production

Each production metric is converted to a historical z-score:

`z = (player result - historical mean) / historical standard deviation`

Extreme z-scores are capped at **±3**, preventing one unusual statistic from dominating the calculation.

### 2. Calculate player quality

The composite quality score is:

`Q = 30% shooting load + 20% scoring efficiency + 25% playmaking + 10% rebounding/defensive events + 15% plus-minus`

Component details:

- Playmaking: **⅔ assists per 100 − ⅓ turnovers per 100**
- Rebounding and defense: equal parts rebound percentage, steal percentage, and block percentage
- Scoring efficiency: points added relative to a **52.48% historical true-shooting reference**, adjusted for shooting volume
- Overall impact: plus-minus per 100 possessions

### 3. Adjust for sample size

Reliability is based on estimated player possessions:

`Reliability = possessions / (possessions + 250)`

At 250 possessions, a player retains 50% of the measured quality adjustment. Larger samples retain progressively more.

### 4. Adjust for demonstrated role

Role capacity is based on minutes per game:

`Role capacity = min(MPG / 30, 1) ^ 0.5`

A player reaches full role capacity at **30 minutes per game**. The square-root adjustment gives legitimate rotation players meaningful credit while still discounting very small roles.

The role-scaled baseline is:

`Base = 0.2 + 0.8 × role capacity`

Final player value is:

`Player value = max(0, Base + 0.5 × Reliability × Role capacity × Q)`

Players with fewer than **100 season minutes** are not assigned a calculated production value.

### 5. Calculate package value

The most valuable player in a package contributes their full value. Each additional player contributes:

`max(player value - 0.50, 0)`

The **0.50 additional-player slot cost** prevents several marginal players from automatically becoming more valuable than one high-level player.

### 6. Value draft compensation

Draft picks use five projected tiers:

| Pick tier | Model value |
|---|---:|
| Top-five first | 1.355 |
| Lottery first | 1.144 |
| Late first | 0.561 |
| Early second | 0.229 |
| Late second | 0.229 |

The overall pick curve was calibrated using **313 historical development trades** and a regularization strength of **0.3**.

The calibration used **500 bootstrap samples**. The middle 95% of the fitted top-five scale ranged from approximately **1.232 to 1.493**.

Future pick projections are discounted for uncertainty:

- 2027: **100% confidence**
- 2028: **85% confidence**
- 2029: **65% confidence**

### 7. Score the trade

For each team:

`Residual = player value received - player value sent + draft value received - draft value sent`

- A positive residual indicates a modeled advantage.
- A negative residual indicates that additional compensation is required.
- An absolute residual of **0.05 or less** is considered within the application's arithmetic tolerance.

## How to Interpret Results and Why Production Value Is Not Trade Value

The application answers a narrow question: **how balanced is this transaction when current-season player production and historically calibrated draft compensation are measured on the same scale?**

### Reading the results

- **Player package value** is the slot-adjusted production value of the players sent by each team.
- **Pick package value** is the modeled value of the selected projected draft assets.
- **Trade imbalance or residual** shows the difference between the two packages from the named team's perspective.
- **Positive residual:** the team receives more modeled production-equivalent value than it sends.
- **Negative residual:** the team sends more modeled production-equivalent value than it receives.
- **Within tolerance:** the absolute residual is no greater than 0.05. This means the trade is arithmetically balanced on the model's scale, not necessarily fair in the real NBA market.
- **Balance improvement** is the reduction in the absolute residual after the application adds a suggested package.
- **Player-match cost** summarizes the production-value gaps between paired players. Lower values indicate closer individual matches.
- **Unmatched player value** is the value belonging to players who could not be paired directly across the two packages.

The three balancing modes answer slightly different questions:

- **Player Matching:** which additional players create the closest individual production matches, with picks used where needed?
- **Best Fit:** which allowed combination of players and picks produces the smallest overall residual?
- **Picks First:** can draft compensation resolve the imbalance before more players are introduced?

### Why production value is not trade value

Production value measures what a player has demonstrated on the court. Actual trade value also depends on information the model deliberately excludes, including:

- Age and expected development or decline
- Contract salary, length, options, and guarantees
- Team control and free-agency status
- Injury history and current health
- Positional scarcity and roster fit
- Playoff utility and matchup considerations
- Prospect upside and organizational timelines
- Salary-cap, luxury-tax, and ownership constraints
- Pick protections, swaps, and legal trade restrictions

A young player on a low-cost contract and an older veteran can receive similar production values while having very different trade markets. A rebuilding team can also rationally trade away current production for future value, even if the application identifies an immediate production disadvantage.

The output is therefore best interpreted as one analytical lens: **production-equivalent balance**, not a definitive judgment about which team wins a trade.

## Data and Historical Scope

### Current application season

Player production displayed in the finished application is calculated from the **2025-26 NBA regular season**. The current-roster dataset contains 587 players, of whom 503 meet the production model's eligibility requirements.

Players must have at least **100 minutes**, positive games and possessions, and the required finalized production inputs to receive a calculated value. Players who do not qualify remain outside the production calculation rather than being treated as demonstrated zero-production players.

### Historical transaction scope

The source transaction data spans **1976 through 2019**, but not every season was usable for player-production modeling.

The following seasons were excluded:

- 1976-77 through 1984-85, because the available box scores did not consistently separate offensive and defensive rebounds required for the percentage calculations
- 2000-01, because team-stat data was missing for **597 of 1,189 regular-season games**, or approximately **50.2%** of the season

The usable historical production and transaction-calibration period therefore runs from **1985-86 through 2018-19**, excluding **2000-01**.

Historical player statistics were calculated using only games played before each transaction date. Depending on the transaction timing, the workflow constructed season-to-date, career-to-date, last-10, or previous-season measurements without using future player performance.

### Draft scope

Only first- and second-round picks receive modeled value. Picks after the second round are assigned zero value because the modern NBA draft contains two rounds.

Historical picks were classified using the relevant team's record at the transaction date:

- Current-season record for in-season transactions
- Previous-season record for offseason transactions

The finished application uses projected own-pick tiers for 2027, 2028, and 2029. These projections are manually maintained assumptions rather than verified live pick ownership records. Protections, swaps, conveyance conditions, and Stepien Rule restrictions remain outside the application scope.
