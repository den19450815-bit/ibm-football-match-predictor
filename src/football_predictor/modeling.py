from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


FEATURE_COLUMNS = [
    "team_a_winrate",
    "team_b_winrate",
    "team_a_goal_avg",
    "team_b_goal_avg",
    "team_a_recent_form",
    "team_b_recent_form",
    "is_neutral",
    "is_major_tournament",
]

OUTCOME_LABELS = {
    0: "Team A win",
    1: "Draw",
    2: "Team B win",
}

MAJOR_TOURNAMENTS = {
    "Soccer World Cup",
    "Soccer World Cup qualification",
    "UEFA Euro",
    "UEFA Euro qualification",
    "Copa América",
    "African Cup of Nations",
    "African Cup of Nations qualification",
    "AFC Asian Cup",
    "AFC Asian Cup qualification",
    "Gold Cup",
    "CONCACAF Nations League",
    "UEFA Nations League",
}


def load_matches(csv_path: str | Path = "data/results.csv") -> pd.DataFrame:
    """Load the Kaggle international football results data."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Place results.csv in data/ or run from the project root."
        )

    matches = pd.read_csv(path, parse_dates=["date"])
    required = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    }
    missing = required.difference(matches.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if matches["neutral"].dtype != bool:
        matches["neutral"] = (
            matches["neutral"].astype(str).str.strip().str.upper().map({"TRUE": True, "FALSE": False})
        )
    return matches.dropna(subset=["home_score", "away_score"]).sort_values("date").reset_index(drop=True)


def _winrate(history: list[tuple[int, int, int]]) -> float:
    return sum(won for _, _, won in history) / len(history) if history else 0.5


def _goal_avg(history: list[tuple[int, int, int]]) -> float:
    return sum(goals_for for goals_for, _, _ in history) / len(history) if history else 1.0


def _recent_form(history: list[tuple[int, int, int]], window: int = 10) -> float:
    recent = history[-window:]
    return sum(won for _, _, won in recent) / len(recent) if len(recent) == window else 0.5


def build_training_frame(matches: pd.DataFrame, start_date: str = "1990-01-01") -> pd.DataFrame:
    """Create leak-resistant chronological features from prior team results."""
    recent_matches = matches[matches["date"] >= pd.Timestamp(start_date)].sort_values("date").reset_index(drop=True)
    team_history: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []

    for _, row in recent_matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_history = team_history[home]
        away_history = team_history[away]

        if row["home_score"] > row["away_score"]:
            outcome, home_won, away_won = 0, 1, 0
        elif row["home_score"] < row["away_score"]:
            outcome, home_won, away_won = 2, 0, 1
        else:
            outcome, home_won, away_won = 1, 0, 0

        rows.append(
            {
                "date": row["date"],
                "team_a": home,
                "team_b": away,
                "team_a_winrate": _winrate(home_history),
                "team_b_winrate": _winrate(away_history),
                "team_a_goal_avg": _goal_avg(home_history),
                "team_b_goal_avg": _goal_avg(away_history),
                "team_a_recent_form": _recent_form(home_history),
                "team_b_recent_form": _recent_form(away_history),
                "is_neutral": int(bool(row["neutral"])),
                "is_major_tournament": int(row["tournament"] in MAJOR_TOURNAMENTS),
                "outcome": outcome,
            }
        )

        team_history[home].append((int(row["home_score"]), int(row["away_score"]), home_won))
        team_history[away].append((int(row["away_score"]), int(row["home_score"]), away_won))

    return pd.DataFrame(rows)


def build_team_stats(matches: pd.DataFrame, min_matches: int = 30) -> dict[str, dict[str, float | int]]:
    """Build current team summaries used by the prototype for new matchups."""
    world_cup_qualifiers = matches[matches["tournament"] == "Soccer World Cup qualification"]
    eligible_teams = set(pd.concat([world_cup_qualifiers["home_team"], world_cup_qualifiers["away_team"]]).unique())
    if not eligible_teams:
        eligible_teams = set(pd.concat([matches["home_team"], matches["away_team"]]).unique())

    team_stats: dict[str, dict[str, float | int]] = {}
    all_teams = sorted(set(pd.concat([matches["home_team"], matches["away_team"]]).unique()))
    for team in all_teams:
        if team not in eligible_teams:
            continue

        home = matches[matches["home_team"] == team]
        away = matches[matches["away_team"] == team]
        total = len(home) + len(away)
        if total < min_matches:
            continue

        home_wins = int((home["home_score"] > home["away_score"]).sum())
        away_wins = int((away["away_score"] > away["home_score"]).sum())
        total_goals = int(home["home_score"].sum() + away["away_score"].sum())

        last10 = matches[(matches["home_team"] == team) | (matches["away_team"] == team)].sort_values("date").tail(10)
        recent_wins = 0
        for _, row in last10.iterrows():
            if row["home_team"] == team and row["home_score"] > row["away_score"]:
                recent_wins += 1
            elif row["away_team"] == team and row["away_score"] > row["home_score"]:
                recent_wins += 1

        team_stats[team] = {
            "winrate": (home_wins + away_wins) / total,
            "goal_avg": total_goals / total,
            "recent_form": recent_wins / len(last10) if len(last10) == 10 else 0.5,
            "matches_played": total,
        }

    return team_stats


def train_model(features: pd.DataFrame, cutoff: str = "2018-01-01") -> tuple[RandomForestClassifier, dict[str, Any]]:
    train_mask = features["date"] < pd.Timestamp(cutoff)
    test_mask = features["date"] >= pd.Timestamp(cutoff)

    x_train = features.loc[train_mask, FEATURE_COLUMNS]
    x_test = features.loc[test_mask, FEATURE_COLUMNS]
    y_train = features.loc[train_mask, "outcome"]
    y_test = features.loc[test_mask, "outcome"]

    model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    majority_class = y_train.value_counts().idxmax()
    report = {
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "baseline_accuracy": float((y_test == majority_class).mean()),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1, 2]).tolist(),
        "feature_importance": dict(zip(FEATURE_COLUMNS, map(float, model.feature_importances_))),
        "cutoff": cutoff,
    }
    return model, report


def save_artifacts(
    model: RandomForestClassifier,
    team_stats: dict[str, dict[str, float | int]],
    report: dict[str, Any],
    model_dir: str | Path = "models",
) -> None:
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / "match_predictor.pkl", compress=3)
    joblib.dump(
        {
            "team_stats": team_stats,
            "feature_cols": FEATURE_COLUMNS,
            "outcome_labels": OUTCOME_LABELS,
            "report": report,
        },
        path / "team_data.pkl",
    )


def train_and_save(
    csv_path: str | Path = "data/results.csv",
    model_dir: str | Path = "models",
    start_date: str = "1990-01-01",
    cutoff: str = "2018-01-01",
) -> dict[str, Any]:
    matches = load_matches(csv_path)
    features = build_training_frame(matches, start_date=start_date)
    model, report = train_model(features, cutoff=cutoff)
    team_stats = build_team_stats(matches)
    save_artifacts(model, team_stats, report, model_dir=model_dir)
    return {
        **report,
        "match_rows": int(len(matches)),
        "feature_rows": int(len(features)),
        "team_count": int(len(team_stats)),
    }


def load_artifacts(model_dir: str | Path = "models") -> tuple[RandomForestClassifier, dict[str, Any]]:
    path = Path(model_dir)
    model = joblib.load(path / "match_predictor.pkl")
    data = joblib.load(path / "team_data.pkl")
    return model, data


def make_feature_row(
    team_a: str,
    team_b: str,
    team_stats: dict[str, dict[str, float | int]],
    is_neutral: bool = True,
    is_major_tournament: bool = True,
) -> pd.DataFrame:
    if team_a not in team_stats:
        raise ValueError(f"Unknown team: {team_a!r}")
    if team_b not in team_stats:
        raise ValueError(f"Unknown team: {team_b!r}")
    if team_a == team_b:
        raise ValueError("Choose two different teams.")

    a = team_stats[team_a]
    b = team_stats[team_b]
    return pd.DataFrame(
        [
            {
                "team_a_winrate": a["winrate"],
                "team_b_winrate": b["winrate"],
                "team_a_goal_avg": a["goal_avg"],
                "team_b_goal_avg": b["goal_avg"],
                "team_a_recent_form": a["recent_form"],
                "team_b_recent_form": b["recent_form"],
                "is_neutral": int(is_neutral),
                "is_major_tournament": int(is_major_tournament),
            }
        ]
    )[FEATURE_COLUMNS]


def predict_match(
    team_a: str,
    team_b: str,
    model: RandomForestClassifier,
    team_stats: dict[str, dict[str, float | int]],
    is_neutral: bool = True,
    is_major_tournament: bool = True,
) -> dict[str, Any]:
    row = make_feature_row(team_a, team_b, team_stats, is_neutral, is_major_tournament)
    probabilities = model.predict_proba(row)[0]
    by_class = {int(label): float(prob) for label, prob in zip(model.classes_, probabilities)}

    result = {
        "team_a": team_a,
        "team_b": team_b,
        "probabilities": {
            "team_a_win": by_class.get(0, 0.0),
            "draw": by_class.get(1, 0.0),
            "team_b_win": by_class.get(2, 0.0),
        },
        "feature_values": row.iloc[0].to_dict(),
    }
    winner_key = max(result["probabilities"], key=result["probabilities"].get)
    result["predicted_outcome"] = {
        "team_a_win": f"{team_a} win",
        "draw": "Draw",
        "team_b_win": f"{team_b} win",
    }[winner_key]
    return result


def explanation_lines(
    team_a: str,
    team_b: str,
    team_stats: dict[str, dict[str, float | int]],
    report: dict[str, Any] | None = None,
) -> list[str]:
    a = team_stats[team_a]
    b = team_stats[team_b]
    lines = [
        f"{team_a} historical win rate: {a['winrate']:.1%}; {team_b}: {b['winrate']:.1%}.",
        f"{team_a} average goals: {a['goal_avg']:.2f}; {team_b}: {b['goal_avg']:.2f}.",
        f"Recent form over last 10 matches: {team_a} {a['recent_form']:.1%}; {team_b} {b['recent_form']:.1%}.",
    ]
    if report and report.get("feature_importance"):
        top_features = sorted(report["feature_importance"].items(), key=lambda item: item[1], reverse=True)[:3]
        readable = ", ".join(name.replace("_", " ") for name, _ in top_features)
        lines.append(f"The model's most influential features in this run are: {readable}.")
    return lines
