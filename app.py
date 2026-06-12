from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_predictor import load_artifacts, predict_match  # noqa: E402
from football_predictor.modeling import explanation_lines  # noqa: E402


st.set_page_config(page_title="International Football Predictor", page_icon="AI", layout="wide")


@st.cache_resource
def get_artifacts():
    return load_artifacts(ROOT / "models")


model, data = get_artifacts()
team_stats = data["team_stats"]
report = data.get("report", {})
team_names = sorted(team_stats.keys())

st.title("International Football Match Predictor")
st.caption("IBM SkillsBuild AI Builders Challenge June 2026 prototype")

left, right = st.columns([0.42, 0.58], gap="large")

with left:
    st.subheader("Match setup")
    team_a = st.selectbox(
        "Team A / home side",
        team_names,
        index=team_names.index("Brazil") if "Brazil" in team_names else 0,
    )
    team_b = st.selectbox(
        "Team B / away side",
        team_names,
        index=team_names.index("Argentina") if "Argentina" in team_names else min(1, len(team_names) - 1),
    )
    neutral = st.toggle("Neutral venue", value=True)
    major = st.toggle("Major tournament context", value=True)
    run_prediction = st.button("Predict outcome", type="primary", use_container_width=True)

    st.divider()
    st.metric("Teams available", f"{len(team_names):,}")
    st.metric("Held-out test accuracy", f"{report.get('accuracy', 0):.1%}")
    st.metric("Baseline accuracy", f"{report.get('baseline_accuracy', 0):.1%}")

with right:
    st.subheader("Prediction")
    if run_prediction:
        if team_a == team_b:
            st.error("Pick two different teams.")
        else:
            result = predict_match(
                team_a,
                team_b,
                model=model,
                team_stats=team_stats,
                is_neutral=neutral,
                is_major_tournament=major,
            )
            probabilities = result["probabilities"]
            st.success(f"Predicted outcome: {result['predicted_outcome']}")

            col_a, col_draw, col_b = st.columns(3)
            col_a.metric(f"{team_a} win", f"{probabilities['team_a_win']:.1%}")
            col_draw.metric("Draw", f"{probabilities['draw']:.1%}")
            col_b.metric(f"{team_b} win", f"{probabilities['team_b_win']:.1%}")

            chart = pd.DataFrame(
                {
                    "Outcome": [f"{team_a} win", "Draw", f"{team_b} win"],
                    "Probability": [
                        probabilities["team_a_win"],
                        probabilities["draw"],
                        probabilities["team_b_win"],
                    ],
                }
            )
            st.bar_chart(chart, x="Outcome", y="Probability", height=260)

            st.subheader("Explanation")
            for line in explanation_lines(team_a, team_b, team_stats, report):
                st.write(f"- {line}")

            st.subheader("Team stats used by the model")
            comparison = pd.DataFrame(
                {
                    team_a: team_stats[team_a],
                    team_b: team_stats[team_b],
                }
            )
            st.dataframe(comparison, use_container_width=True)
    else:
        st.info("Choose two teams and run a prediction.")

