from __future__ import annotations

import json
import textwrap
import uuid
from pathlib import Path


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        """
        # International Football Match Predictor

        Completed notebook for the IBM SkillsBuild AI Builders Challenge June 2026 football lab. This notebook follows the official lab flow and uses reusable Python modules from this repository so the prototype can run both in Jupyter and as a Streamlit app.
        """
    ),
    markdown(
        """
        ## IBM Bob prompts used

        - Help me set up a Python/Jupyter environment for the IBM SkillsBuild football prediction lab.
        - Generate Python code to load and explore the international football results dataset.
        - Explain how to engineer leak-resistant team features from historical match results.
        - Debug model training errors and package import issues.
        - Help me create a Streamlit prototype that predicts match winner probabilities and explains the result.
        """
    ),
    code(
        """
        from pathlib import Path
        import sys

        project_root = Path.cwd()
        if not (project_root / "data" / "results.csv").exists():
            project_root = project_root.parent
        sys.path.insert(0, str(project_root / "src"))
        print(project_root)
        """
    ),
    markdown("## Load the dataset"),
    code(
        """
        from football_predictor import load_matches

        matches = load_matches(project_root / "data" / "results.csv")
        print(matches.shape)
        print(matches["date"].min(), matches["date"].max())
        matches.head()
        """
    ),
    markdown("## Explore tournaments and teams"),
    code(
        """
        import pandas as pd

        print("Top tournaments")
        print(matches["tournament"].value_counts().head(10).to_string())

        all_teams = pd.concat([matches["home_team"], matches["away_team"]])
        print("\\nTop teams by match count")
        print(all_teams.value_counts().head(15).to_string())
        """
    ),
    markdown("## Build chronological features"),
    code(
        """
        from football_predictor import build_training_frame

        features = build_training_frame(matches)
        print(features.shape)
        features.head()
        """
    ),
    markdown("## Train and evaluate the model"),
    code(
        """
        from football_predictor.modeling import train_model

        model, report = train_model(features)
        print(f"Accuracy: {report['accuracy']:.2%}")
        print(f"Baseline: {report['baseline_accuracy']:.2%}")
        print(report["confusion_matrix"])
        sorted(report["feature_importance"].items(), key=lambda item: item[1], reverse=True)
        """
    ),
    markdown("## Save model artifacts for the prototype"),
    code(
        """
        from football_predictor import build_team_stats
        from football_predictor.modeling import save_artifacts

        team_stats = build_team_stats(matches)
        save_artifacts(model, team_stats, report, project_root / "models")
        print(f"Saved artifacts for {len(team_stats)} teams.")
        """
    ),
    markdown("## Try a prediction"),
    code(
        """
        from football_predictor import predict_match
        from football_predictor.modeling import explanation_lines

        result = predict_match("Brazil", "Argentina", model, team_stats, is_neutral=True, is_major_tournament=True)
        print(result["predicted_outcome"])
        print(result["probabilities"])
        for line in explanation_lines("Brazil", "Argentina", team_stats, report):
            print("-", line)
        """
    ),
    markdown("## Run the prototype"),
    code(
        """
        print("From the project root, run:")
        print("streamlit run app.py")
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("notebooks/corelab.ipynb")
out.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote {out}")
