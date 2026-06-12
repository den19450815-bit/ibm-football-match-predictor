from __future__ import annotations

import argparse

from football_predictor import load_artifacts, predict_match
from football_predictor.modeling import explanation_lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict an international football match outcome.")
    parser.add_argument("--team-a", required=True, help="Home/team A name, for example Brazil")
    parser.add_argument("--team-b", required=True, help="Away/team B name, for example Argentina")
    parser.add_argument("--models", default="models", help="Directory containing model artifacts")
    parser.add_argument("--neutral", action="store_true", help="Treat the match as a neutral venue")
    parser.add_argument("--major", action="store_true", help="Treat the match as a major tournament")
    args = parser.parse_args()

    model, data = load_artifacts(args.models)
    team_stats = data["team_stats"]
    result = predict_match(
        args.team_a,
        args.team_b,
        model=model,
        team_stats=team_stats,
        is_neutral=args.neutral,
        is_major_tournament=args.major,
    )

    print(f"Prediction: {result['predicted_outcome']}")
    print(f"{args.team_a} win: {result['probabilities']['team_a_win']:.1%}")
    print(f"Draw:       {result['probabilities']['draw']:.1%}")
    print(f"{args.team_b} win: {result['probabilities']['team_b_win']:.1%}")
    print("\nExplanation:")
    for line in explanation_lines(args.team_a, args.team_b, team_stats, data.get("report")):
        print(f"- {line}")


if __name__ == "__main__":
    main()

