from __future__ import annotations

import argparse
from pathlib import Path

from football_predictor import train_and_save


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the football match outcome model.")
    parser.add_argument("--data", default="data/results.csv", help="Path to results.csv")
    parser.add_argument("--models", default="models", help="Directory to write model artifacts")
    parser.add_argument("--start-date", default="1990-01-01", help="First match date to use for training features")
    parser.add_argument("--cutoff", default="2018-01-01", help="Date cutoff for train/test split")
    args = parser.parse_args()

    report = train_and_save(
        csv_path=Path(args.data),
        model_dir=Path(args.models),
        start_date=args.start_date,
        cutoff=args.cutoff,
    )

    print("Training complete")
    print(f"Rows in source data:      {report['match_rows']:,}")
    print(f"Rows in feature table:    {report['feature_rows']:,}")
    print(f"Teams available in demo:  {report['team_count']:,}")
    print(f"Train rows:               {report['train_rows']:,}")
    print(f"Test rows:                {report['test_rows']:,}")
    print(f"Model accuracy:           {report['accuracy']:.2%}")
    print(f"Baseline accuracy:        {report['baseline_accuracy']:.2%}")
    print("Confusion matrix [Team A win, Draw, Team B win]:")
    for row in report["confusion_matrix"]:
        print("  " + " ".join(f"{value:5d}" for value in row))

    top = sorted(report["feature_importance"].items(), key=lambda item: item[1], reverse=True)[:5]
    print("Top feature importances:")
    for name, score in top:
        print(f"  {name:<24} {score:.4f}")


if __name__ == "__main__":
    main()

