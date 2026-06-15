"""Train SmartFlow's congestion model from a Kaggle-compatible CSV."""
import argparse
import os

from ml.congestion_predictor import CongestionPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the extracted traffic CSV")
    parser.add_argument("--output", default="models/congestion.pkl")
    parser.add_argument("--lookahead", type=int, default=5)
    args = parser.parse_args()

    predictor = CongestionPredictor(lookahead=args.lookahead)
    result = predictor.train_csv(args.csv)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    predictor.save(args.output)
    print(f"Model saved to {args.output}")
    print(result)


if __name__ == "__main__":
    main()
