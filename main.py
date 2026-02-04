"""CLI entry point for COICOP BDF Classifier."""

from __future__ import annotations

import argparse
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_train(args: argparse.Namespace) -> None:
    """Train the cascade classifier."""
    from src.train import train_cascade_classifier

    train_cascade_classifier(
        annotations_path=args.annotations,
        output_dir=args.output_dir,
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        patience=args.patience,
        min_samples=args.min_samples,
        mlflow_experiment=args.mlflow_experiment,
    )


def cmd_predict(args: argparse.Namespace) -> None:
    """Predict COICOP codes."""
    from src.predict import COICOPPredictor

    predictor = COICOPPredictor(args.model_path)

    if args.file:
        # File-based prediction
        predictor.predict_file(
            input_path=args.file,
            output_path=args.output,
            text_column=args.text_column,
            batch_size=args.batch_size,
        )
    else:
        # Text-based prediction
        predictions = predictor.predict(args.texts)
        for pred in predictions:
            print(json.dumps(pred, ensure_ascii=False, indent=2))


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate the classifier on a test set."""
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report

    from src.predict import COICOPPredictor

    predictor = COICOPPredictor(args.model_path)

    # Load test data
    if args.test_file.endswith(".parquet"):
        df = pd.read_parquet(args.test_file)
    else:
        df = pd.read_csv(args.test_file)

    # Filter technical codes if needed
    if args.exclude_technical:
        df = df[~df[args.label_column].str.startswith(("98", "99"))]

    logger.info(f"Evaluating on {len(df)} samples...")

    # Predict
    result_df = predictor.predict_dataframe(
        df,
        text_column=args.text_column,
        batch_size=args.batch_size,
    )

    # Calculate metrics
    y_true = result_df[args.label_column].tolist()
    y_pred = result_df["predicted_code"].tolist()

    # Exact match accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nExact Match Accuracy: {accuracy:.4f}")

    # Level 1 accuracy
    y_true_l1 = [c.split(".")[0].zfill(2) for c in y_true]
    y_pred_l1 = result_df["predicted_level1"].tolist()
    accuracy_l1 = accuracy_score(y_true_l1, y_pred_l1)
    print(f"Level 1 Accuracy: {accuracy_l1:.4f}")

    # Detailed classification report for level 1
    if args.detailed:
        print("\nLevel 1 Classification Report:")
        print(classification_report(y_true_l1, y_pred_l1))


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="COICOP BDF Classifier - Hierarchical text classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the cascade classifier")
    train_parser.add_argument(
        "--annotations",
        type=str,
        default="data/annotations.parquet",
        help="Path to annotations parquet file",
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for trained models",
    )
    train_parser.add_argument(
        "--model-name",
        type=str,
        default="camembert-base",
        help="HuggingFace model name for tokenizer",
    )
    train_parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Text embedding dimension",
    )
    train_parser.add_argument(
        "--max-seq-length",
        type=int,
        default=64,
        help="Maximum sequence length",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    train_parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Maximum number of epochs",
    )
    train_parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience",
    )
    train_parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum samples for sub-classifiers",
    )
    train_parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name (optional)",
    )
    train_parser.set_defaults(func=cmd_train)

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict COICOP codes")
    predict_parser.add_argument(
        "--model-path",
        type=str,
        default="models/model",
        help="Path to saved model",
    )
    predict_parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Input file for batch prediction",
    )
    predict_parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output file for batch prediction",
    )
    predict_parser.add_argument(
        "--text-column",
        type=str,
        default="product",
        help="Name of text column in input file",
    )
    predict_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for prediction",
    )
    predict_parser.add_argument(
        "texts",
        nargs="*",
        help="Text(s) to classify (if not using --file)",
    )
    predict_parser.set_defaults(func=cmd_predict)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the classifier")
    eval_parser.add_argument(
        "--model-path",
        type=str,
        default="models/model",
        help="Path to saved model",
    )
    eval_parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path to test file",
    )
    eval_parser.add_argument(
        "--text-column",
        type=str,
        default="product",
        help="Name of text column",
    )
    eval_parser.add_argument(
        "--label-column",
        type=str,
        default="coicop",
        help="Name of label column",
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for prediction",
    )
    eval_parser.add_argument(
        "--exclude-technical",
        action="store_true",
        help="Exclude 98.x and 99.x codes",
    )
    eval_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed classification report",
    )
    eval_parser.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
