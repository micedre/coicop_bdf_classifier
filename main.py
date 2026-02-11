"""CLI entry point for COICOP BDF Classifier."""

from __future__ import annotations

import argparse
import json
import logging
import os
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
        eval_data_path=args.eval_data,
        eval_top_k=args.eval_top_k,
        eval_text_column=args.eval_text_column,
    )


def cmd_train_hierarchical(args: argparse.Namespace) -> None:
    """Train the hierarchical multi-level classifier."""
    from src.train import train_hierarchical_classifier

    train_hierarchical_classifier(
        annotations_path=args.data,
        output_dir=args.output,
        ngram_min_n=args.ngram_min,
        ngram_max_n=args.ngram_max,
        ngram_num_tokens=args.ngram_vocab_size,
        embedding_dim=args.embedding_dim,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        patience=args.patience,
        min_samples=args.min_samples,
        use_parent_features=args.use_parent_features,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        mlflow_experiment=args.mlflow_experiment,
        eval_data_path=args.eval_data,
        eval_top_k=args.eval_top_k,
        eval_text_column=args.eval_text_column,
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


def cmd_predict_hierarchical(args: argparse.Namespace) -> None:
    """Predict COICOP codes using hierarchical classifier."""
    from src.predict import HierarchicalCOICOPPredictor

    predictor = HierarchicalCOICOPPredictor(args.model)

    if args.file:
        # File-based prediction
        predictor.predict_file(
            input_path=args.file,
            output_path=args.output,
            text_column=args.text_column,
            batch_size=args.batch_size,
            top_k=args.top_k,
        )
    else:
        # Text-based prediction
        if args.input:
            texts = [args.input]
        else:
            texts = args.texts

        predictions = predictor.predict(texts, top_k=args.top_k)
        for pred in predictions:
            # Format hierarchical output nicely
            print(f"\nText: {pred['text']}")
            print(f"Final code: {pred['code']} (confidence: {pred['confidence']:.2f})")
            if "levels" in pred:
                print("Level breakdown:")
                for level_name, level_data in pred["levels"].items():
                    print(f"  {level_name}: {level_data['code']} (conf: {level_data['confidence']:.2f})")
                    if "alternatives" in level_data:
                        for i, alt in enumerate(level_data["alternatives"], start=2):
                            print(f"    top {i}: {alt['code']} (conf: {alt['confidence']:.2f})")


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI prediction server."""
    import uvicorn

    os.environ["COICOP_MODEL_PATH"] = args.model
    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


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
    train_parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Path to evaluation parquet for post-training top-k accuracy",
    )
    train_parser.add_argument(
        "--eval-top-k",
        type=int,
        default=5,
        help="Maximum K for top-k accuracy evaluation (default: 5)",
    )
    train_parser.add_argument(
        "--eval-text-column",
        type=str,
        default="text",
        help="Text column name in evaluation data (default: text)",
    )
    train_parser.set_defaults(func=cmd_train)

    # Train-hierarchical command
    train_hier_parser = subparsers.add_parser(
        "train-hierarchical",
        help="Train the hierarchical multi-level classifier with n-gram tokenization",
    )
    train_hier_parser.add_argument(
        "--data",
        type=str,
        default="data/data-train.parquet",
        help="Path to training data (parquet or csv)",
    )
    train_hier_parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/hierarchical",
        help="Output directory for trained models",
    )
    train_hier_parser.add_argument(
        "--ngram-min",
        type=int,
        default=3,
        help="Minimum n-gram size for tokenizer",
    )
    train_hier_parser.add_argument(
        "--ngram-max",
        type=int,
        default=6,
        help="Maximum n-gram size for tokenizer",
    )
    train_hier_parser.add_argument(
        "--ngram-vocab-size",
        type=int,
        default=100000,
        help="Vocabulary size for n-gram tokenizer",
    )
    train_hier_parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Text embedding dimension",
    )
    train_hier_parser.add_argument(
        "--max-seq-length",
        type=int,
        default=64,
        help="Maximum sequence length",
    )
    train_hier_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    train_hier_parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    train_hier_parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Maximum number of epochs",
    )
    train_hier_parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience",
    )
    train_hier_parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum samples per level",
    )
    train_hier_parser.add_argument(
        "--use-parent-features",
        action="store_true",
        default=True,
        help="Use parent predictions as categorical features (default: True)",
    )
    train_hier_parser.add_argument(
        "--no-parent-features",
        action="store_false",
        dest="use_parent_features",
        help="Disable parent prediction features",
    )
    train_hier_parser.add_argument(
        "--teacher-forcing-ratio",
        type=float,
        default=0.8,
        help="Ratio of ground truth to use during training (0.0-1.0)",
    )
    train_hier_parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name (optional)",
    )
    train_hier_parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Path to evaluation parquet for post-training top-k accuracy",
    )
    train_hier_parser.add_argument(
        "--eval-top-k",
        type=int,
        default=5,
        help="Maximum K for top-k accuracy evaluation (default: 5)",
    )
    train_hier_parser.add_argument(
        "--eval-text-column",
        type=str,
        default="text",
        help="Text column name in evaluation data (default: text)",
    )
    train_hier_parser.set_defaults(func=cmd_train_hierarchical)

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

    # Predict-hierarchical command
    predict_hier_parser = subparsers.add_parser(
        "predict-hierarchical",
        help="Predict COICOP codes using hierarchical classifier",
    )
    predict_hier_parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/hierarchical/hierarchical_model",
        help="Path to saved hierarchical model",
    )
    predict_hier_parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Single text to classify",
    )
    predict_hier_parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Input file for batch prediction",
    )
    predict_hier_parser.add_argument(
        "--output",
        type=str,
        default="predictions_hierarchical.csv",
        help="Output file for batch prediction",
    )
    predict_hier_parser.add_argument(
        "--text-column",
        type=str,
        default="product",
        help="Name of text column in input file",
    )
    predict_hier_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for prediction",
    )
    predict_hier_parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of top predictions per level (default: 1)",
    )
    predict_hier_parser.add_argument(
        "texts",
        nargs="*",
        help="Text(s) to classify (if not using --file or --input)",
    )
    predict_hier_parser.set_defaults(func=cmd_predict_hierarchical)

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

    # Serve command
    serve_parser = subparsers.add_parser(
        "serve", help="Start the FastAPI prediction API server"
    )
    serve_parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/hierarchical/hierarchical_model",
        help="Path to saved hierarchical model",
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    serve_parser.set_defaults(func=cmd_serve)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
