"""CLI entry point for COICOP BDF Classifier."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

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
        resume_from=args.resume,
        encryption_key=args.encryption_key,
        max_level=args.max_level,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )


def cmd_fine_tune_hierarchical(args: argparse.Namespace) -> None:
    """Fine-tune a pre-trained hierarchical classifier on new data."""
    from src.train import fine_tune_hierarchical_classifier

    levels = None
    if args.levels:
        levels = [l.strip() for l in args.levels.split(",")]

    fine_tune_hierarchical_classifier(
        model_path=args.model,
        annotations_path=args.data,
        output_dir=args.output,
        levels=levels,
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        mlflow_experiment=args.mlflow_experiment,
        eval_data_path=args.eval_data,
        eval_top_k=args.eval_top_k,
        eval_text_column=args.eval_text_column,
        encryption_key=args.encryption_key,
        max_level=args.max_level,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )


def cmd_train_basic(args: argparse.Namespace) -> None:
    """Train the basic flat classifier."""
    from src.train import train_basic_classifier

    train_basic_classifier(
        data_path=args.data,
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
        mlflow_experiment=args.mlflow_experiment,
        eval_data_path=args.eval_data,
        eval_top_k=args.eval_top_k,
        eval_text_column=args.eval_text_column,
        encryption_key=args.encryption_key,
    )


def cmd_predict_basic(args: argparse.Namespace) -> None:
    """Predict COICOP codes using basic flat classifier."""
    from src.predict import BasicCOICOPPredictor

    predictor = BasicCOICOPPredictor(args.model)

    if args.file:
        predictor.predict_file(
            input_path=args.file,
            output_path=args.output,
            text_column=args.text_column,
            batch_size=args.batch_size,
            top_k=args.top_k,
        )
    else:
        predictions = predictor.predict(args.texts, top_k=args.top_k)
        for pred in predictions:
            print(f"\nText: {pred['text']}")
            print(f"Code: {pred['code']} (confidence: {pred['confidence']:.2f})")
            if "alternatives" in pred:
                for i, alt in enumerate(pred["alternatives"], start=2):
                    print(f"  top {i}: {alt['code']} (conf: {alt['confidence']:.2f})")


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
            confidence_threshold=args.confidence_threshold,
        )
    else:
        # Text-based prediction
        if args.input:
            texts = [args.input]
        else:
            texts = args.texts

        predictions = predictor.predict(
            texts, top_k=args.top_k, confidence_threshold=args.confidence_threshold
        )
        for pred in predictions:
            # Format hierarchical output nicely
            print(f"\nText: {pred['text']}")
            print(f"Final code: {pred['code']} (confidence: {pred['confidence']:.2f})")
            print(f"Combined confidence: {pred['combined_confidence']:.4f}")
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


def cmd_extract_ddc(args: argparse.Namespace) -> None:
    """Extract DDC data from S3."""
    from src.extract_ddc import extract_ddc

    extract_ddc(
        annee=args.annee,
        mois=args.mois,
        output_s3_path=args.output,
        famille_circana_path=args.famille,
        memory_limit=args.memory,
        dry_run=args.dry_run,
        encrypt=args.encrypt,
        encryption_key=args.encryption_key,
    )


def cmd_build_training_data(args: argparse.Namespace) -> None:
    """Build a balanced training dataset from DDC and synthetic data."""
    from src.build_training_data import build_training_data

    build_training_data(
        ddc_path=args.ddc,
        output_path=args.output,
        synthetic_path=args.synthetic,
        max_per_code=args.max_per_code,
        seed=args.seed,
        encryption_key=args.encryption_key,
    )


def cmd_evaluate_report(args: argparse.Namespace) -> None:
    """Generate a comprehensive evaluation report on annotated data."""
    from src.evaluation_report import run_evaluation, format_report, log_metrics_to_mlflow

    metrics = run_evaluation(
        model_path=args.model,
        data_dir=args.data_dir,
        top_k=args.top_k,
        text_column=args.text_column,
        amount_threshold=args.amount_threshold,
    )
    report = format_report(metrics)
    print(report)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report, encoding="utf-8")
        logger.info(f"Report saved to {args.output}")

    if args.mlflow_run_id or args.mlflow_experiment:
        log_metrics_to_mlflow(
            metrics,
            run_id=args.mlflow_run_id,
            experiment_name=args.mlflow_experiment,
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
    train_hier_parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from a previous checkpoint (uses output directory)",
    )
    train_hier_parser.add_argument(
        "--encryption-key",
        type=str,
        default=None,
        help="Parquet encryption key (hex, 32 chars) for reading/writing encrypted files",
    )
    train_hier_parser.add_argument(
        "--max-level",
        type=int,
        default=5,
        choices=range(1, 6),
        metavar="{1,2,3,4,5}",
        help="Maximum COICOP hierarchy depth to train (1-5, default: 5)",
    )
    train_hier_parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0 = main process only, safest on Windows; increase on Linux)",
    )
    train_hier_parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=True,
        help="Pin memory for faster CPU→GPU transfer (default: True)",
    )
    train_hier_parser.add_argument(
        "--no-pin-memory",
        action="store_false",
        dest="pin_memory",
        help="Disable pinned memory",
    )
    train_hier_parser.set_defaults(func=cmd_train_hierarchical)

    # Fine-tune-hierarchical command
    ft_hier_parser = subparsers.add_parser(
        "fine-tune-hierarchical",
        help="Fine-tune a pre-trained hierarchical classifier on new data",
    )
    ft_hier_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the pre-trained hierarchical model",
    )
    ft_hier_parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to new training data (parquet or csv)",
    )
    ft_hier_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the fine-tuned model",
    )
    ft_hier_parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Comma-separated list of levels to fine-tune (e.g. level3,level4). Default: all",
    )
    ft_hier_parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: original lr / 10)",
    )
    ft_hier_parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of epochs (default: 5)",
    )
    ft_hier_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (default: same as original)",
    )
    ft_hier_parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (default: 3)",
    )
    ft_hier_parser.add_argument(
        "--teacher-forcing-ratio",
        type=float,
        default=None,
        help="Teacher forcing ratio (default: same as original)",
    )
    ft_hier_parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name (optional)",
    )
    ft_hier_parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Path to evaluation data for post-training top-k accuracy",
    )
    ft_hier_parser.add_argument(
        "--eval-top-k",
        type=int,
        default=5,
        help="Maximum K for top-k accuracy evaluation (default: 5)",
    )
    ft_hier_parser.add_argument(
        "--eval-text-column",
        type=str,
        default="text",
        help="Text column name in evaluation data (default: text)",
    )
    ft_hier_parser.add_argument(
        "--encryption-key",
        type=str,
        default=None,
        help="Parquet encryption key (hex, 32 chars) for reading/writing encrypted files",
    )
    ft_hier_parser.add_argument(
        "--max-level",
        type=int,
        default=None,
        choices=range(1, 6),
        metavar="{1,2,3,4,5}",
        help="Maximum COICOP hierarchy depth (1-5, default: use model's setting)",
    )
    ft_hier_parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: use model's setting)",
    )
    ft_hier_parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=None,
        help="Pin memory for faster CPU→GPU transfer",
    )
    ft_hier_parser.add_argument(
        "--no-pin-memory",
        action="store_false",
        dest="pin_memory",
        help="Disable pinned memory",
    )
    ft_hier_parser.set_defaults(func=cmd_fine_tune_hierarchical)

    # Train-basic command
    train_basic_parser = subparsers.add_parser(
        "train-basic",
        help="Train the basic flat classifier with n-gram tokenization",
    )
    train_basic_parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training parquet (from build-training-data)",
    )
    train_basic_parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/basic",
        help="Output directory for trained model",
    )
    train_basic_parser.add_argument(
        "--ngram-min",
        type=int,
        default=3,
        help="Minimum n-gram size for tokenizer",
    )
    train_basic_parser.add_argument(
        "--ngram-max",
        type=int,
        default=6,
        help="Maximum n-gram size for tokenizer",
    )
    train_basic_parser.add_argument(
        "--ngram-vocab-size",
        type=int,
        default=100000,
        help="N-gram vocabulary size",
    )
    train_basic_parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding dimension",
    )
    train_basic_parser.add_argument(
        "--max-seq-length",
        type=int,
        default=64,
        help="Maximum sequence length",
    )
    train_basic_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    train_basic_parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate",
    )
    train_basic_parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Maximum number of epochs",
    )
    train_basic_parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience",
    )
    train_basic_parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name (enables MLflow + pyfunc logging)",
    )
    train_basic_parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Path to evaluation parquet for post-training top-k accuracy",
    )
    train_basic_parser.add_argument(
        "--eval-top-k",
        type=int,
        default=5,
        help="Maximum K for top-k accuracy evaluation (default: 5)",
    )
    train_basic_parser.add_argument(
        "--eval-text-column",
        type=str,
        default="product",
        help="Text column name in evaluation data (default: product)",
    )
    train_basic_parser.add_argument(
        "--encryption-key",
        type=str,
        default=None,
        help="Parquet encryption key (hex, 32 chars) for reading/writing encrypted files",
    )
    train_basic_parser.set_defaults(func=cmd_train_basic)

    # Predict-basic command
    predict_basic_parser = subparsers.add_parser(
        "predict-basic",
        help="Predict COICOP codes using basic flat classifier",
    )
    predict_basic_parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/basic/basic_model",
        help="Path to saved basic model",
    )
    predict_basic_parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Input file for batch prediction",
    )
    predict_basic_parser.add_argument(
        "--output",
        type=str,
        default="predictions_basic.csv",
        help="Output file for batch prediction",
    )
    predict_basic_parser.add_argument(
        "--text-column",
        type=str,
        default="product",
        help="Name of text column in input file",
    )
    predict_basic_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for prediction",
    )
    predict_basic_parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of top predictions (default: 1)",
    )
    predict_basic_parser.add_argument(
        "texts",
        nargs="*",
        help="Text(s) to classify (if not using --file)",
    )
    predict_basic_parser.set_defaults(func=cmd_predict_basic)

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
        "--confidence-threshold",
        type=float,
        default=None,
        help="Minimum confidence per level; stop at the deepest level meeting this threshold",
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

    # Evaluate-report command
    eval_report_parser = subparsers.add_parser(
        "evaluate-report",
        help="Generate comprehensive evaluation report on annotated data",
    )
    eval_report_parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/basic/basic_model",
        help="Path to trained model (local dir, runs:/..., or models:/...)",
    )
    eval_report_parser.add_argument(
        "--data-dir",
        type=str,
        default="data/annotated",
        help="Directory with annotated CSVs",
    )
    eval_report_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Max K for top-k accuracy (default: 5)",
    )
    eval_report_parser.add_argument(
        "--text-column",
        type=str,
        default="product",
        help="Text column after normalization (default: product)",
    )
    eval_report_parser.add_argument(
        "--mlflow-run-id",
        type=str,
        default=None,
        help="Existing MLflow run ID to log metrics to",
    )
    eval_report_parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name (creates new run if no run-id)",
    )
    eval_report_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save report to file (txt)",
    )
    eval_report_parser.add_argument(
        "--amount-threshold",
        type=float,
        default=200,
        help="Spending threshold in euros (default: 200)",
    )
    eval_report_parser.set_defaults(func=cmd_evaluate_report)

    # Build-training-data command
    build_data_parser = subparsers.add_parser(
        "build-training-data",
        help="Build a balanced training dataset from DDC extraction and synthetic data",
    )
    build_data_parser.add_argument(
        "--ddc",
        type=str,
        required=True,
        help="Path to DDC parquet file (local, S3, or HTTP URL)",
    )
    build_data_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output parquet file path",
    )
    build_data_parser.add_argument(
        "--synthetic",
        type=str,
        default="data/synthetic_data.csv",
        help="Path to synthetic data CSV (default: data/synthetic_data.csv)",
    )
    build_data_parser.add_argument(
        "--max-per-code",
        type=int,
        default=1000,
        help="Max DDC texts per level-4 code (default: 1000)",
    )
    build_data_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    build_data_parser.add_argument(
        "--encryption-key",
        type=str,
        default=None,
        help="Parquet encryption key (hex, 32 chars) for reading/writing encrypted files",
    )
    build_data_parser.set_defaults(func=cmd_build_training_data)

    # Extract-ddc command
    extract_ddc_parser = subparsers.add_parser(
        "extract-ddc",
        help="Extract DDC data from S3 and apply COICOP code mapping",
    )
    extract_ddc_parser.add_argument(
        "--annee",
        type=int,
        nargs="+",
        required=True,
        help="Year(s) to extract",
    )
    extract_ddc_parser.add_argument(
        "--mois",
        type=int,
        nargs="*",
        default=None,
        help="Month(s) to extract (default: all months)",
    )
    extract_ddc_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override S3 output path",
    )
    extract_ddc_parser.add_argument(
        "--famille",
        type=str,
        default="data/famille_circana.csv",
        help="Path to famille_circana mapping CSV",
    )
    extract_ddc_parser.add_argument(
        "--memory",
        type=str,
        default="6GB",
        help="DuckDB memory limit (default: 6GB)",
    )
    extract_ddc_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the SQL without executing",
    )
    extract_ddc_parser.add_argument(
        "--encrypt",
        action="store_true",
        help="Encrypt the output parquet file (AES-GCM 256 bits)",
    )
    extract_ddc_parser.add_argument(
        "--encryption-key",
        type=str,
        default=None,
        help="Parquet encryption key (hex, 32 chars). Implies --encrypt",
    )
    extract_ddc_parser.set_defaults(func=cmd_extract_ddc)

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
