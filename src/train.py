"""Training script for COICOP cascade classifier."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow

from .cascade_classifier import CascadeCOICOPClassifier
from .data_preparation import load_annotations
from .hierarchical_classifier import HierarchicalCOICOPClassifier, HierarchicalConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _evaluate_on_annotations(
    classifier,
    model_path: str | Path,
    classifier_type: str,
    eval_data_path: str,
    eval_text_column: str,
    eval_top_k: int,
) -> dict[str, float]:
    """Predict on eval data and return top-k accuracy metrics dict.

    Args:
        classifier: Trained classifier instance (saved already).
        model_path: Path to the saved model directory.
        classifier_type: "hierarchical" or "cascade".
        eval_data_path: Path to evaluation parquet/csv file.
        eval_text_column: Name of text column in eval data.
        eval_top_k: Maximum K for top-k accuracy.

    Returns:
        Dict of {f"eval_level{N}_top-{K}": accuracy, ...}.
    """
    import pandas as pd

    from .predict import HierarchicalCOICOPPredictor
    from topk_accuracy import compute_topk_accuracy, detect_levels, detect_max_k, ensure_true_labels

    logger.info(f"Running evaluation on {eval_data_path}...")

    # Load eval data
    eval_path = Path(eval_data_path)
    if eval_path.suffix == ".parquet":
        eval_df = pd.read_parquet(eval_path)
    else:
        eval_df = pd.read_csv(eval_path)

    logger.info(f"Loaded {len(eval_df)} evaluation samples")

    if classifier_type == "hierarchical":
        predictor = HierarchicalCOICOPPredictor(model_path)
        result_df = predictor.predict_dataframe(
            eval_df,
            text_column=eval_text_column,
            top_k=eval_top_k,
        )
    else:
        from .predict import COICOPPredictor
        predictor = COICOPPredictor(model_path)
        result_df = predictor.predict_dataframe(
            eval_df,
            text_column=eval_text_column,
        )

    # Ensure true label columns exist
    result_df = ensure_true_labels(result_df)

    # Compute top-k accuracy per level
    levels = detect_levels(list(result_df.columns))
    ks = list(range(1, eval_top_k + 1))
    eval_metrics: dict[str, float] = {}

    for level in levels:
        max_k = detect_max_k(list(result_df.columns), level)
        row = compute_topk_accuracy(result_df, level, ks, max_k)
        for k in ks:
            key = f"top-{k}"
            if key in row and not (isinstance(row[key], float) and row[key] != row[key]):
                eval_metrics[f"eval_level{level}_top-{k}"] = row[key]

    logger.info(f"Evaluation metrics: {eval_metrics}")
    return eval_metrics


def train_cascade_classifier(
    annotations_path: str,
    output_dir: str,
    model_name: str = "camembert-base",
    embedding_dim: int = 128,
    max_seq_length: int = 64,
    batch_size: int = 32,
    lr: float = 2e-5,
    num_epochs: int = 20,
    patience: int = 5,
    min_samples: int = 50,
    mlflow_experiment: str | None = None,
    eval_data_path: str | None = None,
    eval_top_k: int = 5,
    eval_text_column: str = "text",
) -> CascadeCOICOPClassifier:
    """Train the cascade COICOP classifier.

    Args:
        annotations_path: Path to annotations.parquet file
        output_dir: Directory to save trained models
        model_name: HuggingFace model name for tokenizer
        embedding_dim: Text embedding dimension
        max_seq_length: Maximum sequence length
        batch_size: Training batch size
        lr: Learning rate
        num_epochs: Maximum epochs per classifier
        patience: Early stopping patience
        min_samples: Minimum samples for sub-classifiers
        mlflow_experiment: MLflow experiment name (optional)
        eval_data_path: Path to evaluation data for post-training metrics
        eval_top_k: Maximum K for top-k accuracy evaluation
        eval_text_column: Text column name in evaluation data

    Returns:
        Trained CascadeCOICOPClassifier
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading annotations from {annotations_path}...")
    df = load_annotations(annotations_path, exclude_technical=True)
    logger.info(f"Loaded {len(df)} samples (excluding 98.x and 99.x codes)")

    # Log data statistics
    unique_codes = df["code"].nunique()
    unique_level1 = df["level1"].nunique()
    logger.info(f"Unique codes: {unique_codes}")
    logger.info(f"Level 1 categories: {unique_level1}")

    # Initialize MLflow if experiment name provided
    trainer_params = None
    if mlflow_experiment:
        mlflow.set_experiment(mlflow_experiment)
        mlflow.start_run()
        mlflow.log_params({
            "model_name": model_name,
            "embedding_dim": embedding_dim,
            "max_seq_length": max_seq_length,
            "batch_size": batch_size,
            "lr": lr,
            "num_epochs": num_epochs,
            "patience": patience,
            "min_samples": min_samples,
            "num_samples": len(df),
            "unique_codes": unique_codes,
            "unique_level1": unique_level1,
        })

        from .mlflow_utils import make_trainer_params

        run_id = mlflow.active_run().info.run_id
        trainer_params = make_trainer_params(
            experiment_name=mlflow_experiment,
            run_id=run_id,
            tracking_uri=mlflow.get_tracking_uri(),
        )

    # Create cascade classifier
    classifier = CascadeCOICOPClassifier(
        model_name=model_name,
        embedding_dim=embedding_dim,
        max_seq_length=max_seq_length,
        min_samples=min_samples,
    )

    # Train
    logger.info("Starting cascade classifier training...")
    metrics = classifier.train(
        df=df,
        text_column="text",
        code_column="code",
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        patience=patience,
        save_dir=str(output_path / "checkpoints"),
        trainer_params=trainer_params,
    )

    # Log metrics to MLflow
    if mlflow_experiment:
        mlflow.log_metrics({
            "level1_num_classes": metrics["level1"]["num_classes"],
            "level1_train_samples": metrics["level1"]["train_samples"],
            "level1_val_samples": metrics["level1"]["val_samples"],
            "num_sub_classifiers": len(metrics["sub_classifiers"]),
        })

        for code, sub_metrics in metrics["sub_classifiers"].items():
            mlflow.log_metrics({
                f"sub_{code}_num_classes": sub_metrics["num_classes"],
                f"sub_{code}_train_samples": sub_metrics["train_samples"],
            })

    # Save the complete cascade classifier
    model_save_path = output_path / "model"
    classifier.save(model_save_path)
    logger.info(f"Model saved to {model_save_path}")

    if mlflow_experiment:
        # Post-training evaluation
        if eval_data_path:
            eval_metrics = _evaluate_on_annotations(
                classifier=classifier,
                model_path=model_save_path,
                classifier_type="cascade",
                eval_data_path=eval_data_path,
                eval_text_column=eval_text_column,
                eval_top_k=eval_top_k,
            )
            mlflow.log_metrics(eval_metrics)

        mlflow.log_artifacts(str(model_save_path), artifact_path="model")
        mlflow.end_run()

    return classifier


def train_hierarchical_classifier(
    annotations_path: str,
    output_dir: str,
    ngram_min_n: int = 3,
    ngram_max_n: int = 6,
    ngram_num_tokens: int = 100000,
    embedding_dim: int = 50,
    max_seq_length: int = 64,
    batch_size: int = 32,
    lr: float = 0.1,
    num_epochs: int = 20,
    patience: int = 5,
    min_samples: int = 50,
    use_parent_features: bool = True,
    teacher_forcing_ratio: float = 0.9,
    mlflow_experiment: str | None = None,
    eval_data_path: str | None = None,
    eval_top_k: int = 5,
    eval_text_column: str = "text",
) -> HierarchicalCOICOPClassifier:
    """Train the hierarchical multi-level COICOP classifier.

    This classifier trains separate models for each COICOP level (1-5),
    with each level receiving predictions from the parent level as
    categorical features.

    Args:
        annotations_path: Path to annotations.parquet file
        output_dir: Directory to save trained models
        ngram_min_n: Minimum n-gram size for tokenizer
        ngram_max_n: Maximum n-gram size for tokenizer
        ngram_num_tokens: Vocabulary size for n-gram tokenizer
        embedding_dim: Text embedding dimension
        max_seq_length: Maximum sequence length
        batch_size: Training batch size
        lr: Learning rate
        num_epochs: Maximum epochs per classifier
        patience: Early stopping patience
        min_samples: Minimum samples per level
        use_parent_features: Whether to use parent predictions as features
        teacher_forcing_ratio: Ratio of ground truth to use during training
        mlflow_experiment: MLflow experiment name (optional)
        eval_data_path: Path to evaluation data for post-training metrics
        eval_top_k: Maximum K for top-k accuracy evaluation
        eval_text_column: Text column name in evaluation data

    Returns:
        Trained HierarchicalCOICOPClassifier
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading annotations from {annotations_path}...")
    df = load_annotations(annotations_path, exclude_technical=True)
    logger.info(f"Loaded {len(df)} samples (excluding 98.x and 99.x codes)")

    # Log data statistics
    unique_codes = df["code"].nunique()
    unique_level1 = df["level1"].nunique()
    logger.info(f"Unique codes: {unique_codes}")
    logger.info(f"Level 1 categories: {unique_level1}")

    # Initialize MLflow if experiment name provided
    mlflow_run_info = None
    if mlflow_experiment:
        mlflow.set_experiment(mlflow_experiment)
        mlflow.start_run()
        mlflow.log_params({
            "classifier_type": "hierarchical",
            "ngram_min_n": ngram_min_n,
            "ngram_max_n": ngram_max_n,
            "ngram_num_tokens": ngram_num_tokens,
            "embedding_dim": embedding_dim,
            "max_seq_length": max_seq_length,
            "batch_size": batch_size,
            "lr": lr,
            "num_epochs": num_epochs,
            "patience": patience,
            "min_samples": min_samples,
            "use_parent_features": use_parent_features,
            "teacher_forcing_ratio": teacher_forcing_ratio,
            "num_samples": len(df),
            "unique_codes": unique_codes,
            "unique_level1": unique_level1,
        })

        run_id = mlflow.active_run().info.run_id
        mlflow_run_info = {
            "experiment_name": mlflow_experiment,
            "run_id": run_id,
            "tracking_uri": mlflow.get_tracking_uri(),
        }

    # Create config
    config = HierarchicalConfig(
        ngram_min_n=ngram_min_n,
        ngram_max_n=ngram_max_n,
        ngram_num_tokens=ngram_num_tokens,
        embedding_dim=embedding_dim,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        patience=patience,
        min_samples_per_level=min_samples,
        use_parent_features=use_parent_features,
        teacher_forcing_ratio=teacher_forcing_ratio,
    )

    # Create and train classifier
    classifier = HierarchicalCOICOPClassifier(config=config)

    logger.info("Starting hierarchical classifier training...")
    metrics = classifier.train(
        df=df,
        text_column="text",
        code_column="code",
        save_dir=str(output_path / "checkpoints"),
        mlflow_run_info=mlflow_run_info,
    )

    # Log metrics to MLflow
    if mlflow_experiment:
        for level_name, level_metrics in metrics.items():
            mlflow.log_metrics({
                f"{level_name}_num_classes": level_metrics["num_classes"],
                f"{level_name}_train_samples": level_metrics["train_samples"],
                f"{level_name}_val_samples": level_metrics["val_samples"],
            })

    # Save the complete classifier
    model_path = output_path / "hierarchical_model"
    classifier.save(model_path)
    logger.info(f"Model saved to {model_path}")

    if mlflow_experiment:
        # Post-training evaluation
        if eval_data_path:
            eval_metrics = _evaluate_on_annotations(
                classifier=classifier,
                model_path=model_path,
                classifier_type="hierarchical",
                eval_data_path=eval_data_path,
                eval_text_column=eval_text_column,
                eval_top_k=eval_top_k,
            )
            mlflow.log_metrics(eval_metrics)

        mlflow.log_artifacts(str(model_path), artifact_path="model")
        mlflow.end_run()

    return classifier


def main():
    parser = argparse.ArgumentParser(
        description="Train COICOP cascade classifier"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="data/annotations.parquet",
        help="Path to annotations parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="camembert-base",
        help="HuggingFace model name for tokenizer",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Text embedding dimension",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=64,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum samples for sub-classifiers",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name (optional)",
    )

    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
