"""Training script for COICOP cascade classifier."""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import mlflow

from .basic_classifier import BasicCOICOPClassifier, BasicConfig
from .cascade_classifier import CascadeCOICOPClassifier
from .data_preparation import load_annotations
from .hierarchical_classifier import HierarchicalCOICOPClassifier, HierarchicalConfig
from .multihead_classifier import MultiHeadCOICOPClassifier, MultiHeadConfig

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
    elif classifier_type == "multihead":
        from .predict import MultiHeadCOICOPPredictor
        predictor = MultiHeadCOICOPPredictor(model_path)
        result_df = predictor.predict_dataframe(
            eval_df,
            text_column=eval_text_column,
            top_k=eval_top_k,
        )
    elif classifier_type == "basic":
        from .predict import BasicCOICOPPredictor
        predictor = BasicCOICOPPredictor(model_path)
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

    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_file = Path(tmp_dir) / "eval_predictions.parquet"
        result_df.to_parquet(artifact_file, index=False)
        mlflow.log_artifact(str(artifact_file), artifact_path="evaluation")
    logger.info("Evaluation predictions logged as MLflow artifact")

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
            "data_path": annotations_path,
        })
        if eval_data_path:
            mlflow.log_param("eval_data_path", eval_data_path)

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
    resume_from: bool = False,
    encryption_key: str | None = None,
    max_level: int = 5,
    num_workers: int = 0,
    pin_memory: bool = True,
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
        resume_from: Whether to resume from a previous checkpoint in output_dir

    Returns:
        Trained HierarchicalCOICOPClassifier
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Compute model path early (used for checkpoint and resume)
    model_path = output_path / "hierarchical_model"
    resume_path = str(model_path) if resume_from else None

    # Load data
    logger.info(f"Loading annotations from {annotations_path}...")
    df = load_annotations(annotations_path, exclude_technical=True, encryption_key=encryption_key)
    logger.info(f"Loaded {len(df)} samples (excluding 98.x and 99.x codes)")

    # Log data statistics
    unique_codes = df["code"].nunique()
    unique_level1 = df["level1"].nunique()
    logger.info(f"Unique codes: {unique_codes}")
    logger.info(f"Level 1 categories: {unique_level1}")

    # Try to reuse MLflow run on resume
    saved_mlflow_run_id = None
    if resume_from and model_path.exists():
        import pickle
        meta_file = model_path / "hierarchical_metadata.pkl"
        if meta_file.exists():
            with open(meta_file, "rb") as f:
                saved_meta = pickle.load(f)
            saved_mlflow_run_id = saved_meta.get("mlflow_run_id")

    # Initialize MLflow if experiment name provided
    mlflow_run_info = None
    if mlflow_experiment:
        mlflow.set_experiment(mlflow_experiment)
        if saved_mlflow_run_id:
            logger.info(f"Resuming MLflow run {saved_mlflow_run_id}")
            mlflow.start_run(run_id=saved_mlflow_run_id)
        else:
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
            "max_level": max_level,
            "num_samples": len(df),
            "unique_codes": unique_codes,
            "unique_level1": unique_level1,
            "data_path": annotations_path,
        })
        if eval_data_path:
            mlflow.log_param("eval_data_path", eval_data_path)

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
        max_level=max_level,
        num_workers=num_workers,
        pin_memory=pin_memory,
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
        resume_from=resume_path,
        checkpoint_path=str(model_path),
    )

    # Log metrics to MLflow
    if mlflow_experiment:
        for level_name, level_metrics in metrics.items():
            mlflow.log_metrics({
                f"{level_name}_num_classes": level_metrics["num_classes"],
                f"{level_name}_train_samples": level_metrics["train_samples"],
                f"{level_name}_val_samples": level_metrics["val_samples"],
            })

    # Save the complete classifier (final save with MLflow run_id)
    mlflow_run_id = mlflow.active_run().info.run_id if mlflow_experiment else None
    classifier.save(model_path, mlflow_run_id=mlflow_run_id)
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

        # Log pyfunc model for end-to-end serving
        mlflow.pyfunc.log_model(
            name="pyfunc_model",
            python_model="src/mlflow_model_hierarchical.py",
            artifacts={
                "model_dir": str(model_path),
                "stopwords": "data/text/stopwords.json",
            },
        )

        mlflow.end_run()

    return classifier


def fine_tune_hierarchical_classifier(
    model_path: str,
    annotations_path: str,
    output_dir: str,
    levels: list[str] | None = None,
    lr: float | None = None,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    patience: int | None = None,
    teacher_forcing_ratio: float | None = None,
    mlflow_experiment: str | None = None,
    eval_data_path: str | None = None,
    eval_top_k: int = 5,
    eval_text_column: str = "text",
    encryption_key: str | None = None,
    max_level: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> HierarchicalCOICOPClassifier:
    """Fine-tune a pre-trained hierarchical classifier on new data.

    Args:
        model_path: Path to the pre-trained hierarchical model directory.
        annotations_path: Path to new training data (parquet or csv).
        output_dir: Directory to save the fine-tuned model.
        levels: List of level names to fine-tune (None = all).
        lr: Learning rate override.
        num_epochs: Number of epochs override.
        batch_size: Batch size override.
        patience: Early stopping patience override.
        teacher_forcing_ratio: Teacher forcing ratio override.
        mlflow_experiment: MLflow experiment name (optional).
        eval_data_path: Path to evaluation data for post-training metrics.
        eval_top_k: Maximum K for top-k accuracy evaluation.
        eval_text_column: Text column name in evaluation data.

    Returns:
        Fine-tuned HierarchicalCOICOPClassifier.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load pre-trained model
    logger.info(f"Loading pre-trained model from {model_path}...")
    classifier = HierarchicalCOICOPClassifier.load(model_path)

    # Override max_level if provided
    if max_level is not None:
        classifier.config.max_level = max_level

    # Override DataLoader settings if provided
    if num_workers is not None:
        classifier.config.num_workers = num_workers
    if pin_memory is not None:
        classifier.config.pin_memory = pin_memory

    # Load new data
    logger.info(f"Loading new training data from {annotations_path}...")
    df = load_annotations(annotations_path, exclude_technical=True, encryption_key=encryption_key)
    logger.info(f"Loaded {len(df)} samples (excluding 98.x and 99.x codes)")

    # Log data statistics
    unique_codes = df["code"].nunique()
    logger.info(f"Unique codes: {unique_codes}")

    # Initialize MLflow if experiment name provided
    mlflow_run_info = None
    if mlflow_experiment:
        mlflow.set_experiment(mlflow_experiment)
        mlflow.start_run()
        params = {
            "task": "fine-tuning",
            "base_model_path": model_path,
            "num_samples": len(df),
            "unique_codes": unique_codes,
            "data_path": annotations_path,
        }
        if levels is not None:
            params["levels"] = ",".join(levels)
        if lr is not None:
            params["lr"] = lr
        if num_epochs is not None:
            params["num_epochs"] = num_epochs
        if batch_size is not None:
            params["batch_size"] = batch_size
        if patience is not None:
            params["patience"] = patience
        if teacher_forcing_ratio is not None:
            params["teacher_forcing_ratio"] = teacher_forcing_ratio
        mlflow.log_params(params)
        if eval_data_path:
            mlflow.log_param("eval_data_path", eval_data_path)

        run_id = mlflow.active_run().info.run_id
        mlflow_run_info = {
            "experiment_name": mlflow_experiment,
            "run_id": run_id,
            "tracking_uri": mlflow.get_tracking_uri(),
        }

    # Fine-tune
    ft_model_path = output_path / "hierarchical_model"
    logger.info("Starting fine-tuning...")
    metrics = classifier.fine_tune(
        df=df,
        text_column="text",
        code_column="code",
        save_dir=str(output_path / "checkpoints"),
        mlflow_run_info=mlflow_run_info,
        levels=levels,
        lr=lr,
        num_epochs=num_epochs,
        batch_size=batch_size,
        patience=patience,
        teacher_forcing_ratio=teacher_forcing_ratio,
        checkpoint_path=str(ft_model_path),
    )

    # Log metrics to MLflow
    if mlflow_experiment:
        for level_name, level_metrics in metrics.items():
            mlflow.log_metrics({
                f"ft_{level_name}_num_classes": level_metrics["num_classes"],
                f"ft_{level_name}_train_samples": level_metrics["train_samples"],
                f"ft_{level_name}_val_samples": level_metrics["val_samples"],
                f"ft_{level_name}_n_dropped": level_metrics["n_dropped"],
            })

    # Save the fine-tuned model
    ft_model_path = output_path / "hierarchical_model"
    classifier.save(ft_model_path)
    logger.info(f"Fine-tuned model saved to {ft_model_path}")

    if mlflow_experiment:
        # Post-training evaluation
        if eval_data_path:
            eval_metrics = _evaluate_on_annotations(
                classifier=classifier,
                model_path=ft_model_path,
                classifier_type="hierarchical",
                eval_data_path=eval_data_path,
                eval_text_column=eval_text_column,
                eval_top_k=eval_top_k,
            )
            mlflow.log_metrics(eval_metrics)

        mlflow.log_artifacts(str(ft_model_path), artifact_path="model")

        # Log pyfunc model for end-to-end serving
        mlflow.pyfunc.log_model(
            name="pyfunc_model",
            python_model="src/mlflow_model_hierarchical.py",
            artifacts={
                "model_dir": str(ft_model_path),
                "stopwords": "data/text/stopwords.json",
            },
        )

        mlflow.end_run()

    return classifier


def train_basic_classifier(
    data_path: str,
    output_dir: str,
    ngram_min_n: int = 3,
    ngram_max_n: int = 6,
    ngram_num_tokens: int = 100_000,
    embedding_dim: int = 128,
    max_seq_length: int = 64,
    batch_size: int = 32,
    lr: float = 0.1,
    num_epochs: int = 20,
    patience: int = 5,
    code_column: str = 'code8',
    mlflow_experiment: str | None = None,
    eval_data_path: str | None = None,
    eval_top_k: int = 5,
    eval_text_column: str = "product",
    encryption_key: str | None = None,
) -> BasicCOICOPClassifier:
    """Train the basic flat COICOP classifier.

    Reads a preprocessed parquet file directly (e.g. from build-training-data)
    and trains a single flat classifier on all COICOP codes.

    Args:
        data_path: Path to training parquet (product, code columns).
        output_dir: Directory to save trained model.
        ngram_min_n: Minimum n-gram size for tokenizer.
        ngram_max_n: Maximum n-gram size for tokenizer.
        ngram_num_tokens: Vocabulary size for n-gram tokenizer.
        embedding_dim: Text embedding dimension.
        max_seq_length: Maximum sequence length.
        batch_size: Training batch size.
        lr: Learning rate.
        num_epochs: Maximum epochs.
        patience: Early stopping patience.
        mlflow_experiment: MLflow experiment name (optional).
        eval_data_path: Path to evaluation data for post-training metrics.
        eval_top_k: Maximum K for top-k accuracy evaluation.
        eval_text_column: Text column name in evaluation data.

    Returns:
        Trained BasicCOICOPClassifier.
    """
    from .data_preparation import read_parquet

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data directly (already preprocessed by build-training-data)
    logger.info(f"Loading training data from {data_path}...")
    df = read_parquet(data_path, encryption_key)
    logger.info(f"Loaded {len(df)} samples")

    unique_codes = df[code_column].nunique()
    logger.info(f"Unique codes: {unique_codes}")

    # Initialize MLflow if experiment name provided
    trainer_params = None
    if mlflow_experiment:
        mlflow.set_experiment(mlflow_experiment)
        mlflow.start_run()
        mlflow.log_params({
            "classifier_type": "basic",
            "ngram_min_n": ngram_min_n,
            "ngram_max_n": ngram_max_n,
            "ngram_num_tokens": ngram_num_tokens,
            "embedding_dim": embedding_dim,
            "max_seq_length": max_seq_length,
            "batch_size": batch_size,
            "lr": lr,
            "num_epochs": num_epochs,
            "patience": patience,
            "num_samples": len(df),
            "unique_codes": unique_codes,
            "data_path": data_path,
        })
        if eval_data_path:
            mlflow.log_param("eval_data_path", eval_data_path)

        from .mlflow_utils import make_trainer_params

        run_id = mlflow.active_run().info.run_id
        trainer_params = make_trainer_params(
            experiment_name=mlflow_experiment,
            run_id=run_id,
            tracking_uri=mlflow.get_tracking_uri(),
        )

    # Create config and classifier
    config = BasicConfig(
        ngram_min_n=ngram_min_n,
        ngram_max_n=ngram_max_n,
        ngram_num_tokens=ngram_num_tokens,
        embedding_dim=embedding_dim,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        patience=patience,
    )
    classifier = BasicCOICOPClassifier(config=config)

    logger.info("Starting basic classifier training...")
    metrics = classifier.train(
        df=df,
        text_column="product",
        code_column=code_column,
        save_dir=str(output_path / "checkpoints"),
        trainer_params=trainer_params,
    )

    # Save model
    model_path = output_path / "basic_model"
    classifier.save(model_path)
    logger.info(f"Model saved to {model_path}")

    if mlflow_experiment:
        mlflow.log_metrics({
            "num_classes": metrics["num_classes"],
            "train_samples": metrics["train_samples"],
            "val_samples": metrics["val_samples"],
            "dropped_samples": metrics["dropped_samples"],
        })

        # Post-training evaluation
        if eval_data_path:
            eval_metrics = _evaluate_on_annotations(
                classifier=classifier,
                model_path=model_path,
                classifier_type="basic",
                eval_data_path=eval_data_path,
                eval_text_column=eval_text_column,
                eval_top_k=eval_top_k,
            )
            mlflow.log_metrics(eval_metrics)

        mlflow.log_artifacts(str(model_path), artifact_path="model")

        # Log pyfunc model for end-to-end serving
        mlflow.pyfunc.log_model(
            name="pyfunc_model",
            python_model="src/mlflow_model_cascade.py",
            artifacts={
                "model_dir": str(model_path),
                "stopwords": "data/text/stopwords.json",
            },
        )

        mlflow.end_run()

    return classifier


def train_multihead_classifier(
    annotations_path: str,
    output_dir: str,
    ngram_min_n: int = 3,
    ngram_max_n: int = 6,
    ngram_num_tokens: int = 100_000,
    embedding_dim: int = 128,
    max_seq_length: int = 64,
    n_attention_layers: int = 2,
    n_attention_heads: int = 4,
    n_kv_heads: int = 4,
    n_label_attention_heads: int = 4,
    batch_size: int = 32,
    lr: float = 0.01,
    num_epochs: int = 20,
    patience: int = 5,
    min_samples: int = 50,
    max_level: int = 4,
    loss_weights: list[float] | None = None,
    mlflow_experiment: str | None = None,
    eval_data_path: str | None = None,
    eval_top_k: int = 5,
    eval_text_column: str = "text",
    encryption_key: str | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> MultiHeadCOICOPClassifier:
    """Train the multi-head COICOP classifier with shared backbone.

    Args:
        annotations_path: Path to training data (parquet or csv).
        output_dir: Directory to save trained model.
        ngram_min_n: Minimum n-gram size for tokenizer.
        ngram_max_n: Maximum n-gram size for tokenizer.
        ngram_num_tokens: Vocabulary size for n-gram tokenizer.
        embedding_dim: Text embedding dimension.
        max_seq_length: Maximum sequence length.
        n_attention_layers: Number of transformer blocks in shared backbone.
        n_attention_heads: Heads per self-attention layer.
        n_kv_heads: KV heads (GQA if < n_attention_heads).
        n_label_attention_heads: Heads in each LabelAttentionClassifier.
        batch_size: Training batch size.
        lr: Learning rate.
        num_epochs: Maximum epochs.
        patience: Early stopping patience.
        min_samples: Minimum samples per level.
        max_level: Number of COICOP levels to train (1-5).
        loss_weights: Per-level loss weights (defaults to equal).
        mlflow_experiment: MLflow experiment name (optional).
        eval_data_path: Path to evaluation data for post-training metrics.
        eval_top_k: Maximum K for top-k accuracy evaluation.
        eval_text_column: Text column name in evaluation data.
        encryption_key: Parquet encryption key.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for faster CPU->GPU transfer.

    Returns:
        Trained MultiHeadCOICOPClassifier.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading annotations from {annotations_path}...")
    df = load_annotations(annotations_path, exclude_technical=True, encryption_key=encryption_key)
    logger.info(f"Loaded {len(df)} samples (excluding 98.x and 99.x codes)")

    unique_codes = df["code"].nunique()
    unique_level1 = df["level1"].nunique()
    logger.info(f"Unique codes: {unique_codes}")
    logger.info(f"Level 1 categories: {unique_level1}")

    # Initialize MLflow
    mlflow_run_info = None
    if mlflow_experiment:
        mlflow.set_experiment(mlflow_experiment)
        mlflow.start_run()
        mlflow.log_params({
            "classifier_type": "multihead",
            "ngram_min_n": ngram_min_n,
            "ngram_max_n": ngram_max_n,
            "ngram_num_tokens": ngram_num_tokens,
            "embedding_dim": embedding_dim,
            "max_seq_length": max_seq_length,
            "n_attention_layers": n_attention_layers,
            "n_attention_heads": n_attention_heads,
            "n_kv_heads": n_kv_heads,
            "n_label_attention_heads": n_label_attention_heads,
            "batch_size": batch_size,
            "lr": lr,
            "num_epochs": num_epochs,
            "patience": patience,
            "min_samples": min_samples,
            "max_level": max_level,
            "num_samples": len(df),
            "unique_codes": unique_codes,
            "unique_level1": unique_level1,
            "data_path": annotations_path,
        })
        if loss_weights is not None:
            mlflow.log_param("loss_weights", str(loss_weights))
        if eval_data_path:
            mlflow.log_param("eval_data_path", eval_data_path)

        run_id = mlflow.active_run().info.run_id
        mlflow_run_info = {
            "experiment_name": mlflow_experiment,
            "run_id": run_id,
            "tracking_uri": mlflow.get_tracking_uri(),
        }

    # Create config
    config = MultiHeadConfig(
        ngram_min_n=ngram_min_n,
        ngram_max_n=ngram_max_n,
        ngram_num_tokens=ngram_num_tokens,
        embedding_dim=embedding_dim,
        max_seq_length=max_seq_length,
        n_attention_layers=n_attention_layers,
        n_attention_heads=n_attention_heads,
        n_kv_heads=n_kv_heads,
        n_label_attention_heads=n_label_attention_heads,
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        patience=patience,
        max_level=max_level,
        loss_weights=loss_weights,
        min_samples_per_level=min_samples,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Create and train classifier
    classifier = MultiHeadCOICOPClassifier(config=config)

    logger.info("Starting multi-head classifier training...")
    model_path = output_path / "multihead_model"
    metrics = classifier.train(
        df=df,
        text_column="text",
        code_column="code",
        save_dir=str(output_path / "checkpoints"),
        mlflow_run_info=mlflow_run_info,
    )

    # Log metrics to MLflow
    if mlflow_experiment:
        for level_name, level_metrics in metrics["levels"].items():
            mlflow.log_metrics({
                f"{level_name}_num_classes": level_metrics["num_classes"],
                f"{level_name}_train_samples": level_metrics["train_samples"],
                f"{level_name}_val_samples": level_metrics["val_samples"],
            })

    # Save the classifier
    mlflow_run_id = mlflow.active_run().info.run_id if mlflow_experiment else None
    classifier.save(model_path, mlflow_run_id=mlflow_run_id)
    logger.info(f"Model saved to {model_path}")

    if mlflow_experiment:
        # Post-training evaluation
        if eval_data_path:
            eval_metrics = _evaluate_on_annotations(
                classifier=classifier,
                model_path=model_path,
                classifier_type="multihead",
                eval_data_path=eval_data_path,
                eval_text_column=eval_text_column,
                eval_top_k=eval_top_k,
            )
            mlflow.log_metrics(eval_metrics)

        mlflow.log_artifacts(str(model_path), artifact_path="model")

        # Log pyfunc model
        mlflow.pyfunc.log_model(
            name="pyfunc_model",
            python_model="src/mlflow_model_multihead.py",
            artifacts={
                "model_dir": str(model_path),
                "stopwords": "data/text/stopwords.json",
            },
        )

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
