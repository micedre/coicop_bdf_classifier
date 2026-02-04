"""Cascade classifier for hierarchical COICOP classification."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .classifier import COICOPClassifier
from .data_preparation import COICOPDataset

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# Minimum samples required to train a sub-classifier
MIN_SAMPLES_PER_SUBCLASSIFIER = 50


class CascadeCOICOPClassifier:
    """Cascade classifier for hierarchical COICOP classification.

    This classifier implements a hierarchical prediction strategy:
    1. First predicts level 1 category (01-13)
    2. Based on level 1 prediction, uses sub-classifier to predict deeper levels
    3. Continues until leaf level or no sub-classifier is available
    """

    def __init__(
        self,
        model_name: str = "camembert-base",
        embedding_dim: int = 128,
        max_seq_length: int = 64,
        min_samples: int = MIN_SAMPLES_PER_SUBCLASSIFIER,
    ):
        """Initialize the cascade classifier.

        Args:
            model_name: HuggingFace model name for tokenizer
            embedding_dim: Dimension of text embeddings
            max_seq_length: Maximum sequence length
            min_samples: Minimum samples to train a sub-classifier
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.min_samples = min_samples

        # Level 1 classifier
        self.level1_classifier: COICOPClassifier | None = None

        # Sub-classifiers: {parent_code: classifier}
        # e.g., {"01": classifier_for_01_subcategories, ...}
        self.sub_classifiers: dict[str, COICOPClassifier] = {}

        # Hierarchy information
        self.hierarchy: dict[str, dict] = {}

        self._is_trained = False

    def _get_level_from_code(self, code: str) -> int:
        """Get the hierarchy level from a COICOP code."""
        return len(code.split("."))

    def _get_parent_code(self, code: str) -> str | None:
        """Get the parent code from a COICOP code."""
        parts = code.split(".")
        if len(parts) <= 1:
            return None
        return ".".join(parts[:-1])

    def _build_hierarchy(self, codes: list[str]) -> dict[str, dict]:
        """Build hierarchy structure from codes.

        Returns:
            Dictionary mapping parent codes to their children
        """
        hierarchy: dict[str, list[str]] = {}

        for code in codes:
            parent = self._get_parent_code(code)
            if parent is not None:
                if parent not in hierarchy:
                    hierarchy[parent] = []
                if code not in hierarchy[parent]:
                    hierarchy[parent].append(code)

        return {k: {"children": sorted(v)} for k, v in hierarchy.items()}

    def train(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        code_column: str = "code",
        batch_size: int = 32,
        lr: float = 2e-5,
        num_epochs: int = 20,
        patience: int = 5,
        save_dir: str | None = None,
    ) -> dict:
        """Train the cascade classifier.

        Args:
            df: DataFrame with text and code columns
            text_column: Name of text column
            code_column: Name of code column
            batch_size: Training batch size
            lr: Learning rate
            num_epochs: Maximum epochs per classifier
            patience: Early stopping patience
            save_dir: Directory to save models

        Returns:
            Dictionary with training metrics
        """
        import pandas as pd

        # Build hierarchy from all codes
        all_codes = df[code_column].unique().tolist()
        self.hierarchy = self._build_hierarchy(all_codes)

        # Extract level 1 labels
        df = df.copy()
        df["level1"] = df[code_column].apply(lambda x: x.split(".")[0].zfill(2))

        metrics = {"level1": {}, "sub_classifiers": {}}

        # Step 1: Train level 1 classifier
        logger.info("Training level 1 classifier...")
        level1_labels = sorted(df["level1"].unique().tolist())

        level1_dataset = COICOPDataset.from_dataframe(
            df,
            text_column=text_column,
            label_column="level1",
            min_samples_per_class=2,
        )

        self.level1_classifier = COICOPClassifier(
            num_classes=len(level1_labels),
            label_names=level1_labels,
            model_name=self.model_name,
            embedding_dim=self.embedding_dim,
            max_seq_length=self.max_seq_length,
        )

        save_path = None
        if save_dir:
            save_path = str(Path(save_dir) / "level1")

        self.level1_classifier.train(
            dataset=level1_dataset,
            batch_size=batch_size,
            lr=lr,
            num_epochs=num_epochs,
            patience=patience,
            save_path=save_path,
        )

        metrics["level1"] = {
            "num_classes": len(level1_labels),
            "train_samples": len(level1_dataset.train_labels),
            "val_samples": len(level1_dataset.val_labels),
        }

        # Step 2: Train sub-classifiers for each level 1 category
        logger.info("Training sub-classifiers...")

        for level1_code in level1_labels:
            # Filter data for this level 1 category
            subset = df[df["level1"] == level1_code].copy()

            if len(subset) < self.min_samples:
                logger.info(
                    f"Skipping sub-classifier for {level1_code}: "
                    f"insufficient samples ({len(subset)} < {self.min_samples})"
                )
                continue

            # Get unique full codes for this category
            unique_codes = sorted(subset[code_column].unique().tolist())

            # Skip if only one unique code (no sub-classification needed)
            if len(unique_codes) <= 1:
                logger.info(
                    f"Skipping sub-classifier for {level1_code}: "
                    f"only one unique code"
                )
                continue

            # Check if we have enough samples per class
            code_counts = subset[code_column].value_counts()
            valid_codes = code_counts[code_counts >= 2].index.tolist()

            if len(valid_codes) <= 1:
                logger.info(
                    f"Skipping sub-classifier for {level1_code}: "
                    f"insufficient codes with >= 2 samples"
                )
                continue

            # Filter to valid codes only
            subset = subset[subset[code_column].isin(valid_codes)].copy()

            logger.info(
                f"Training sub-classifier for {level1_code} "
                f"({len(subset)} samples, {len(valid_codes)} codes)"
            )

            try:
                sub_dataset = COICOPDataset.from_dataframe(
                    subset,
                    text_column=text_column,
                    label_column=code_column,
                    min_samples_per_class=2,
                )

                sub_classifier = COICOPClassifier(
                    num_classes=len(sub_dataset.label_names),
                    label_names=sub_dataset.label_names,
                    model_name=self.model_name,
                    embedding_dim=self.embedding_dim,
                    max_seq_length=self.max_seq_length,
                )

                sub_save_path = None
                if save_dir:
                    sub_save_path = str(Path(save_dir) / f"sub_{level1_code}")

                sub_classifier.train(
                    dataset=sub_dataset,
                    batch_size=batch_size,
                    lr=lr,
                    num_epochs=num_epochs,
                    patience=patience,
                    save_path=sub_save_path,
                )

                self.sub_classifiers[level1_code] = sub_classifier

                metrics["sub_classifiers"][level1_code] = {
                    "num_classes": len(sub_dataset.label_names),
                    "train_samples": len(sub_dataset.train_labels),
                    "val_samples": len(sub_dataset.val_labels),
                    "codes": sub_dataset.label_names,
                }

            except Exception as e:
                logger.warning(
                    f"Failed to train sub-classifier for {level1_code}: {e}"
                )
                continue

        self._is_trained = True

        logger.info(
            f"Cascade classifier trained: "
            f"1 level1 classifier + {len(self.sub_classifiers)} sub-classifiers"
        )

        return metrics

    def predict(
        self,
        texts: list[str],
        return_all_levels: bool = True,
    ) -> dict:
        """Predict COICOP codes for input texts.

        Args:
            texts: List of text strings to classify
            return_all_levels: Whether to return predictions at all levels

        Returns:
            Dictionary with:
                - predictions: List of predicted codes (most specific available)
                - level1: Level 1 predictions
                - confidence: Confidence scores for final prediction
                - all_levels: (if return_all_levels) predictions at each level
        """
        if not self._is_trained:
            raise RuntimeError("Classifier must be trained before prediction.")

        # Step 1: Predict level 1
        level1_result = self.level1_classifier.predict(texts, top_k=1)
        level1_predictions = level1_result["predictions"]
        level1_confidence = level1_result["confidence"]

        # Initialize results
        final_predictions = list(level1_predictions)
        final_confidence = level1_confidence.flatten().tolist()
        all_levels = {"level1": level1_predictions}

        # Step 2: For each text, use sub-classifier if available
        for i, (text, level1_code) in enumerate(zip(texts, level1_predictions)):
            if level1_code in self.sub_classifiers:
                sub_classifier = self.sub_classifiers[level1_code]
                sub_result = sub_classifier.predict([text], top_k=1)
                final_predictions[i] = sub_result["predictions"][0]
                final_confidence[i] = float(sub_result["confidence"][0, 0])

        result = {
            "predictions": final_predictions,
            "level1": level1_predictions,
            "confidence": final_confidence,
        }

        if return_all_levels:
            result["all_levels"] = all_levels

        return result

    def predict_with_hierarchy(
        self,
        texts: list[str],
    ) -> list[dict]:
        """Predict with full hierarchical information.

        Args:
            texts: List of text strings to classify

        Returns:
            List of dictionaries, each containing:
                - code: Final predicted code
                - level1: Level 1 code
                - level1_confidence: Confidence for level 1
                - final_confidence: Confidence for final prediction
        """
        result = self.predict(texts, return_all_levels=True)

        predictions = []
        for i in range(len(texts)):
            pred = {
                "code": result["predictions"][i],
                "level1": result["level1"][i],
                "confidence": result["confidence"][i],
            }
            predictions.append(pred)

        return predictions

    def save(self, path: str | Path) -> None:
        """Save the cascade classifier.

        Args:
            path: Directory path to save all models
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save level 1 classifier
        if self.level1_classifier:
            self.level1_classifier.save(path / "level1")

        # Save sub-classifiers
        sub_path = path / "sub_classifiers"
        sub_path.mkdir(exist_ok=True)

        for code, classifier in self.sub_classifiers.items():
            classifier.save(sub_path / code)

        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "max_seq_length": self.max_seq_length,
            "min_samples": self.min_samples,
            "hierarchy": self.hierarchy,
            "sub_classifier_codes": list(self.sub_classifiers.keys()),
        }

        with open(path / "cascade_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Cascade classifier saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> CascadeCOICOPClassifier:
        """Load a trained cascade classifier.

        Args:
            path: Directory path where models were saved

        Returns:
            Loaded CascadeCOICOPClassifier instance
        """
        path = Path(path)

        # Load metadata
        with open(path / "cascade_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        # Create instance
        instance = cls(
            model_name=metadata["model_name"],
            embedding_dim=metadata["embedding_dim"],
            max_seq_length=metadata["max_seq_length"],
            min_samples=metadata["min_samples"],
        )

        instance.hierarchy = metadata["hierarchy"]

        # Load level 1 classifier
        instance.level1_classifier = COICOPClassifier.load(path / "level1")

        # Load sub-classifiers
        sub_path = path / "sub_classifiers"
        for code in metadata["sub_classifier_codes"]:
            instance.sub_classifiers[code] = COICOPClassifier.load(sub_path / code)

        instance._is_trained = True

        logger.info(
            f"Cascade classifier loaded: "
            f"1 level1 + {len(instance.sub_classifiers)} sub-classifiers"
        )

        return instance
