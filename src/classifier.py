"""Base classifier module wrapping torchtextclassifiers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import HuggingFaceTokenizer

if TYPE_CHECKING:
    from .data_preparation import COICOPDataset

logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL_NAME = "camembert-base"
DEFAULT_EMBEDDING_DIM = 128
DEFAULT_MAX_SEQ_LENGTH = 64
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 2e-5
DEFAULT_NUM_EPOCHS = 20
DEFAULT_PATIENCE = 5


class COICOPClassifier:
    """COICOP text classifier using torchtextclassifiers with CamemBERT tokenizer."""

    def __init__(
        self,
        num_classes: int,
        label_names: list[str],
        model_name: str = DEFAULT_MODEL_NAME,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    ):
        """Initialize the classifier.

        Args:
            num_classes: Number of output classes
            label_names: List of class labels in order
            model_name: HuggingFace model name for tokenizer
            embedding_dim: Dimension of text embeddings
            max_seq_length: Maximum sequence length for tokenization
        """
        self.num_classes = num_classes
        self.label_names = label_names
        self.label_to_idx = {label: idx for idx, label in enumerate(label_names)}
        self.idx_to_label = {idx: label for idx, label in enumerate(label_names)}
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length

        # Initialize tokenizer from pretrained CamemBERT
        self.tokenizer = HuggingFaceTokenizer.load_from_pretrained(
            model_name, output_dim=max_seq_length
        )

        # Model config
        self.model_config = ModelConfig(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
        )

        # Initialize the classifier
        self.classifier = torchTextClassifiers(
            tokenizer=self.tokenizer,
            model_config=self.model_config,
        )

        self._is_trained = False

    def _labels_to_indices(self, labels: list[str]) -> np.ndarray:
        """Convert string labels to integer indices."""
        return np.array([self.label_to_idx[label] for label in labels])

    def _indices_to_labels(self, indices: np.ndarray) -> list[str]:
        """Convert integer indices to string labels."""
        return [self.idx_to_label[int(idx)] for idx in indices]

    def train(
        self,
        dataset: COICOPDataset,
        batch_size: int = DEFAULT_BATCH_SIZE,
        lr: float = DEFAULT_LR,
        num_epochs: int = DEFAULT_NUM_EPOCHS,
        patience: int = DEFAULT_PATIENCE,
        save_path: str | None = None,
        trainer_params: dict | None = None,
    ) -> dict:
        """Train the classifier.

        Args:
            dataset: COICOPDataset with train/val splits
            batch_size: Training batch size
            lr: Learning rate
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            save_path: Path to save the trained model

        Returns:
            Dictionary with training metrics
        """
        # Prepare training data
        X_train = np.array(dataset.train_texts)
        y_train = self._labels_to_indices(dataset.train_labels)

        X_val = np.array(dataset.val_texts)
        y_val = self._labels_to_indices(dataset.val_labels)

        # Training config
        training_config = TrainingConfig(
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            patience_early_stopping=patience,
            save_path=save_path or "coicop_classifier",
            **({"trainer_params": trainer_params} if trainer_params else {}),
        )

        # Train
        logger.info(f"Training classifier with {len(X_train)} samples...")
        self.classifier.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            training_config=training_config,
            verbose=True,
        )

        self._is_trained = True

        return {"status": "trained"}

    def predict(
        self,
        texts: list[str],
        top_k: int = 1,
    ) -> dict:
        """Predict labels for input texts.

        Args:
            texts: List of text strings to classify
            top_k: Number of top predictions to return

        Returns:
            Dictionary with:
                - predictions: List of predicted labels
                - confidence: Confidence scores
        """
        if not self._is_trained:
            raise RuntimeError("Classifier must be trained before prediction.")

        X = np.array(texts)
        result = self.classifier.predict(X, top_k=top_k)

        # Convert indices to labels
        predictions = result["prediction"].numpy()
        confidence = result["confidence"].numpy()

        # Handle top_k predictions
        if top_k == 1:
            labels = self._indices_to_labels(predictions.flatten())
        else:
            labels = [
                self._indices_to_labels(predictions[i])
                for i in range(len(predictions))
            ]

        return {
            "predictions": labels,
            "confidence": confidence,
        }

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Get prediction probabilities for all classes.

        Args:
            texts: List of text strings

        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        if not self._is_trained:
            raise RuntimeError("Classifier must be trained before prediction.")

        X = np.array(texts)
        result = self.classifier.predict(X, top_k=self.num_classes)

        # Reconstruct full probability distribution
        predictions = result["prediction"].numpy()
        confidence = result["confidence"].numpy()

        proba = np.zeros((len(texts), self.num_classes))
        for i in range(len(texts)):
            for j in range(self.num_classes):
                class_idx = predictions[i, j]
                proba[i, class_idx] = confidence[i, j]

        return proba

    def save(self, path: str | Path) -> None:
        """Save the trained classifier.

        Args:
            path: Directory path to save the model
        """
        import pickle

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save classifier using built-in method
        self.classifier.save(path / "model")

        # Save additional metadata
        metadata = {
            "label_names": self.label_names,
            "label_to_idx": self.label_to_idx,
            "idx_to_label": self.idx_to_label,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "max_seq_length": self.max_seq_length,
            "num_classes": self.num_classes,
        }

        with open(path / "classifier_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Classifier saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> COICOPClassifier:
        """Load a trained classifier.

        Args:
            path: Directory path where model was saved

        Returns:
            Loaded COICOPClassifier instance
        """
        import pickle

        path = Path(path)

        # Load metadata
        with open(path / "classifier_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        # Create instance
        instance = cls(
            num_classes=metadata["num_classes"],
            label_names=metadata["label_names"],
            model_name=metadata["model_name"],
            embedding_dim=metadata["embedding_dim"],
            max_seq_length=metadata["max_seq_length"],
        )

        # Load the trained classifier
        instance.classifier = torchTextClassifiers.load(path / "model")
        instance._is_trained = True

        logger.info(f"Classifier loaded from {path}")
        return instance
