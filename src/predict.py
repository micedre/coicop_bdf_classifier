"""Inference module for COICOP classification."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .data_preparation import preprocess_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_STOPWORDS_PATH = Path(__file__).parent.parent / "data" / "text" / "stopwords.json"

_MLFLOW_PREFIXES = ("runs:/", "models:/", "mlflow-artifacts:/")


def _resolve_mlflow_path(model_path: str | Path) -> Path:
    """Resolve an MLflow artifact URI to a local path, or return as-is."""
    model_path_str = str(model_path)
    if any(model_path_str.startswith(p) for p in _MLFLOW_PREFIXES):
        import mlflow
        logger.info(f"Downloading MLflow artifacts from {model_path_str}...")
        model_path = mlflow.artifacts.download_artifacts(artifact_uri=model_path_str)
        logger.info(f"Downloaded to {model_path}")
    return Path(model_path)


def _load_stopwords() -> list[str]:
    """Load stopwords from the project data directory."""
    with open(_STOPWORDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_input_file(input_path: Path) -> pd.DataFrame:
    """Read a CSV or parquet input file."""
    if input_path.suffix == ".parquet":
        return pd.read_parquet(input_path)
    elif input_path.suffix == ".csv":
        return pd.read_csv(input_path, sep=";")
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")


def _write_output_file(df: pd.DataFrame, output_path: Path) -> None:
    """Write a DataFrame to CSV or parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)


class _HierarchicalBasePredictor:
    """Shared base for hierarchical and multi-head predictors.

    Subclasses only need to set ``self.classifier`` in ``__init__``.
    """

    classifier: object
    model_path: Path

    def predict(
        self,
        texts: list[str],
        return_all_levels: bool = True,
        top_k: int = 1,
        confidence_threshold: float | None = None,
    ) -> list[dict]:
        """Predict COICOP codes for input texts."""
        result = self.classifier.predict(
            texts,
            return_all_levels=return_all_levels,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
        )

        predictions = []
        for i, text in enumerate(texts):
            pred = {
                "text": text,
                "code": result["final_code"][i],
                "final_level": result["final_level"][i],
                "confidence": result["final_confidence"][i],
                "combined_confidence": result["combined_confidence"][i],
            }

            if return_all_levels and "all_levels" in result:
                pred["levels"] = {}
                for level_name, level_data in result["all_levels"].items():
                    if top_k > 1:
                        pred["levels"][level_name] = {
                            "code": level_data["predictions"][i][0],
                            "confidence": level_data["confidence"][i][0],
                            "alternatives": [
                                {"code": level_data["predictions"][i][k], "confidence": level_data["confidence"][i][k]}
                                for k in range(1, top_k)
                            ],
                        }
                    else:
                        pred["levels"][level_name] = {
                            "code": level_data["predictions"][i],
                            "confidence": level_data["confidence"][i],
                        }

            predictions.append(pred)

        return predictions

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        return_all_levels: bool = True,
        top_k: int = 1,
        confidence_threshold: float | None = None,
    ) -> list[dict]:
        """Predict in batches for large datasets."""
        all_predictions = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            predictions = self.predict(
                batch,
                return_all_levels=return_all_levels,
                top_k=top_k,
                confidence_threshold=confidence_threshold,
            )
            all_predictions.extend(predictions)

            if (i + batch_size) % (batch_size * 10) == 0:
                logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)}")

        return all_predictions

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "product",
        batch_size: int = 64,
        top_k: int = 1,
        confidence_threshold: float | None = None,
    ) -> pd.DataFrame:
        """Predict codes for a DataFrame."""
        texts = df[text_column].tolist()
        predictions = self.predict_batch(
            texts,
            batch_size=batch_size,
            return_all_levels=True,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
        )

        result_df = df.copy()
        result_df["predicted_code"] = [p["code"] for p in predictions]
        result_df["final_level"] = [p["final_level"] for p in predictions]
        result_df["confidence"] = [p["confidence"] for p in predictions]
        result_df["combined_confidence"] = [p["combined_confidence"] for p in predictions]

        if predictions and "levels" in predictions[0]:
            for level_name in predictions[0]["levels"]:
                result_df[f"predicted_{level_name}"] = [
                    p["levels"].get(level_name, {}).get("code", "") for p in predictions
                ]
                result_df[f"confidence_{level_name}"] = [
                    p["levels"].get(level_name, {}).get("confidence", 0.0) for p in predictions
                ]

                if top_k > 1:
                    for k in range(1, top_k):
                        rank = k + 1
                        result_df[f"predicted_{level_name}_top{rank}"] = [
                            p["levels"].get(level_name, {}).get("alternatives", [{}] * k)[k - 1].get("code", "")
                            if len(p["levels"].get(level_name, {}).get("alternatives", [])) >= k
                            else ""
                            for p in predictions
                        ]
                        result_df[f"confidence_{level_name}_top{rank}"] = [
                            p["levels"].get(level_name, {}).get("alternatives", [{}] * k)[k - 1].get("confidence", 0.0)
                            if len(p["levels"].get(level_name, {}).get("alternatives", [])) >= k
                            else 0.0
                            for p in predictions
                        ]

        return result_df

    def predict_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        text_column: str = "product",
        batch_size: int = 64,
        top_k: int = 1,
        confidence_threshold: float | None = None,
    ) -> None:
        """Predict codes for a file and save results."""
        input_path = Path(input_path)
        output_path = Path(output_path)

        logger.info(f"Loading texts from {input_path}...")
        df = _read_input_file(input_path)
        df = preprocess_text(df, text_column, _load_stopwords())
        logger.info(f"Loaded {len(df)} samples from {input_path}")

        result_df = self.predict_dataframe(
            df,
            text_column=text_column,
            batch_size=batch_size,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
        )

        _write_output_file(result_df, output_path)
        logger.info(f"Saved predictions to {output_path}")


class HierarchicalCOICOPPredictor(_HierarchicalBasePredictor):
    """Predictor class for hierarchical COICOP classification."""

    def __init__(self, model_path: str | Path):
        from .hierarchical_classifier import HierarchicalCOICOPClassifier

        self.model_path = _resolve_mlflow_path(model_path)
        self.classifier = HierarchicalCOICOPClassifier.load(self.model_path)
        logger.info(f"Loaded hierarchical model from {model_path}")


class MultiHeadCOICOPPredictor(_HierarchicalBasePredictor):
    """Predictor class for multi-head COICOP classification."""

    def __init__(self, model_path: str | Path):
        from .multihead_classifier import MultiHeadCOICOPClassifier

        self.model_path = _resolve_mlflow_path(model_path)
        self.classifier = MultiHeadCOICOPClassifier.load(self.model_path)
        logger.info(f"Loaded multi-head model from {model_path}")


class BasicCOICOPPredictor:
    """Predictor class for basic flat COICOP classification."""

    def __init__(self, model_path: str | Path):
        from .basic_classifier import BasicCOICOPClassifier

        self.model_path = _resolve_mlflow_path(model_path)
        self.classifier = BasicCOICOPClassifier.load(self.model_path)
        logger.info(f"Loaded basic model from {model_path}")

    def predict(
        self,
        texts: list[str],
        top_k: int = 1,
    ) -> list[dict]:
        """Predict COICOP codes for input texts."""
        result = self.classifier.predict(texts, top_k=top_k)

        predictions = []
        for i, text in enumerate(texts):
            if top_k > 1:
                pred = {
                    "text": text,
                    "code": result["predictions"][i][0],
                    "confidence": result["confidence"][i][0],
                    "alternatives": [
                        {"code": result["predictions"][i][k], "confidence": result["confidence"][i][k]}
                        for k in range(1, top_k)
                    ],
                }
            else:
                pred = {
                    "text": text,
                    "code": result["predictions"][i],
                    "confidence": result["confidence"][i],
                }
            predictions.append(pred)

        return predictions

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        top_k: int = 1,
    ) -> list[dict]:
        """Predict in batches for large datasets."""
        all_predictions = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            predictions = self.predict(batch, top_k=top_k)
            all_predictions.extend(predictions)

            if (i + batch_size) % (batch_size * 10) == 0:
                logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)}")

        return all_predictions

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "product",
        batch_size: int = 64,
        top_k: int = 1,
    ) -> pd.DataFrame:
        """Predict codes for a DataFrame."""
        texts = df[text_column].tolist()
        predictions = self.predict_batch(texts, batch_size=batch_size, top_k=top_k)

        result_df = df.copy()
        result_df["predicted_code"] = [p["code"] for p in predictions]
        result_df["confidence"] = [p["confidence"] for p in predictions]

        if top_k > 1:
            for k in range(1, top_k):
                rank = k + 1
                result_df[f"predicted_code_top{rank}"] = [
                    p.get("alternatives", [{}] * k)[k - 1].get("code", "")
                    if len(p.get("alternatives", [])) >= k
                    else ""
                    for p in predictions
                ]
                result_df[f"confidence_top{rank}"] = [
                    p.get("alternatives", [{}] * k)[k - 1].get("confidence", 0.0)
                    if len(p.get("alternatives", [])) >= k
                    else 0.0
                    for p in predictions
                ]

        return result_df

    def predict_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        text_column: str = "product",
        batch_size: int = 64,
        top_k: int = 1,
    ) -> None:
        """Predict codes for a file and save results."""
        input_path = Path(input_path)
        output_path = Path(output_path)

        logger.info(f"Loading texts from {input_path}...")
        df = _read_input_file(input_path)
        df = preprocess_text(df, text_column, _load_stopwords())
        logger.info(f"Loaded {len(df)} samples from {input_path}")

        result_df = self.predict_dataframe(
            df,
            text_column=text_column,
            batch_size=batch_size,
            top_k=top_k,
        )

        _write_output_file(result_df, output_path)
        logger.info(f"Saved predictions to {output_path}")


def predict_texts_hierarchical(
    model_path: str,
    texts: list[str],
) -> list[dict]:
    """Convenience function to predict COICOP codes using hierarchical classifier."""
    predictor = HierarchicalCOICOPPredictor(model_path)
    return predictor.predict(texts)
