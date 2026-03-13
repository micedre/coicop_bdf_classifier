"""Inference module for COICOP classification."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .basic_classifier import BasicCOICOPClassifier
from .cascade_classifier import CascadeCOICOPClassifier
from .hierarchical_classifier import HierarchicalCOICOPClassifier
from .data_preparation import preprocess_text
if TYPE_CHECKING:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class COICOPPredictor:
    """Predictor class for COICOP classification."""

    def __init__(self, model_path: str | Path):
        """Initialize the predictor with a trained model.

        Args:
            model_path: Path to the saved cascade classifier
        """
        self.model_path = Path(model_path)
        self.classifier = CascadeCOICOPClassifier.load(self.model_path)
        logger.info(f"Loaded model from {model_path}")

    def predict(
        self,
        texts: list[str],
        return_all_levels: bool = False,
    ) -> list[dict]:
        """Predict COICOP codes for input texts.

        Args:
            texts: List of product descriptions
            return_all_levels: Whether to include all hierarchy levels

        Returns:
            List of prediction dictionaries
        """
        result = self.classifier.predict(texts, return_all_levels=return_all_levels)

        predictions = []
        for i, text in enumerate(texts):
            pred = {
                "text": text,
                "code": result["predictions"][i],
                "level1": result["level1"][i],
                "confidence": result["confidence"][i],
            }
            predictions.append(pred)

        return predictions

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        return_all_levels: bool = False,
    ) -> list[dict]:
        """Predict in batches for large datasets.

        Args:
            texts: List of product descriptions
            batch_size: Number of texts per batch
            return_all_levels: Whether to include all hierarchy levels

        Returns:
            List of prediction dictionaries
        """
        all_predictions = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            predictions = self.predict(batch, return_all_levels=return_all_levels)
            all_predictions.extend(predictions)

            if (i + batch_size) % (batch_size * 10) == 0:
                logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)}")

        return all_predictions

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "product",
        batch_size: int = 64,
    ) -> pd.DataFrame:
        """Predict codes for a DataFrame.

        Args:
            df: DataFrame with text column
            text_column: Name of the text column
            batch_size: Batch size for prediction

        Returns:
            DataFrame with predictions added
        """
        texts = df[text_column].tolist()
        predictions = self.predict_batch(texts, batch_size=batch_size)

        # Add predictions to DataFrame
        result_df = df.copy()
        result_df["predicted_code"] = [p["code"] for p in predictions]
        result_df["predicted_level1"] = [p["level1"] for p in predictions]
        result_df["confidence"] = [p["confidence"] for p in predictions]

        return result_df

    def predict_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        text_column: str = "product",
        batch_size: int = 64,
    ) -> None:
        """Predict codes for a file and save results.

        Args:
            input_path: Path to input file (CSV or parquet)
            output_path: Path to save output file
            text_column: Name of the text column
            batch_size: Batch size for prediction
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Load input file
        if input_path.suffix == ".parquet":
            df = pd.read_parquet(input_path)
        elif input_path.suffix == ".csv":
            df = pd.read_csv(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")

        logger.info(f"Loaded {len(df)} samples from {input_path}")

        # Predict
        result_df = self.predict_dataframe(df, text_column=text_column, batch_size=batch_size)

        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".parquet":
            result_df.to_parquet(output_path, index=False)
        elif output_path.suffix == ".csv":
            result_df.to_csv(output_path, index=False)
        else:
            result_df.to_csv(output_path, index=False)

        logger.info(f"Saved predictions to {output_path}")


class HierarchicalCOICOPPredictor:
    """Predictor class for hierarchical COICOP classification."""

    def __init__(self, model_path: str | Path):
        """Initialize the predictor with a trained hierarchical model.

        Args:
            model_path: Path to the saved hierarchical classifier, or an MLflow
                artifact URI (runs:/, models:/, mlflow-artifacts:/)
        """
        model_path_str = str(model_path)
        if any(model_path_str.startswith(p) for p in ("runs:/", "models:/", "mlflow-artifacts:/")):
            import mlflow
            logger.info(f"Downloading MLflow artifacts from {model_path_str}...")
            model_path = mlflow.artifacts.download_artifacts(artifact_uri=model_path_str)
            logger.info(f"Downloaded to {model_path}")
        self.model_path = Path(model_path)
        self.classifier = HierarchicalCOICOPClassifier.load(self.model_path)
        logger.info(f"Loaded hierarchical model from {model_path}")

    def predict(
        self,
        texts: list[str],
        return_all_levels: bool = True,
        top_k: int = 1,
        confidence_threshold: float | None = None,
    ) -> list[dict]:
        """Predict COICOP codes for input texts.

        Args:
            texts: List of product descriptions
            return_all_levels: Whether to include predictions at all levels
            top_k: Number of top predictions per level
            confidence_threshold: Minimum confidence per level; stop at the
                deepest level meeting this threshold.

        Returns:
            List of prediction dictionaries
        """
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

            # Add level-by-level predictions if requested
            if return_all_levels and "all_levels" in result:
                pred["levels"] = {}
                for level_name, level_data in result["all_levels"].items():
                    if top_k > 1:
                        # level_data["predictions"][i] is list[str], confidence is list[float]
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
        """Predict in batches for large datasets.

        Args:
            texts: List of product descriptions
            batch_size: Number of texts per batch
            return_all_levels: Whether to include all hierarchy levels
            top_k: Number of top predictions per level
            confidence_threshold: Minimum confidence per level; stop at the
                deepest level meeting this threshold.

        Returns:
            List of prediction dictionaries
        """
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
        """Predict codes for a DataFrame.

        Args:
            df: DataFrame with text column
            text_column: Name of the text column
            batch_size: Batch size for prediction
            top_k: Number of top predictions per level
            confidence_threshold: Minimum confidence per level; stop at the
                deepest level meeting this threshold.

        Returns:
            DataFrame with predictions added
        """
        texts = df[text_column].tolist()
        predictions = self.predict_batch(
            texts,
            batch_size=batch_size,
            return_all_levels=True,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
        )

        # Add predictions to DataFrame
        result_df = df.copy()
        result_df["predicted_code"] = [p["code"] for p in predictions]
        result_df["final_level"] = [p["final_level"] for p in predictions]
        result_df["confidence"] = [p["confidence"] for p in predictions]
        result_df["combined_confidence"] = [p["combined_confidence"] for p in predictions]

        # Add individual level columns
        if predictions and "levels" in predictions[0]:
            for level_name in predictions[0]["levels"]:
                result_df[f"predicted_{level_name}"] = [
                    p["levels"].get(level_name, {}).get("code", "") for p in predictions
                ]
                result_df[f"confidence_{level_name}"] = [
                    p["levels"].get(level_name, {}).get("confidence", 0.0) for p in predictions
                ]

                # Add top-K alternative columns
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
        """Predict codes for a file and save results.

        Args:
            input_path: Path to input file (CSV or parquet)
            output_path: Path to save output file
            text_column: Name of the text column
            batch_size: Batch size for prediction
            top_k: Number of top predictions per level
            confidence_threshold: Minimum confidence per level; stop at the
                deepest level meeting this threshold.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Load input file
        logger.info(f"Loading texts from {input_path}...")

        if input_path.suffix == ".parquet":
            df = pd.read_parquet(input_path)
        elif input_path.suffix == ".csv":
            df = pd.read_csv(input_path, sep=';')
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")

         # Load data
        with open("data/text/stopwords.json", "r", encoding="utf-8") as json_file:
            stopwords = json.load(json_file)

        df = preprocess_text(df, text_column, stopwords)

        logger.info(f"Loaded {len(df)} samples from {input_path}")

        # Predict
        result_df = self.predict_dataframe(
            df,
            text_column=text_column,
            batch_size=batch_size,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
        )

        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".parquet":
            result_df.to_parquet(output_path, index=False)
        elif output_path.suffix == ".csv":
            result_df.to_csv(output_path, index=False)
        else:
            result_df.to_csv(output_path, index=False)

        logger.info(f"Saved predictions to {output_path}")


class BasicCOICOPPredictor:
    """Predictor class for basic flat COICOP classification."""

    def __init__(self, model_path: str | Path):
        """Initialize the predictor with a trained basic model.

        Args:
            model_path: Path to the saved basic classifier
        """
        self.model_path = Path(model_path)
        self.classifier = BasicCOICOPClassifier.load(self.model_path)
        logger.info(f"Loaded basic model from {model_path}")

    def predict(
        self,
        texts: list[str],
        top_k: int = 1,
    ) -> list[dict]:
        """Predict COICOP codes for input texts.

        Args:
            texts: List of preprocessed product descriptions.
            top_k: Number of top predictions.

        Returns:
            List of prediction dictionaries.
        """
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
        """Predict in batches for large datasets.

        Args:
            texts: List of product descriptions.
            batch_size: Number of texts per batch.
            top_k: Number of top predictions.

        Returns:
            List of prediction dictionaries.
        """
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
        """Predict codes for a DataFrame.

        Args:
            df: DataFrame with text column.
            text_column: Name of the text column.
            batch_size: Batch size for prediction.
            top_k: Number of top predictions.

        Returns:
            DataFrame with predictions added.
        """
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
        """Predict codes for a file and save results.

        Applies text preprocessing on raw text before prediction.

        Args:
            input_path: Path to input file (CSV or parquet).
            output_path: Path to save output file.
            text_column: Name of the text column.
            batch_size: Batch size for prediction.
            top_k: Number of top predictions.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Load input file
        logger.info(f"Loading texts from {input_path}...")
        if input_path.suffix == ".parquet":
            df = pd.read_parquet(input_path)
        elif input_path.suffix == ".csv":
            df = pd.read_csv(input_path, sep=";")
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")

        # Preprocess raw text
        with open("data/text/stopwords.json", "r", encoding="utf-8") as json_file:
            stopwords = json.load(json_file)

        df = preprocess_text(df, text_column, stopwords)

        logger.info(f"Loaded {len(df)} samples from {input_path}")

        # Predict
        result_df = self.predict_dataframe(
            df,
            text_column=text_column,
            batch_size=batch_size,
            top_k=top_k,
        )

        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".parquet":
            result_df.to_parquet(output_path, index=False)
        elif output_path.suffix == ".csv":
            result_df.to_csv(output_path, index=False)
        else:
            result_df.to_csv(output_path, index=False)

        logger.info(f"Saved predictions to {output_path}")


def predict_texts(
    model_path: str,
    texts: list[str],
) -> list[dict]:
    """Convenience function to predict COICOP codes.

    Args:
        model_path: Path to saved model
        texts: List of product descriptions

    Returns:
        List of prediction dictionaries
    """
    predictor = COICOPPredictor(model_path)
    return predictor.predict(texts)


def predict_texts_hierarchical(
    model_path: str,
    texts: list[str],
) -> list[dict]:
    """Convenience function to predict COICOP codes using hierarchical classifier.

    Args:
        model_path: Path to saved hierarchical model
        texts: List of product descriptions

    Returns:
        List of prediction dictionaries with all levels
    """
    predictor = HierarchicalCOICOPPredictor(model_path)
    return predictor.predict(texts)


def main():
    parser = argparse.ArgumentParser(description="Predict COICOP codes")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/model",
        help="Path to saved model",
    )

    subparsers = parser.add_subparsers(dest="command", help="Prediction mode")

    # Single text prediction
    text_parser = subparsers.add_parser("text", help="Predict for single texts")
    text_parser.add_argument(
        "texts",
        nargs="+",
        help="Text(s) to classify",
    )

    # File prediction
    file_parser = subparsers.add_parser("file", help="Predict for file")
    file_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file path",
    )
    file_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path",
    )
    file_parser.add_argument(
        "--text-column",
        type=str,
        default="product",
        help="Name of text column",
    )
    file_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for prediction",
    )

    args = parser.parse_args()

    predictor = COICOPPredictor(args.model_path)

    if args.command == "text":
        predictions = predictor.predict(args.texts)
        for pred in predictions:
            print(json.dumps(pred, ensure_ascii=False, indent=2))

    elif args.command == "file":
        predictor.predict_file(
            input_path=args.input,
            output_path=args.output,
            text_column=args.text_column,
            batch_size=args.batch_size,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
