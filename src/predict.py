"""Inference module for COICOP classification."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .cascade_classifier import CascadeCOICOPClassifier

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
