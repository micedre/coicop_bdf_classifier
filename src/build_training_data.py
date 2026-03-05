"""Build a balanced training dataset from DDC extraction and synthetic data."""

from __future__ import annotations

import json
import logging

import pandas as pd

from src.data_preparation import preprocess_text

logger = logging.getLogger(__name__)

STOPWORDS_PATH = "data/text/stopwords.json"


def _load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _extract_level4(code: pd.Series) -> pd.Series:
    return code.apply(lambda c: ".".join(str(c).split(".")[:4]))


def build_training_data(
    ddc_path: str,
    output_path: str,
    synthetic_path: str = "data/synthetic_data.csv",
    max_per_code: int = 1000,
    seed: int = 42,
) -> None:
    """Build a balanced training dataset from DDC and synthetic data.

    Args:
        ddc_path: Path to DDC parquet (local, S3, or HTTP).
        output_path: Output parquet file path.
        synthetic_path: Path to synthetic data CSV (semicolon-separated).
        max_per_code: Max DDC rows per level-4 code before sampling.
        seed: Random seed for reproducible sampling.
    """
    stopwords = _load_stopwords()

    # --- DDC data ---
    logger.info("Reading DDC data from %s", ddc_path)
    ddc = pd.read_parquet(ddc_path)
    ddc = ddc.rename(columns={"description_ean": "product", "coicop_code": "code"})
    ddc = ddc[["product", "code"]].copy()
    ddc["source"] = "ddc"

    logger.info("DDC rows before preprocessing: %d", len(ddc))
    ddc = preprocess_text(ddc, "product", stopwords)
    ddc = ddc.drop_duplicates(subset=["product", "code"])
    logger.info("DDC rows after preprocessing + dedup: %d", len(ddc))

    # --- Synthetic data ---
    logger.info("Reading synthetic data from %s", synthetic_path)
    synthetic = pd.read_csv(synthetic_path, sep=";")
    synthetic = synthetic[["product", "code"]].copy()
    synthetic["source"] = "synthetic"

    logger.info("Synthetic rows before preprocessing: %d", len(synthetic))
    synthetic = preprocess_text(synthetic, "product", stopwords)
    synthetic = synthetic.drop_duplicates(subset=["product", "code"])
    logger.info("Synthetic rows after preprocessing + dedup: %d", len(synthetic))

    # --- Balance at level 4 ---
    ddc["code_level4"] = _extract_level4(ddc["code"])
    synthetic["code_level4"] = _extract_level4(synthetic["code"])

    ddc_counts = ddc.groupby("code_level4").size()
    all_level4_codes = set(ddc["code_level4"]) | set(synthetic["code_level4"])

    parts: list[pd.DataFrame] = []

    for code_l4 in sorted(all_level4_codes):
        ddc_subset = ddc[ddc["code_level4"] == code_l4]
        synthetic_subset = synthetic[synthetic["code_level4"] == code_l4]
        n_ddc = len(ddc_subset)

        if n_ddc > max_per_code:
            # Oversample: random sample DDC, no synthetic
            parts.append(ddc_subset.sample(n=max_per_code, random_state=seed))
        else:
            # Keep all DDC + all synthetic
            if n_ddc > 0:
                parts.append(ddc_subset)
            if len(synthetic_subset) > 0:
                parts.append(synthetic_subset)

    result = pd.concat(parts, ignore_index=True)
    result = result.drop(columns=["code_level4"])

    # --- Save ---
    result.to_parquet(output_path, index=False)

    # --- Summary ---
    n_ddc = (result["source"] == "ddc").sum()
    n_synthetic = (result["source"] == "synthetic").sum()
    n_codes = result["code"].nunique()
    logger.info(
        "Training data saved to %s: %d rows (%d DDC, %d synthetic), %d unique codes",
        output_path,
        len(result),
        n_ddc,
        n_synthetic,
        n_codes,
    )
