"""Data preparation module for COICOP classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split

from typing import List, Union
import unidecode
import json
import os
import numpy as np


if TYPE_CHECKING:
    from collections.abc import Sequence


def read_parquet(path: str | Path, encryption_key: str | None = None) -> pd.DataFrame:
    """Read parquet file, with optional DuckDB decryption."""
    if encryption_key:
        con = duckdb.connect()
        con.execute(f"PRAGMA add_parquet_key('encryption_key', '{encryption_key}');")
        return con.execute(f"SELECT * FROM read_parquet('{path}')").df()
    return pd.read_parquet(path)


def extract_levels(code: str) -> dict[str, str | None]:
    """Extract hierarchical levels from a COICOP code.

    Args:
        code: COICOP code string (e.g., "01.1.2.3.4")

    Returns:
        Dictionary with level1 through level5 keys
    """
    parts = code.split(".")
    return {
        "level1": parts[0].zfill(2),
        "level2": ".".join(parts[:2]) if len(parts) >= 2 else None,
        "level3": ".".join(parts[:3]) if len(parts) >= 3 else None,
        "level4": ".".join(parts[:4]) if len(parts) >= 4 else None,
        "level5": ".".join(parts[:5]) if len(parts) >= 5 else None,
    }


def load_coicop_hierarchy(path: str | Path) -> pd.DataFrame:
    """Load COICOP hierarchy definitions.

    Args:
        path: Path to the COICOP definitions CSV file

    Returns:
        DataFrame with columns: code, libelle, level1-level5
    """
    df = pd.read_csv(path, sep=";", encoding="utf-8")
    df.columns = ["libelle", "code"]

    # Extract hierarchical levels
    levels = df["code"].apply(extract_levels).apply(pd.Series)
    df = pd.concat([df, levels], axis=1)

    return df


def load_annotations(
    path: str | Path,
    exclude_technical: bool = True,
    encryption_key: str | None = None,
) -> pd.DataFrame:
    """Load and preprocess annotation data.

    Args:
        path: Path to annotations.parquet file
        exclude_technical: Whether to exclude 98.x and 99.x technical codes
        encryption_key: Parquet encryption key for reading encrypted files

    Returns:
        DataFrame with product text and hierarchical labels
    """
    df = read_parquet(path, encryption_key)

    with open("data/text/stopwords.json", "r", encoding="utf-8") as json_file:
        stopwords = json.load(json_file)

    df = preprocess_text(df, 'product', stopwords)

    # The 'code' column contains the COICOP codes
    # 'coicop' column contains the label text (description)

    # Filter out technical codes if requested
    if exclude_technical:
        mask = ~df["code"].str.startswith(("98", "99"))
        df = df[mask].copy()

    # Extract hierarchical levels from the code
    levels = df["code"].apply(extract_levels).apply(pd.Series)
    df = pd.concat([df, levels], axis=1)

    # Clean product text
    df["text"] = df["product"].str.strip().str.lower()

    return df


@dataclass
class COICOPDataset:
    """Container for train/validation data splits."""

    train_texts: list[str]
    train_labels: list[str]
    val_texts: list[str]
    val_labels: list[str]
    label_names: list[str]

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "level1",
        test_size: float = 0.2,
        random_state: int = 42,
        min_samples_per_class: int = 2,
    ) -> COICOPDataset:
        """Create dataset from DataFrame with stratified split.

        Args:
            df: DataFrame with text and label columns
            text_column: Name of column containing text
            label_column: Name of column containing labels
            test_size: Fraction of data to use for validation
            random_state: Random seed for reproducibility
            min_samples_per_class: Minimum samples required for a class

        Returns:
            COICOPDataset instance
        """
        # Filter out classes with too few samples for stratified split
        df = df.dropna(subset=[label_column])
        label_counts = df[label_column].value_counts()
        valid_labels = label_counts[label_counts >= min_samples_per_class].index
        df = df[df[label_column].isin(valid_labels)].copy()

        texts = df[text_column].tolist()
        labels = df[label_column].tolist()

        # Stratified split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        # Get sorted unique labels
        label_names = sorted(set(labels))

        return cls(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            label_names=label_names,
        )


def prepare_cascade_data(
    df: pd.DataFrame,
    parent_level: str,
    parent_value: str,
    child_level: str,
    min_samples: int = 50,
) -> pd.DataFrame | None:
    """Prepare data for a sub-classifier in the cascade.

    Args:
        df: Full DataFrame with hierarchical labels
        parent_level: Parent level column name (e.g., 'level1')
        parent_value: Value to filter on (e.g., '01')
        child_level: Child level column name (e.g., 'level2')
        min_samples: Minimum total samples required

    Returns:
        Filtered DataFrame or None if insufficient data
    """
    subset = df[df[parent_level] == parent_value].copy()
    subset = subset.dropna(subset=[child_level])

    if len(subset) < min_samples:
        return None

    return subset


def get_class_weights(labels: Sequence[str]) -> dict[str, float]:
    """Calculate class weights for imbalanced data.

    Args:
        labels: Sequence of label strings

    Returns:
        Dictionary mapping label to weight
    """
    label_counts = pd.Series(labels).value_counts()
    total = len(labels)
    n_classes = len(label_counts)

    weights = {}
    for label, count in label_counts.items():
        weights[label] = total / (n_classes * count)

    return weights



def preprocess_text(
    df: pd.DataFrame, text_feature: str, stopwords: Union[List[str], set[str]]
) -> pd.DataFrame:
    """
    Pipeline principal de prétraitement textuel.

    Args:
        df: DataFrame contenant le texte à traiter.
        text_feature: Nom de la colonne texte à prétraiter.
        stopwords: Liste ou ensemble de mots à exclure.

    Returns:
        DataFrame prétraitée.
    """
    df[text_feature + "_orig"] = df[text_feature].copy()
    df[text_feature] = df[text_feature].fillna("").map(unidecode.unidecode)
    df[text_feature] = df[text_feature].str.lower()
    df = remove_noise(df, text_feature)
    df = tokenize_and_clean(df, text_feature)
    df = remove_empty_and_strip(df, text_feature)
    df = remove_stopwords(df, text_feature, stopwords)
    return df


def remove_noise(df: pd.DataFrame, text_feature: str) -> pd.DataFrame:
    """
    Supprime le bruit textuel : mots inutiles, ponctuation, chiffres, etc.

    Args:
        df: DataFrame contenant la colonne texte.
        text_feature: Nom de la colonne texte.

    Returns:
        DataFrame nettoyé.
    """
    lib_to_remove = r"\brien\b|\rien du tout\b"
    words_to_remove = r"\brien\b|\rien du tout\b"

    # On supprime les libellés vide de sens
    df[text_feature] = df[text_feature].str.replace(lib_to_remove, "", regex=True)
    # On supprime toutes les ponctuations
    df[text_feature] = df[text_feature].str.replace(r"[^\w\s]+", " ", regex=True)
    # On supprime les stopwords custom
    df[text_feature] = df[text_feature].str.replace(words_to_remove, "", regex=True)
    # CHIFFRES : QUOI FAIRE ? TEMPORAIREMENT ON SUPPRIME
    df[text_feature] = df[text_feature].str.replace(r"[\d+]", " ", regex=True)
    # On supprime les mots d'une seule lettre
    df[text_feature] = df[text_feature].apply(
        lambda x: " ".join([w for w in x.split() if len(w) > 1])
    )
    # On supprime les multiple space
    df[text_feature] = df[text_feature].str.replace(r"\s\s+", " ", regex=True)
    return df


def tokenize_and_clean(df: pd.DataFrame, text_feature: str) -> pd.DataFrame:
    """
    Tokenise chaque texte, supprime les doublons tout en conservant l'ordre.

    Args:
        df: DataFrame contenant la colonne texte.
        text_feature: Nom de la colonne texte.

    Returns:
        DataFrame avec texte nettoyé.
    """
    libs_token = [lib.split() for lib in df[text_feature].to_list()]
    libs_token = [
        sorted(set(libs_token[i]), key=libs_token[i].index)
        for i in range(len(libs_token))
    ]
    df[text_feature] = [" ".join(libs_token[i]) for i in range(len(libs_token))]
    return df


def remove_empty_and_strip(df, text_feature):
    df[text_feature] = df[text_feature].str.strip()
    df[text_feature] = df[text_feature].replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(subset=[text_feature])
    return df


def remove_stopwords(df, text_feature, stopwords):
    libs_token = [lib.split() for lib in df[text_feature].to_list()]
    df[text_feature] = [
        " ".join([word for word in libs_token[i] if word not in stopwords])
        for i in range(len(libs_token))
    ]
    return df
