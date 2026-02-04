"""Synthetic data generation for COICOP classification using LangChain."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_experimental.synthetic_data import create_data_generation_chain
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class COICOPExample(BaseModel):
    """Schema for a synthetic COICOP product example."""

    product: str = Field(
        description="Product or service name/description in French"
    )
    code: str = Field(description="COICOP code (e.g., '01.1.1.1.1')")
    libelle: str = Field(description="COICOP category label in French")


def get_llm_from_env(model_name: str | None = None) -> ChatOpenAI:
    """Create LLM instance from environment variables.

    Args:
        model_name: Model name override. If None, uses OPENAI_MODEL env var.

    Environment variables:
        OPENAI_API_KEY: API key for OpenAI-compatible endpoint
        OPENAI_BASE_URL: Base URL for API (optional, defaults to OpenAI)
        OPENAI_MODEL: Model name (optional, defaults to gpt-oss:20b)

    Returns:
        Configured ChatOpenAI instance
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        msg = "OPENAI_API_KEY environment variable is required"
        raise ValueError(msg)

    base_url = os.environ.get("OPENAI_BASE_URL")
    if model_name is None:
        model_name = os.environ.get("OPENAI_MODEL", "gpt-oss:20b")

    logger.info(f"Configuring llm: {base_url} with model : {model_name}")


    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0.8,
    )


class COICOPSyntheticGenerator:
    """Generator for synthetic COICOP classification training data.

    Uses LangChain's synthetic data generation capabilities to create
    realistic product descriptions for each COICOP category.
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        coicop_path: str | Path = "data/20260130-coicop_et_codes_techniques.csv",
        examples_per_category: int = 10,
    ) -> None:
        """Initialize the synthetic data generator.

        Args:
            llm: LangChain chat model (defaults to OpenAI from env vars)
            coicop_path: Path to COICOP definitions CSV
            examples_per_category: Number of examples to generate per category
        """
        self.llm = llm if llm is not None else get_llm_from_env()
        self.coicop_path = Path(coicop_path)
        self.examples_per_category = examples_per_category
        self._coicop_df: pd.DataFrame | None = None

    @property
    def coicop_df(self) -> pd.DataFrame:
        """Lazy load the COICOP hierarchy."""
        if self._coicop_df is None:
            self._coicop_df = self._load_coicop()
        return self._coicop_df

    def _load_coicop(self) -> pd.DataFrame:
        """Load COICOP hierarchy from CSV."""
        df = pd.read_csv(self.coicop_path, sep=";", encoding="utf-8")
        df.columns = ["libelle", "code"]
        return df

    def _get_leaf_categories(self) -> pd.DataFrame:
        """Get only leaf-level COICOP categories (most specific).

        Leaf categories have 5-level codes (e.g., '01.1.1.1.1').
        """
        df = self.coicop_df.copy()
        # Filter for codes with 5 parts (leaf level)
        mask = df["code"].str.count(r"\.") == 4
        return df[mask].copy()

    def _get_categories_by_level(self, level: int) -> pd.DataFrame:
        """Get COICOP categories at a specific hierarchy level.

        Args:
            level: Hierarchy level (1-5). Level 1 = '01', Level 5 = '01.1.1.1.1'

        Returns:
            DataFrame with categories at the specified level
        """
        df = self.coicop_df.copy()
        # Level 1 has 0 dots, level 5 has 4 dots
        mask = df["code"].str.count(r"\.") == (level - 1)
        return df[mask].copy()

    def _build_generation_prompt(self) -> PromptTemplate:
        """Build the prompt template for synthetic data generation."""
        template = """Tu es un expert en classification des produits et services selon la nomenclature COICOP (Classification of Individual Consumption According to Purpose).

Génère {num_examples} exemples réalistes de produits ou services qui appartiennent à la catégorie COICOP suivante:

Code COICOP: {code}
Libellé: {libelle}

Pour chaque exemple, fournis un nom de produit ou une description courte et réaliste en français, comme on pourrait le trouver sur un ticket de caisse ou dans un relevé de dépenses.

Les exemples doivent être:
- Variés (différentes marques, variantes, formulations)
- Réalistes (comme sur un vrai ticket de caisse)
- En français
- Courts (1 à 5 mots généralement)

Exemples de produits pour cette catégorie:"""
        return PromptTemplate(
            input_variables=["num_examples", "code", "libelle"],
            template=template,
        )

    def _create_few_shot_prompt(
        self,
        examples: list[dict[str, str]],
    ) -> FewShotPromptTemplate:
        """Create a few-shot prompt with examples.

        Args:
            examples: List of example dictionaries with 'product', 'code', 'libelle'

        Returns:
            Configured FewShotPromptTemplate
        """
        example_template = PromptTemplate(
            input_variables=["product", "code", "libelle"],
            template="Produit: {product}\nCode: {code}\nCatégorie: {libelle}",
        )

        prefix = """Tu es un expert en classification COICOP. Voici quelques exemples de produits avec leurs codes COICOP:

"""
        suffix = """
Maintenant, génère {num_examples} nouveaux exemples de produits pour la catégorie suivante:
Code: {code}
Libellé: {libelle}

Produits (un par ligne):"""

        return FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_template,
            prefix=prefix,
            suffix=suffix,
            input_variables=["num_examples", "code", "libelle"],
            example_separator="\n\n",
        )

    def generate_for_category(
        self,
        code: str,
        libelle: str,
        num_examples: int | None = None,
    ) -> list[dict[str, str]]:
        """Generate synthetic examples for a single COICOP category.

        Args:
            code: COICOP code
            libelle: COICOP category label
            num_examples: Number of examples (defaults to examples_per_category)

        Returns:
            List of dictionaries with 'product', 'code', 'libelle' keys
        """
        if num_examples is None:
            num_examples = self.examples_per_category

        prompt = self._build_generation_prompt()

        # Format the prompt
        formatted_prompt = prompt.format(
            num_examples=num_examples,
            code=code,
            libelle=libelle,
        )

        # Generate using LLM
        response = self.llm.invoke(formatted_prompt)
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Parse the response into individual products
        products = self._parse_response(response_text)

        return [
            {"product": product, "code": code, "libelle": libelle}
            for product in products[:num_examples]
        ]

    def _parse_response(self, response: str) -> list[str]:
        """Parse LLM response into list of product names.

        Args:
            response: Raw LLM response text

        Returns:
            List of product name strings
        """
        lines = response.strip().split("\n")
        products = []

        for line in lines:
            # Clean up the line
            line = line.strip()
            if not line:
                continue

            # Remove common prefixes like "1.", "- ", "• ", etc.
            for prefix in ["- ", "• ", "* ", "– "]:
                if line.startswith(prefix):
                    line = line[len(prefix) :]
                    break

            # Remove numbered prefixes like "1. ", "2. "
            if len(line) > 2 and line[0].isdigit() and line[1] in [".", ")", ":"]:
                line = line[2:].strip()
            elif (
                len(line) > 3
                and line[0].isdigit()
                and line[1].isdigit()
                and line[2] in [".", ")", ":"]
            ):
                line = line[3:].strip()

            if line:
                products.append(line)

        return products

    def generate_dataset(
        self,
        level: int = 5,
        max_categories: int | None = None,
        exclude_technical: bool = True,
    ) -> pd.DataFrame:
        """Generate a complete synthetic dataset.

        Args:
            level: COICOP hierarchy level to generate for (1-5)
            max_categories: Maximum number of categories to process (for testing)
            exclude_technical: Whether to exclude 98.x and 99.x technical codes

        Returns:
            DataFrame with 'product', 'code', 'libelle' columns
        """
        categories = self._get_categories_by_level(level)

        if exclude_technical:
            mask = ~categories["code"].str.startswith(("98", "99"))
            categories = categories[mask]

        if max_categories is not None:
            categories = categories.head(max_categories)

        all_examples: list[dict[str, str]] = []

        for _, row in categories.iterrows():
            code = row["code"]
            libelle = row["libelle"]

            logger.info(f"Generating examples for {code}: {libelle}")

            try:
                examples = self.generate_for_category(code, libelle)
                all_examples.extend(examples)
                logger.info(f"  Generated {len(examples)} examples")
            except Exception as e:
                logger.warning(f"  Failed to generate for {code}: {e}")

        return pd.DataFrame(all_examples)

    def generate_with_dataset_generator(
        self,
        level: int = 5,
        max_categories: int | None = None,
        exclude_technical: bool = True,
    ) -> pd.DataFrame:
        """Generate dataset using LangChain's DatasetGenerator.

        This is an alternative method using the experimental DatasetGenerator
        for more structured output.

        Args:
            level: COICOP hierarchy level to generate for (1-5)
            max_categories: Maximum number of categories to process
            exclude_technical: Whether to exclude 98.x and 99.x technical codes

        Returns:
            DataFrame with synthetic examples
        """
        categories = self._get_categories_by_level(level)

        if exclude_technical:
            mask = ~categories["code"].str.startswith(("98", "99"))
            categories = categories[mask]

        if max_categories is not None:
            categories = categories.head(max_categories)

        # Create the data generation chain
        chain = create_data_generation_chain(self.llm, COICOPExample)

        all_examples: list[dict[str, str]] = []

        for _, row in categories.iterrows():
            code = row["code"]
            libelle = row["libelle"]

            logger.info(f"Generating structured examples for {code}: {libelle}")

            # Build subject description for the generator
            subject = f"produit ou service de la catégorie COICOP '{libelle}' (code {code})"

            try:
                for _ in range(self.examples_per_category):
                    result = chain.invoke({"subject": subject})
                    if isinstance(result, dict) and "text" in result:
                        # Extract the generated text
                        all_examples.append(
                            {
                                "product": result.get("text", ""),
                                "code": code,
                                "libelle": libelle,
                            }
                        )
            except Exception as e:
                logger.warning(f"  Failed to generate for {code}: {e}")

        return pd.DataFrame(all_examples)


def generate_and_save(
    output_path: str | Path,
    coicop_path: str | Path = "data/20260130-coicop_et_codes_techniques.csv",
    examples_per_category: int = 10,
    level: int = 5,
    max_categories: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic data and save to file.

    Uses environment variables for LLM configuration:
        OPENAI_API_KEY: API key (required)
        OPENAI_BASE_URL: Base URL for API (optional)
        OPENAI_MODEL_NAME: Model name (optional, defaults to gpt-3.5-turbo)

    Args:
        output_path: Path to save the generated data (parquet or csv)
        coicop_path: Path to COICOP definitions
        examples_per_category: Number of examples per category
        level: COICOP hierarchy level
        max_categories: Maximum categories to process (for testing)

    Returns:
        Generated DataFrame
    """
    generator = COICOPSyntheticGenerator(
        coicop_path=coicop_path,
        examples_per_category=examples_per_category,
    )

    df = generator.generate_dataset(
        level=level,
        max_categories=max_categories,
    )

    output_path = Path(output_path)
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False, sep=";", encoding="utf-8")

    logger.info(f"Saved {len(df)} examples to {output_path}")
    return df


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Generate synthetic COICOP training data"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/synthetic_coicop.parquet",
        help="Output file path",
    )
    parser.add_argument(
        "--coicop",
        type=str,
        default="data/20260130-coicop_et_codes_techniques.csv",
        help="Path to COICOP definitions CSV",
    )
    parser.add_argument(
        "--examples",
        "-n",
        type=int,
        default=10,
        help="Number of examples per category",
    )
    parser.add_argument(
        "--level",
        "-l",
        type=int,
        default=5,
        help="COICOP hierarchy level (1-5)",
    )
    parser.add_argument(
        "--max-categories",
        "-m",
        type=int,
        default=None,
        help="Maximum categories to process (for testing)",
    )

    args = parser.parse_args()

    generate_and_save(
        output_path=args.output,
        coicop_path=args.coicop,
        examples_per_category=args.examples,
        level=args.level,
        max_categories=args.max_categories,
    )
