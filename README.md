# COICOP BDF Classifier

A hierarchical text classifier for COICOP (Classification of Individual Consumption According to Purpose) codes using a cascade architecture with CamemBERT tokenization.

## Overview

This project implements a cascade classifier that predicts COICOP codes for French product descriptions. The classifier uses a hierarchical approach:

1. **Level 1 Classifier**: Predicts the main category (01-13)
2. **Sub-classifiers**: For each level 1 category with sufficient data, a specialized sub-classifier predicts the full COICOP code

This architecture handles the hierarchical nature of COICOP codes and class imbalance better than a flat multi-class classifier.

## Features

- Cascade/hierarchical classification architecture
- CamemBERT tokenizer for French text processing
- Automatic exclusion of technical codes (98.x, 99.x)
- MLflow integration for experiment tracking
- CLI interface for training, prediction, and evaluation
- Batch prediction support for large datasets

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd coicop_bdf_classifier

# Install dependencies with uv
uv sync
```

### Dependencies

- `torchtextclassifiers>=1.0.3` - Text classification framework
- `transformers>=4.30.0` - For CamemBERT tokenizer
- `mlflow>=3.9.0` - Experiment tracking
- `pandas>=2.0.0` - Data manipulation
- `pyarrow>=12.0.0` - Parquet file support
- `scikit-learn>=1.3.0` - Train/test splitting and metrics

## Data

The classifier expects the following data files in the `data/` directory:

### annotations.parquet

Training data with columns:
- `product`: Product description (text)
- `code`: COICOP code (e.g., "01.1.2.3.4")
- `coicop`: COICOP label description

### 20260130-coicop_et_codes_techniques.csv

COICOP hierarchy definitions with columns:
- `Libelle`: Code description
- `Code`: COICOP code

## Usage

### Training

Train the cascade classifier:

```bash
uv run python main.py train \
    --annotations data/annotations.parquet \
    --output-dir models \
    --batch-size 32 \
    --num-epochs 20 \
    --patience 5
```

#### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--annotations` | `data/annotations.parquet` | Path to training data |
| `--output-dir` | `models` | Output directory for models |
| `--model-name` | `camembert-base` | HuggingFace tokenizer model |
| `--embedding-dim` | `128` | Text embedding dimension |
| `--max-seq-length` | `64` | Maximum sequence length |
| `--batch-size` | `32` | Training batch size |
| `--lr` | `2e-5` | Learning rate |
| `--num-epochs` | `20` | Maximum epochs per classifier |
| `--patience` | `5` | Early stopping patience |
| `--min-samples` | `50` | Minimum samples for sub-classifiers |
| `--mlflow-experiment` | `None` | MLflow experiment name |

### Prediction

Predict COICOP codes for text:

```bash
# Single or multiple texts
uv run python main.py predict "pain complet bio" "bouteille eau minerale"

# From file
uv run python main.py predict \
    --file input.csv \
    --output predictions.csv \
    --text-column product
```

#### Prediction Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-path` | `models/model` | Path to trained model |
| `--file` | `None` | Input file for batch prediction |
| `--output` | `predictions.csv` | Output file path |
| `--text-column` | `product` | Name of text column |
| `--batch-size` | `64` | Batch size for prediction |

### Evaluation

Evaluate the classifier on a test set:

```bash
uv run python main.py evaluate \
    --model-path models/model \
    --test-file data/test.parquet \
    --text-column product \
    --label-column coicop \
    --exclude-technical \
    --detailed
```

#### Evaluation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-path` | `models/model` | Path to trained model |
| `--test-file` | (required) | Path to test file |
| `--text-column` | `product` | Name of text column |
| `--label-column` | `coicop` | Name of label column |
| `--batch-size` | `64` | Batch size |
| `--exclude-technical` | `False` | Exclude 98.x/99.x codes |
| `--detailed` | `False` | Show detailed report |

## Extraction des données de caisse (DDC)

La commande `extract-ddc` extrait les données de caisse (DDC) depuis le stockage S3 via DuckDB, applique un mapping COICOP, et produit un fichier parquet prêt à l'emploi.

### Fonctionnement détaillé

1. **Lecture S3** : les fichiers parquet DDC sont lus depuis `s3://projet-ddc/.../annee={ANNEE}/mois={MOIS}/` pour les années et mois demandés.
2. **Dédoublonnage** : les lignes sont dédoublonnées sur le triplet `(description_ean, variete, id_famille)`.
3. **Mapping COICOP** :
   - Si la variété commence par `"99"`, le code COICOP est récupéré depuis la table `famille_circana.csv` par jointure sur `id_famille`.
   - Sinon, le champ `variete` est utilisé directement comme code COICOP.
4. **Filtrage** : seuls les codes COICOP d'au moins 10 caractères sont conservés, et les codes commençant par `"99"` sont exclus.
5. **Écriture** : le résultat est écrit en parquet sur S3.

### Colonnes de sortie

| Colonne | Description |
|---------|-------------|
| `description_ean` | Texte du produit |
| `variete` | Code variété d'origine |
| `coicop_code` | Code COICOP après mapping |

### Commande CLI

```bash
uv run python main.py extract-ddc \
    --annee 2024 2025 \
    --mois 1 2 3 \
    --famille data/famille_circana.csv \
    --memory 6GB
```

#### Arguments

| Argument | Obligatoire | Défaut | Description |
|----------|:-----------:|--------|-------------|
| `--annee` | oui | — | Année(s) à extraire |
| `--mois` | non | tous les mois | Mois à extraire |
| `--output` | non | `s3://travail/.../ddc_{DATE}.parquet` | Chemin S3 de sortie |
| `--famille` | non | `data/famille_circana.csv` | Fichier CSV de mapping famille Circana |
| `--memory` | non | `6GB` | Limite mémoire DuckDB |
| `--dry-run` | non | `False` | Affiche le SQL généré sans l'exécuter |

### Mode `--dry-run`

Le mode `--dry-run` affiche l'intégralité du SQL qui serait exécuté (création des vues, jointures, `COPY`) sans se connecter à S3. Utile pour vérifier la requête avant exécution.

```bash
uv run python main.py extract-ddc --annee 2024 --dry-run
```

## Construction du jeu d'entraînement

La commande `build-training-data` construit un jeu de données équilibré prêt pour l'entraînement à partir des données de caisse (DDC) et de données synthétiques.

### Pipeline de prétraitement textuel

Chaque texte passe par la fonction `preprocess_text` (définie dans `src/data_preparation.py`) qui enchaîne les étapes suivantes :

1. **Translittération Unicode** (`unidecode`) — suppression des accents et caractères spéciaux
   - `"Crème brûlée BIO"` → `"Creme brulee BIO"`
2. **Passage en minuscules**
   - `"Creme brulee BIO"` → `"creme brulee bio"`
3. **Suppression du bruit** (`remove_noise`) :
   - Suppression des expressions vides de sens (`"rien"`, `"rien du tout"`)
   - Suppression de la ponctuation (remplacée par des espaces)
   - Suppression des chiffres
   - Suppression des mots d'une seule lettre
   - Nettoyage des espaces multiples
   - `"creme brulee bio - 2x125g"` → `"creme brulee bio"`
4. **Tokenisation et dédoublonnage** (`tokenize_and_clean`) — suppression des mots répétés tout en conservant l'ordre d'apparition
   - `"lait lait entier lait"` → `"lait entier"`
5. **Suppression des lignes vides** (`remove_empty_and_strip`)
6. **Suppression des stopwords** (`remove_stopwords`) — mots courants définis dans `data/text/stopwords.json`
   - `"boite de conserve de haricots"` → `"boite conserve haricots"`

### Logique d'équilibrage

L'équilibrage opère au **niveau 4 de la COICOP** (préfixe formé des 4 premiers segments du code, ex. `01.1.2.3`). Pour chaque code de niveau 4 :

- **Code surreprésenté** (nombre de lignes DDC > `max_per_code`) → échantillonnage aléatoire de `max_per_code` lignes DDC, pas de données synthétiques ajoutées.
- **Code sous-représenté** (nombre de lignes DDC ≤ `max_per_code`) → conservation de toutes les lignes DDC + ajout de **toutes** les lignes synthétiques disponibles pour ce code.
- **Code absent de la DDC** (0 lignes DDC) → inclusion de toutes les lignes synthétiques.

### Flux de données

```
┌──────────────────┐     ┌──────────────────────┐
│  DDC (parquet)   │     │ Synthétique (CSV)     │
└────────┬─────────┘     └──────────┬────────────┘
         │                          │
         ▼                          ▼
┌──────────────────────────────────────────────┐
│           preprocess_text()                  │
│  unidecode → lower → bruit → dedup tokens   │
│  → vides → stopwords                        │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  Dédoublonnage      │
            │  (product, code)    │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  Équilibrage        │
            │  par code niveau 4  │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  Sortie (parquet)   │
            │  product, code,     │
            │  source             │
            └─────────────────────┘
```

### Schéma de sortie

| Colonne | Description |
|---------|-------------|
| `product` | Texte du produit prétraité |
| `code` | Code COICOP complet |
| `source` | Origine de la donnée : `"ddc"` ou `"synthetic"` |

### Commande CLI

```bash
uv run python main.py build-training-data \
    --ddc data/raw/ddc.parquet \
    --output data/data-train.parquet \
    --synthetic data/synthetic_data.csv \
    --max-per-code 1000 \
    --seed 42
```

#### Arguments

| Argument | Obligatoire | Défaut | Description |
|----------|:-----------:|--------|-------------|
| `--ddc` | oui | — | Chemin vers le parquet DDC (local, S3 ou HTTP) |
| `--output` | oui | — | Chemin du fichier parquet de sortie |
| `--synthetic` | non | `data/synthetic_data.csv` | Chemin vers le CSV de données synthétiques (séparateur `;`) |
| `--max-per-code` | non | `1000` | Nombre max de lignes DDC par code de niveau 4 |
| `--seed` | non | `42` | Graine aléatoire pour la reproductibilité |

## Python API

### Training

```python
from src.cascade_classifier import CascadeCOICOPClassifier
from src.data_preparation import load_annotations

# Load data
df = load_annotations("data/annotations.parquet", exclude_technical=True)

# Create and train classifier
classifier = CascadeCOICOPClassifier(
    model_name="camembert-base",
    embedding_dim=128,
    max_seq_length=64,
    min_samples=50,
)

metrics = classifier.train(
    df=df,
    text_column="text",
    code_column="code",
    batch_size=32,
    lr=2e-5,
    num_epochs=20,
    patience=5,
    save_dir="models",
)

# Save model
classifier.save("models/model")
```

### Prediction

```python
from src.predict import COICOPPredictor

# Load model
predictor = COICOPPredictor("models/model")

# Predict single texts
predictions = predictor.predict(["pain complet bio", "bouteille eau"])
for pred in predictions:
    print(f"{pred['text']}: {pred['code']} (confidence: {pred['confidence']:.2f})")

# Predict from DataFrame
import pandas as pd
df = pd.read_csv("products.csv")
result_df = predictor.predict_dataframe(df, text_column="product")
```

## Project Structure

```
coicop_bdf_classifier/
├── main.py                    # CLI entry point
├── src/
│   ├── __init__.py           # Package exports
│   ├── data_preparation.py   # Data loading and preprocessing
│   ├── classifier.py         # Base COICOPClassifier
│   ├── cascade_classifier.py # Cascade classifier implementation
│   ├── train.py              # Training script
│   └── predict.py            # Inference module
├── data/
│   ├── annotations.parquet   # Training data
│   └── *.csv                 # COICOP definitions
├── models/                    # Trained models (generated)
├── pyproject.toml            # Project dependencies
└── README.md
```

## Architecture Details

### Cascade Classifier

The `CascadeCOICOPClassifier` implements a two-stage prediction:

1. **Level 1 Classification**: A classifier trained on 13 main COICOP categories (01-13)
2. **Sub-classification**: For each level 1 category with >= 50 samples and multiple unique codes, a specialized classifier predicts the full COICOP code

This approach:
- Handles class imbalance by training specialized models per category
- Respects the hierarchical structure of COICOP codes
- Falls back to level 1 prediction when no sub-classifier is available

### Text Processing

- **Tokenizer**: CamemBERT (French BERT) via HuggingFace Transformers
- **Max Sequence Length**: 64 tokens (product names are typically short)
- **Text Preprocessing**: Lowercase, strip whitespace

### Training Configuration

- **Batch Size**: 32 (smaller due to BERT memory usage)
- **Learning Rate**: 2e-5 (standard BERT fine-tuning rate)
- **Early Stopping**: Patience of 5 epochs monitoring validation loss
- **Minimum Samples**: 50 per sub-classifier to ensure meaningful training

## Data Statistics

After filtering technical codes (98.x, 99.x):
- **Total Samples**: ~18,965
- **Unique Codes**: ~580
- **Level 1 Categories**: 13 (01-13)

## MLflow Tracking

Enable experiment tracking by providing an experiment name:

```bash
uv run python main.py train --mlflow-experiment "coicop-classifier"
```

Tracked metrics:
- Number of classes per classifier
- Training/validation sample counts
- Number of sub-classifiers trained

## License

[Add your license here]
