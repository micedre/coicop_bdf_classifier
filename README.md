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
