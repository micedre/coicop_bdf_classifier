# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hierarchical text classifier for COICOP (Classification of Individual Consumption According to Purpose) codes from French product descriptions. Built for INSEE (French National Institute of Statistics). Uses `torchtextclassifiers` as the core ML framework.

## Commands

All commands use `uv` as the package manager (with INSEE internal PyPI proxy).

```bash
# Install dependencies
uv sync

# Train cascade classifier (CamemBERT tokenizer, 2-level)
uv run python main.py train --annotations data/annotations.parquet --output-dir models

# Train hierarchical classifier (n-gram tokenizer, 5-level)
uv run python main.py train-hierarchical --data data/data-train.parquet --output checkpoints/hierarchical

# Fine-tune a pre-trained hierarchical model
uv run python main.py fine-tune-hierarchical --model checkpoints/hierarchical/hierarchical_model --data data/new.parquet --output checkpoints/fine-tuned

# Predict with cascade model
uv run python main.py predict "pain complet bio" "bouteille eau minerale"

# Predict with hierarchical model (supports top-k)
uv run python main.py predict-hierarchical --model checkpoints/hierarchical/hierarchical_model "pain complet bio"

# Evaluate on test set
uv run python main.py evaluate --model-path models/model --test-file data/test.parquet

# Start FastAPI server
uv run python main.py serve --model checkpoints/hierarchical/hierarchical_model
```

## Architecture

### Two Classification Approaches

**Cascade Classifier** (`src/cascade_classifier.py`): Two-stage architecture using CamemBERT tokenizer. Level 1 predicts main category (01-13), then per-category sub-classifiers predict the full COICOP code. Simpler but limited to 2 levels.

**Hierarchical Classifier** (`src/hierarchical_classifier.py`): 5-level architecture using character n-gram tokenizer (3-6 grams). Each level N receives parent predictions from level N-1 as categorical features (parent code embedding + confidence bucket). Uses teacher forcing during training (default ratio 0.8).

### Key Data Flow

1. **Data loading** (`src/data_preparation.py`): Reads parquet, applies text preprocessing (unidecode → lowercase → noise/punctuation/digit removal → dedup tokens → remove stopwords from `data/text/stopwords.json`), extracts 5 hierarchical levels from dotted COICOP codes (e.g., "01.1.2.3.4" → level1="01", level2="01.1", etc.), filters out technical codes 98.x/99.x.

2. **Training** (`src/train.py`): Orchestrates training for either classifier type, handles MLflow experiment tracking, runs optional post-training top-k evaluation.

3. **Prediction** (`src/predict.py`): `COICOPPredictor` for cascade, `HierarchicalCOICOPPredictor` for hierarchical. Both support single text, batch, DataFrame, and file-based prediction.

4. **API** (`src/api.py`): FastAPI server exposing `/predict`, `/predict/batch`, `/health`, `/model/info`. Model path set via `COICOP_MODEL_PATH` env var. Serves a frontend UI from `src/static/`.

### CLI Structure

`main.py` uses argparse with subcommands: `train`, `train-hierarchical`, `fine-tune-hierarchical`, `predict`, `predict-hierarchical`, `evaluate`, `serve`. Each subcommand has a `cmd_*` handler that imports the relevant module lazily.

## Key Conventions

- Python 3.13 required (`.python-version`)
- PyPI packages fetched through INSEE Nexus proxy (configured in `pyproject.toml` `[tool.uv]`)
- PyTorch is installed from a CPU-only index (`pytorch-cpu`)
- Training data format: parquet with columns `product` (text), `code` (COICOP code), `coicop` (description)
- COICOP codes are dot-separated hierarchical strings (e.g., "01.1.2.3.4")
- Text preprocessing always applies `preprocess_text()` before training/prediction
- Models are serialized with pickle (metadata/mappings) + torchtextclassifiers built-in save/load (weights)
- Some code comments and documentation are in French
