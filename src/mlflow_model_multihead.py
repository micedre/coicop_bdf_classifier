"""MLflow 'models from code' definition for the multi-head classifier."""

import mlflow

from src.mlflow_utils import MultiHeadCOICOPPyfuncWrapper

mlflow.models.set_model(MultiHeadCOICOPPyfuncWrapper())
