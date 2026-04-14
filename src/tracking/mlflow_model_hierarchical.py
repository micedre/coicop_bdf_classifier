"""MLflow 'models from code' definition for the hierarchical classifier."""

import mlflow

from src.mlflow_utils import HierarchicalCOICOPPyfuncWrapper

mlflow.models.set_model(HierarchicalCOICOPPyfuncWrapper())
