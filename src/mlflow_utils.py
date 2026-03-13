"""MLflow utilities for Lightning integration without premature run termination."""

from __future__ import annotations

import mlflow.pyfunc
from pytorch_lightning.loggers import MLFlowLogger


class NonFinalizingMLFlowLogger(MLFlowLogger):
    """MLFlowLogger that does NOT terminate the MLflow run on finalize().

    When Lightning's Trainer finishes, it calls logger.finalize() which
    terminates the MLflow run. Since multiple trainers share one run
    (e.g. 5 levels in hierarchical training), we need to skip run
    termination and let the caller manage the run lifecycle.
    """

    def finalize(self, status: str = "success") -> None:
        if not self._initialized:
            return
        if self._checkpoint_callback:
            self._scan_and_log_checkpoints(self._checkpoint_callback)


def make_trainer_params(
    experiment_name: str,
    run_id: str,
    tracking_uri: str,
    prefix: str = "",
) -> dict:
    """Build trainer_params dict with a NonFinalizingMLFlowLogger.

    Args:
        experiment_name: MLflow experiment name.
        run_id: Active MLflow run ID to log into.
        tracking_uri: MLflow tracking URI.
        prefix: Metric prefix (e.g. "level1" -> metrics appear as "level1-train_loss").

    Returns:
        Dict suitable for TrainingConfig(trainer_params=...).
    """
    logger = NonFinalizingMLFlowLogger(
        experiment_name=experiment_name,
        run_id=run_id,
        tracking_uri=tracking_uri,
        prefix=prefix,
    )
    return {"logger": logger}


class COICOPPyfuncWrapper(mlflow.pyfunc.PythonModel):
    """End-to-end wrapper: raw text -> preprocess -> predict.

    Wraps a BasicCOICOPClassifier so that it can be served via
    ``mlflow.pyfunc.load_model`` with automatic text preprocessing.
    """

    def load_context(self, context):
        import json

        from .basic_classifier import BasicCOICOPClassifier
        from .data_preparation import preprocess_text

        self.classifier = BasicCOICOPClassifier.load(context.artifacts["model_dir"])

        with open(context.artifacts["stopwords"], "r", encoding="utf-8") as f:
            self.stopwords = json.load(f)

        self._preprocess_text = preprocess_text

    def predict(self, context, model_input, params=None):
        import pandas as pd

        top_k = 1
        if params and "top_k" in params:
            top_k = int(params["top_k"])

        df = model_input.copy()
        df = self._preprocess_text(df, "text", self.stopwords)

        texts = df["text"].tolist()
        result = self.classifier.predict(texts, top_k=top_k)

        if top_k > 1:
            return pd.DataFrame({
                "predicted_code": [codes[0] for codes in result["predictions"]],
                "confidence": [confs[0] for confs in result["confidence"]],
            })

        return pd.DataFrame({
            "predicted_code": result["predictions"],
            "confidence": result["confidence"],
        })


class HierarchicalCOICOPPyfuncWrapper(mlflow.pyfunc.PythonModel):
    """End-to-end wrapper: raw text -> preprocess -> hierarchical predict.

    Wraps a HierarchicalCOICOPClassifier so that it can be served via
    ``mlflow.pyfunc.load_model`` with automatic text preprocessing.
    """

    def load_context(self, context):
        import json

        from .hierarchical_classifier import HierarchicalCOICOPClassifier
        from .data_preparation import preprocess_text

        self.classifier = HierarchicalCOICOPClassifier.load(
            context.artifacts["model_dir"]
        )

        with open(context.artifacts["stopwords"], "r", encoding="utf-8") as f:
            self.stopwords = json.load(f)

        self._preprocess_text = preprocess_text

    def predict(self, context, model_input, params=None):
        import pandas as pd

        top_k = 1
        if params and "top_k" in params:
            top_k = int(params["top_k"])

        df = model_input.copy()
        df = self._preprocess_text(df, "text", self.stopwords)

        texts = df["text"].tolist()
        result = self.classifier.predict(
            texts, return_all_levels=True, top_k=top_k
        )

        # Build output DataFrame
        output = {
            "predicted_code": result["final_code"],
            "confidence": result["final_confidence"],
            "combined_confidence": result["combined_confidence"],
        }

        if "all_levels" in result:
            for level_name, level_data in result["all_levels"].items():
                if top_k > 1:
                    # level_data["predictions"][i] is list[str]
                    output[f"predicted_{level_name}"] = [
                        preds[0] for preds in level_data["predictions"]
                    ]
                    output[f"confidence_{level_name}"] = [
                        confs[0] for confs in level_data["confidence"]
                    ]
                    for k in range(1, top_k):
                        rank = k + 1
                        output[f"predicted_{level_name}_top{rank}"] = [
                            preds[k] if k < len(preds) else ""
                            for preds in level_data["predictions"]
                        ]
                        output[f"confidence_{level_name}_top{rank}"] = [
                            confs[k] if k < len(confs) else 0.0
                            for confs in level_data["confidence"]
                        ]
                else:
                    output[f"predicted_{level_name}"] = level_data["predictions"]
                    output[f"confidence_{level_name}"] = level_data["confidence"]

        return pd.DataFrame(output)
