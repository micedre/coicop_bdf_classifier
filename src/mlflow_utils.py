"""MLflow utilities for Lightning integration without premature run termination."""

from __future__ import annotations

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
