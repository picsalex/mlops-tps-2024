"""Helper functions for experiment tracking.

This module contains helper functions that allow experiment tracking to be
performed seamlessly across different experiment trackers and stacks.
"""

from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import (
    MLFlowExperimentTracker,
)

LOCAL_MLFLOW_UI_PORT = 8185


def get_tracker_name() -> str | None:
    """Get the name of the active experiment tracker."""

    experiment_tracker = Client().active_stack.experiment_tracker
    return experiment_tracker.name if experiment_tracker else None


def enable_autolog() -> None:
    """Automatically log to the active experiment tracker."""

    experiment_tracker = Client().active_stack.experiment_tracker
    if isinstance(experiment_tracker, MLFlowExperimentTracker):
        import mlflow

        mlflow.autolog()


def log_metric(key: str, value: float) -> None:
    """Log a metric to the active experiment tracker."""

    experiment_tracker = Client().active_stack.experiment_tracker
    if isinstance(experiment_tracker, MLFlowExperimentTracker):
        import mlflow

        mlflow.log_metric(key, value)


def log_artifact(local_path: str, artifact_path: str) -> None:
    """Log an artifact to the active experiment tracker."""

    experiment_tracker = Client().active_stack.experiment_tracker
    if isinstance(experiment_tracker, MLFlowExperimentTracker):
        import mlflow

        mlflow.log_artifact(local_path, artifact_path)


def log_text(text: str, filename: str) -> None:
    """Log a file to the active experiment tracker."""

    experiment_tracker = Client().active_stack.experiment_tracker
    if isinstance(experiment_tracker, MLFlowExperimentTracker):
        import mlflow

        mlflow.start_run()


def log_model(model, model_name: str) -> None:
    """Log a model to the active experiment tracker."""

    experiment_tracker = Client().active_stack.experiment_tracker
    if isinstance(experiment_tracker, MLFlowExperimentTracker):
        import mlflow

        mlflow.sklearn.log_model(model, model_name)


def get_current_tracker_run_id() -> str | None:
    """Get the URL of the current experiment tracker run."""

    experiment_tracker = Client().active_stack.experiment_tracker
    if isinstance(experiment_tracker, MLFlowExperimentTracker):
        import mlflow

        return mlflow.last_active_run().info.run_id

    return None
