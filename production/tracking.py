"""Shared MLflow configuration for the production pipeline.

Uses a repo-local sqlite backend (the MLflow file store is deprecated as of
MLflow 3.x and does not support the model registry). Artifacts land in
``mlruns/`` at the repo root.

Inspect runs with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db

run from the repo root.
"""

from __future__ import annotations

from pathlib import Path

import mlflow

EXPERIMENT_NAME = "compute-forecasting-production"
MODEL_NAME = "compute-forecast-hybrid"
CHAMPION_ALIAS = "champion"


def configure_mlflow(repo_root: Path) -> None:
    """Point MLflow at the repo-local sqlite store and select the experiment."""
    db_path = repo_root / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")

    client = mlflow.MlflowClient()
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        client.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=(repo_root / "mlruns").as_uri(),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)
