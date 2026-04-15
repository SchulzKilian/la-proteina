"""Linear, MLP, and kNN probe trainers with grouped cross-validation."""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ProbeTask(Enum):
    """Type of probe task. Currently only regression is used."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


def _build_probe(
    probe_type: str,
    ridge_alphas: list[float],
    mlp_hidden: int,
    knn_k: int,
    rng_seed: int,
) -> Any:
    """Instantiate a probe model.

    Parameters
    ----------
    probe_type : str
        "linear", "mlp", or "knn".
    ridge_alphas : list[float]
        Alpha grid for Ridge CV.
    mlp_hidden : int
        Hidden layer size for MLP.
    knn_k : int
        Number of neighbors for kNN.
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    sklearn estimator
    """
    if probe_type == "linear":
        return RidgeCV(alphas=ridge_alphas)
    elif probe_type == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(mlp_hidden,),
            early_stopping=True,
            validation_fraction=0.15,
            max_iter=500,
            random_state=rng_seed,
        )
    elif probe_type == "knn":
        return KNeighborsRegressor(n_neighbors=knn_k)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")


def run_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    probe_type: str,
    n_folds: int = 5,
    ridge_alphas: list[float] | None = None,
    mlp_hidden: int = 64,
    knn_k: int = 10,
    rng_seed: int = 42,
) -> dict[str, float]:
    """Train and evaluate a probe with grouped k-fold cross-validation.

    Folds are grouped by protein_id (via the ``groups`` array) so the
    same protein never appears in both train and test.

    Parameters
    ----------
    X : np.ndarray
        Features, shape ``[N, D]``.
    y : np.ndarray
        Targets, shape ``[N]``.
    groups : np.ndarray
        Group labels, shape ``[N]``. Same protein -> same group.
    probe_type : str
        "linear", "mlp", or "knn".
    n_folds : int
        Number of CV folds.
    ridge_alphas : list[float]
        Alpha grid for Ridge (default: reasonable range).
    mlp_hidden : int
        MLP hidden layer size.
    knn_k : int
        kNN neighbors.
    rng_seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: r2_mean, r2_std, mae_mean, mae_std, n_train, n_test,
        fold_r2s (list of per-fold R²).
    """
    if ridge_alphas is None:
        ridge_alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    n_unique_groups = len(np.unique(groups))
    actual_folds = min(n_folds, n_unique_groups)
    if actual_folds < n_folds:
        logger.warning("Only %d unique groups, reducing folds from %d to %d",
                       n_unique_groups, n_folds, actual_folds)
    if actual_folds < 2:
        logger.warning("Fewer than 2 groups — cannot cross-validate. Returning NaN.")
        return {
            "r2_mean": float("nan"),
            "r2_std": float("nan"),
            "mae_mean": float("nan"),
            "mae_std": float("nan"),
            "n_train": len(X),
            "n_test": 0,
            "fold_r2s": [],
        }

    gkf = GroupKFold(n_splits=actual_folds)
    fold_r2s = []
    fold_maes = []
    n_trains = []
    n_tests = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = _build_probe(probe_type, ridge_alphas, mlp_hidden, knn_k, rng_seed)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        fold_r2s.append(r2_score(y_test, y_pred))
        fold_maes.append(mean_absolute_error(y_test, y_pred))
        n_trains.append(len(train_idx))
        n_tests.append(len(test_idx))

    return {
        "r2_mean": float(np.mean(fold_r2s)),
        "r2_std": float(np.std(fold_r2s)),
        "mae_mean": float(np.mean(fold_maes)),
        "mae_std": float(np.std(fold_maes)),
        "n_train": int(np.mean(n_trains)),
        "n_test": int(np.mean(n_tests)),
        "fold_r2s": [float(r) for r in fold_r2s],
    }


def run_all_probes(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    probe_config: dict,
    rng_seed: int = 42,
) -> list[dict]:
    """Run all three probe types and return results.

    Parameters
    ----------
    X : np.ndarray
        Features.
    y : np.ndarray
        Targets.
    groups : np.ndarray
        Group labels.
    probe_config : dict
        Must contain: cv_folds, ridge_alphas, mlp_hidden, knn_k.
    rng_seed : int
        Random seed.

    Returns
    -------
    list[dict]
        One dict per probe type with probe_type key added.
    """
    results = []
    for pt in ["linear", "mlp", "knn"]:
        logger.info("  Running %s probe (N=%d, D=%d)", pt, X.shape[0], X.shape[1])
        res = run_probe(
            X, y, groups,
            probe_type=pt,
            n_folds=probe_config["cv_folds"],
            ridge_alphas=probe_config["ridge_alphas"],
            mlp_hidden=probe_config["mlp_hidden"],
            knn_k=probe_config["knn_k"],
            rng_seed=rng_seed,
        )
        res["probe_type"] = pt
        results.append(res)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(42)

    # Smoke test: linear relationship
    N = 200
    D = 8
    X = rng.standard_normal((N, D)).astype(np.float32)
    true_w = rng.standard_normal(D)
    y = (X @ true_w + rng.standard_normal(N) * 0.5).astype(np.float32)
    groups = np.repeat(np.arange(20), 10)

    for pt in ["linear", "mlp", "knn"]:
        res = run_probe(X, y, groups, probe_type=pt, n_folds=5)
        print(f"{pt}: R²={res['r2_mean']:.3f}±{res['r2_std']:.3f}, "
              f"MAE={res['mae_mean']:.3f}")
