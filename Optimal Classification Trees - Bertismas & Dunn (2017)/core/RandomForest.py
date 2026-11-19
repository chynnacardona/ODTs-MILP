from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from uci_loader import load_all_uci_datasets

__all__ = [
    "fit_predict_rf_split",
    "evaluate_rf_over_splits",
    "evaluate_rf_datasets",
]


def fit_predict_rf_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    n_trees: int = 100,
    random_state: int = 42,
    n_jobs: int = -1,
    **rf_kwargs: Any,
) -> np.ndarray:
    """
    Trains a Random Forest on combined (train + validation) arrays and returns
    predicted class labels on X_test (Section 6.4 setup)[cite: 11].
    """
    X_combined = np.vstack([np.asarray(X_train), np.asarray(X_val)])
    y_combined = np.concatenate([np.asarray(y_train), np.asarray(y_val)])

    clf = RandomForestClassifier(
        n_estimators=n_trees,
        random_state=random_state,
        n_jobs=n_jobs,
        **rf_kwargs,
    )
    clf.fit(X_combined, y_combined)
    return clf.predict(np.asarray(X_test))


def evaluate_rf_over_splits(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str | None = None,
    n_splits: int = 5,
    n_trees: int = 100,
    test_size: float = 0.2,
    val_size_of_remaining: float = 0.25,
    random_state: int = 42,
    return_per_split: bool = False,
    **rf_kwargs: Any,
) -> dict[str, Any]:
    """
    Evaluates a Random Forest across stratified train/val/test splits and returns
    summary accuracy and Section 6.4 regime metadata[cite: 11].
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    n_samples, n_features = X_arr.shape
    n_classes = int(len(np.unique(y_arr)))

    sss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state
    )
    split_accuracies: list[float] = []

    for split_idx, (train_val_idx, test_idx) in enumerate(sss.split(X_arr, y_arr)):
        X_train_val, y_train_val = X_arr[train_val_idx], y_arr[train_val_idx]
        X_test, y_test = X_arr[test_idx], y_arr[test_idx]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_size_of_remaining,
            stratify=y_train_val,
            random_state=random_state + split_idx,
        )

        preds = fit_predict_rf_split(
            X_tr,
            y_tr,
            X_val,
            y_val,
            X_test,
            n_trees=n_trees,
            random_state=random_state + split_idx,
            **rf_kwargs,
        )
        split_accuracies.append(float(accuracy_score(y_test, preds)))

    mean_acc_pct = float(np.mean(split_accuracies)) * 100.0
    std_acc_pct = float(np.std(split_accuracies)) * 100.0

    result: dict[str, Any] = {
        "n": n_samples,
        "p": n_features,
        "K": n_classes,
        "n/p": round(n_samples / max(n_features, 1), 1),
        "rf_acc_mean": mean_acc_pct,
        "rf_acc_std": std_acc_pct,
        "RF_Acc_%": round(mean_acc_pct, 2),
        "Std_%": round(std_acc_pct, 2),
        "Regime_K23_p<=25": (n_classes in (2, 3)) and (n_features <= 25),
    }

    if dataset_name is not None:
        result = {"Dataset": dataset_name, **result}

    if return_per_split:
        result["split_accuracies"] = split_accuracies

    return result


def evaluate_rf_datasets(
    datasets: list[tuple[str, np.ndarray, np.ndarray]],
    n_splits: int = 5,
    n_trees: int = 100,
    random_state: int = 42,
    **rf_kwargs: Any,
) -> pd.DataFrame:
    """Evaluates Random Forest across a list of (name, X, y) tuples from uci_loader."""
    rows = [
        evaluate_rf_over_splits(
            X=X,
            y=y,
            dataset_name=name,
            n_splits=n_splits,
            n_trees=n_trees,
            random_state=random_state,
            **rf_kwargs,
        )
        for name, X, y in datasets
    ]
    return pd.DataFrame(rows)


def main() -> None:
    datasets = load_all_uci_datasets()
    df = evaluate_rf_datasets(datasets, n_splits=5, n_trees=100)
    cols = [
        "Dataset",
        "n",
        "p",
        "K",
        "n/p",
        "RF_Acc_%",
        "Std_%",
        "Regime_K23_p<=25",
    ]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()