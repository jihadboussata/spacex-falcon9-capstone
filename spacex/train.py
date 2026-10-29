"""Rebuild the portfolio results: python -m spacex.train."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    ConfusionMatrixDisplay, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["PayloadMass", "Flights", "Block", "GridFins", "Reused", "Legs", "Orbit", "LaunchSite"]
LEGACY_PAYLOAD_FILL = 6123.547647058824


def prepare(frame):
    frame = frame.copy().sort_values(["Date", "FlightNumber"]).reset_index(drop=True)
    # The upstream course CSV replaced five missing payloads with a global mean.
    # Restore this documented sentinel so that each training fold imputes its own median.
    frame.loc[np.isclose(frame.PayloadMass, LEGACY_PAYLOAD_FILL, rtol=0, atol=1e-7), "PayloadMass"] = np.nan
    return frame


def pipeline(estimator):
    numeric = ["PayloadMass", "Flights", "Block"]
    categorical = ["GridFins", "Reused", "Legs", "Orbit", "LaunchSite"]
    transformer = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])
    return Pipeline([("preprocess", transformer), ("model", estimator)])


def threshold_from_validation(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[np.argmax(f1)])


def evaluate(y, p, threshold):
    predicted = p >= threshold
    ci = binomtest(int((y == predicted).sum()), len(y)).proportion_ci(confidence_level=.95)
    return {"accuracy": float(accuracy_score(y, predicted)),
        "accuracy_95pct_exact_interval": [ci.low, ci.high],
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "average_precision": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted)), "f1": float(f1_score(y, predicted)), "threshold": threshold}


def run(data, output):
    frame = prepare(pd.read_csv(data, parse_dates=["Date"]))
    X, y = frame[FEATURES], frame.Class.astype(int)
    if len(frame) < 60 or set(y) != {0, 1}:
        raise ValueError("Expected the documented binary launch snapshot with at least 60 rows.")
    a, b = int(.6 * len(frame)), int(.8 * len(frame))
    # The bundled snapshot has one launch per date. Reject ambiguous date boundaries.
    if not frame.Date.iloc[a-1] < frame.Date.iloc[a] or not frame.Date.iloc[b-1] < frame.Date.iloc[b]:
        raise ValueError("A split boundary crosses a date; adjust the date partition.")
    train, valid, test = np.arange(a), np.arange(a, b), np.arange(b, len(frame))
    models = {
        "logistic_regression": (LogisticRegression(max_iter=2000, random_state=42), {"model__C": [.1, 1, 10]}),
        "svm": (SVC(probability=True, random_state=42), {"model__C": [.1, 1, 10], "model__kernel": ["linear", "rbf"]}),
        "decision_tree": (DecisionTreeClassifier(random_state=42), {"model__max_depth": [2, 3], "model__min_samples_leaf": [2, 4]}),
        "knn": (KNeighborsClassifier(), {"model__n_neighbors": [3, 5, 7]}),
    }
    fitted, selection = {}, []
    for name, (model, grid) in models.items():
        search = GridSearchCV(pipeline(model), grid, cv=TimeSeriesSplit(n_splits=3, gap=1),
            scoring="average_precision", error_score="raise", n_jobs=1)
        search.fit(X.iloc[train], y.iloc[train])
        fitted[name] = search.best_estimator_
        selection.append({"model": name, "cv_average_precision": search.best_score_,
            "validation_average_precision": average_precision_score(y.iloc[valid], search.predict_proba(X.iloc[valid])[:, 1]),
            "best_parameters": search.best_params_})
    selection.sort(key=lambda s: s["validation_average_precision"], reverse=True)
    selected = selection[0]["model"]
    estimator = fitted[selected]
    threshold = threshold_from_validation(y.iloc[valid], estimator.predict_proba(X.iloc[valid])[:, 1])
    p = estimator.predict_proba(X.iloc[test])[:, 1]
    baseline = DummyClassifier(strategy="prior").fit(X.iloc[train], y.iloc[train])
    baseline_p = baseline.predict_proba(X.iloc[test])[:, 1]
    summary = {"selected_model": selected, "selection": selection, "features": FEATURES,
        "restored_missing_payloads": int(frame.PayloadMass.isna().sum()),
        "test": evaluate(y.iloc[test], p, threshold), "baseline_test": evaluate(y.iloc[test], baseline_p, .5),
        "partitions": {name: {"rows": len(i), "start": str(frame.Date.iloc[i].min().date()),
            "end": str(frame.Date.iloc[i].max().date()), "successes": int(y.iloc[i].sum())}
            for name, i in zip(("train", "validation", "test"), (train, valid, test))},
        "data_sha256": hashlib.sha256(Path(data).read_bytes()).hexdigest(), "seed": 42,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "protocol": "First 60% train, next 20% validation, final 20% test; expanding CV on train only; validation selects model and F1 threshold; model is not refitted on validation."}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(selection).to_csv(output / "model_selection.csv", index=False)
    predictions = frame.loc[test, ["Date", "FlightNumber", "Class"]].copy()
    predictions["probability"] = p
    predictions["prediction"] = (p >= threshold).astype(int)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    names = ["Selected model", "Training-prior baseline"]
    values = [summary["test"]["balanced_accuracy"], summary["baseline_test"]["balanced_accuracy"]]
    axes[0].bar(names, values, color=["#007f86", "#b4bdca"])
    axes[0].set(ylim=(0, 1), ylabel="Balanced accuracy", title="Chronological test (18 launches)")
    for j, value in enumerate(values): axes[0].text(j, value + .02, f"{value:.3f}", ha="center")
    ConfusionMatrixDisplay.from_predictions(y.iloc[test], p >= threshold, display_labels=["Unsuccessful / no landing", "Successful"],
        ax=axes[1], cmap="Blues", colorbar=False)
    axes[1].set_title(f"{selected}; threshold {threshold:.3f}")
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(output / "evaluation.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"selected_model": selected, "test": summary["test"], "baseline": summary["baseline_test"]}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dataset_part_2.csv")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    run(args.data, args.output)
