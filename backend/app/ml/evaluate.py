import json
import joblib
from pathlib import Path
from typing import Dict, Any

from backend.app.config import (
    SVM_MODEL_PATH,
    DT_MODEL_PATH,
    METRICS_JSON_PATH,
    ROOT_METRICS_JSON_PATH,
)
from backend.app.ml.preprocessing import (
    load_dataset,
    prepare_features_and_target,
    split_data,
)
from backend.app.utils.metrics import calculate_test_metrics


def load_model_results() -> Dict[str, Any]:
    """Load cached model evaluation results from JSON artifact."""
    if METRICS_JSON_PATH.exists():
        with open(METRICS_JSON_PATH, "r") as f:
            return json.load(f)
    elif ROOT_METRICS_JSON_PATH.exists():
        with open(ROOT_METRICS_JSON_PATH, "r") as f:
            return json.load(f)
    raise FileNotFoundError("Model results not found. Please train models first.")


def evaluate_saved_models_on_test() -> Dict[str, Any]:
    """Load saved pipelines and evaluate on fresh test set split."""
    if not SVM_MODEL_PATH.exists() or not DT_MODEL_PATH.exists():
        raise FileNotFoundError("Trained models not found. Please run training pipeline first.")

    svm_model = joblib.load(SVM_MODEL_PATH)
    dt_model = joblib.load(DT_MODEL_PATH)

    df = load_dataset()
    X, y = prepare_features_and_target(df)
    _, X_test, _, y_test = split_data(X, y)

    svm_eval = calculate_test_metrics(svm_model, X_test, y_test)
    dt_eval = calculate_test_metrics(dt_model, X_test, y_test)

    return {
        "SVM": svm_eval,
        "Decision Tree": dt_eval,
    }
