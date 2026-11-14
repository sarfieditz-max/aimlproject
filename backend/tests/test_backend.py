import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.config import DATASET_PATH, SVM_MODEL_PATH, DT_MODEL_PATH, METRICS_JSON_PATH
from backend.app.ml.preprocessing import (
    load_dataset,
    prepare_features_and_target,
    split_data,
    get_preprocessor_for_svm,
    get_preprocessor_for_dt,
)
from backend.app.ml.predict import get_model, predict_single
from backend.app.ml.evaluate import load_model_results
from backend.app.utils.metrics import calculate_test_metrics

client = TestClient(app)

SAMPLE_VALID_INPUT = {
    "Administrative": 2,
    "Administrative_Duration": 54.0,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 15,
    "ProductRelated_Duration": 420.5,
    "BounceRates": 0.01,
    "ExitRates": 0.03,
    "PageValues": 18.5,
    "SpecialDay": 0.0,
    "Month": "Nov",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": False,
    "model": "SVM",
}


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["dataset_present"] is True
    assert data["svm_model_loaded"] is True
    assert data["dt_model_loaded"] is True
    assert data["metrics_present"] is True


def test_dataset_loading():
    df = load_dataset()
    assert df.shape == (12330, 18)
    assert df.isnull().sum().sum() == 0
    assert "Revenue" in df.columns
    assert set(df["Revenue"].unique()) == {False, True}


def test_preprocessing_and_splitting():
    df = load_dataset()
    X, y = prepare_features_and_target(df)
    assert len(X) == 12330
    assert len(y) == 12330
    assert "Revenue" not in X.columns

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42)
    assert len(X_train) == 9864
    assert len(X_test) == 2466
    assert abs(y_train.mean() - y_test.mean()) < 0.01

    svm_prep = get_preprocessor_for_svm()
    X_train_trans = svm_prep.fit_transform(X_train)
    assert X_train_trans.shape[0] == 9864
    assert not np.isnan(X_train_trans).any()


def test_model_loading():
    svm_model, name_svm = get_model("SVM")
    assert name_svm == "SVM"
    assert hasattr(svm_model, "predict_proba")

    dt_model, name_dt = get_model("Decision Tree")
    assert name_dt == "Decision Tree"
    assert hasattr(dt_model, "predict_proba")


def test_metrics_integrity():
    results = load_model_results()
    assert "models" in results
    assert "SVM" in results["models"]
    assert "Decision Tree" in results["models"]
    
    svm_metrics = results["models"]["SVM"]
    assert 0.0 <= svm_metrics["accuracy"] <= 1.0
    assert 0.0 <= svm_metrics["precision"] <= 1.0
    assert 0.0 <= svm_metrics["recall"] <= 1.0
    assert 0.0 <= svm_metrics["f1"] <= 1.0
    assert 0.0 <= svm_metrics["roc_auc"] <= 1.0
    assert len(svm_metrics["confusion_matrix"]) == 2


def test_dataset_summary_endpoint():
    response = client.get("/api/dataset/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_rows"] == 12330
    assert data["total_columns"] == 18
    assert data["training_samples"] == 9864
    assert data["testing_samples"] == 2466
    assert data["missing_values_count"] == 0


def test_models_results_and_comparison_endpoints():
    res_results = client.get("/api/models/results")
    assert res_results.status_code == 200
    assert "models" in res_results.json()

    res_comp = client.get("/api/models/comparison")
    assert res_comp.status_code == 200
    data = res_comp.json()
    assert "best_model_by_f1" in data
    assert len(data["comparison_table"]) == 2


def test_evaluation_endpoints():
    cm_res = client.get("/api/evaluation/confusion-matrix")
    assert cm_res.status_code == 200
    assert "SVM" in cm_res.json()
    assert "Decision Tree" in cm_res.json()

    roc_res = client.get("/api/evaluation/roc")
    assert roc_res.status_code == 200
    assert "fpr" in roc_res.json()["SVM"]

    pr_res = client.get("/api/evaluation/precision-recall")
    assert pr_res.status_code == 200
    assert "precision" in pr_res.json()["SVM"]


def test_features_importance_endpoint():
    response = client.get("/api/features/importance")
    assert response.status_code == 200
    data = response.json()
    assert "svm_permutation_importance" in data
    assert "decision_tree_feature_importance" in data
    assert len(data["svm_permutation_importance"]) > 0


def test_prediction_endpoint_valid():
    # Test SVM prediction
    res_svm = client.post("/api/predict", json=SAMPLE_VALID_INPUT)
    assert res_svm.status_code == 200
    data_svm = res_svm.json()
    assert "prediction" in data_svm
    assert "prediction_label" in data_svm
    assert "probability" in data_svm
    assert 0.0 <= data_svm["probability"] <= 1.0
    assert data_svm["model"] == "SVM"

    # Test Decision Tree prediction
    dt_input = dict(SAMPLE_VALID_INPUT)
    dt_input["model"] = "Decision Tree"
    res_dt = client.post("/api/predict", json=dt_input)
    assert res_dt.status_code == 200
    data_dt = res_dt.json()
    assert data_dt["model"] == "Decision Tree"
    assert 0.0 <= data_dt["probability"] <= 1.0


def test_prediction_endpoint_invalid_input():
    # Negative bounce rate (violates ge=0.0)
    bad_input = dict(SAMPLE_VALID_INPUT)
    bad_input["BounceRates"] = -0.5
    res = client.post("/api/predict", json=bad_input)
    assert res.status_code == 422  # Pydantic validation error

    # Missing mandatory field
    bad_input2 = dict(SAMPLE_VALID_INPUT)
    del bad_input2["PageValues"]
    res2 = client.post("/api/predict", json=bad_input2)
    assert res2.status_code == 422
