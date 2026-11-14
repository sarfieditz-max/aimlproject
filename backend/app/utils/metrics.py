import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate


def calculate_test_metrics(model, X_test, y_test) -> Dict[str, Any]:
    """
    Evaluate a fitted model pipeline on the untouched test set.
    All metrics are calculated directly without fabrication.
    """
    y_pred = model.predict(X_test)
    
    # Calculate probabilities for positive class (1)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_proba))
    
    cm = confusion_matrix(y_test, y_pred).tolist()
    cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # ROC curve points
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    # Precision-Recall curve points
    precision_pts, recall_pts, _ = precision_recall_curve(y_test, y_proba)

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm,
        "classification_report": cr,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": round(auc, 4),
        },
        "precision_recall_curve": {
            "precision": precision_pts.tolist(),
            "recall": recall_pts.tolist(),
            "f1": round(f1, 4),
        },
    }


def perform_kfold_cv(pipeline, X_train, y_train, n_splits=5, random_state=42) -> Dict[str, float]:
    """
    Perform 5-fold Stratified cross-validation on the training set.
    Returns mean and std of F1, accuracy, precision, recall, and ROC-AUC.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    
    scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    
    return {
        "cv_mean_f1": round(float(np.mean(scores["test_f1"])), 4),
        "cv_std_f1": round(float(np.std(scores["test_f1"])), 4),
        "cv_mean_accuracy": round(float(np.mean(scores["test_accuracy"])), 4),
        "cv_std_accuracy": round(float(np.std(scores["test_accuracy"])), 4),
        "cv_mean_precision": round(float(np.mean(scores["test_precision"])), 4),
        "cv_std_precision": round(float(np.std(scores["test_precision"])), 4),
        "cv_mean_recall": round(float(np.mean(scores["test_recall"])), 4),
        "cv_std_recall": round(float(np.std(scores["test_recall"])), 4),
        "cv_mean_roc_auc": round(float(np.mean(scores["test_roc_auc"])), 4),
        "cv_std_roc_auc": round(float(np.std(scores["test_roc_auc"])), 4),
    }
