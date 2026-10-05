from typing import Dict, Any
from backend.app.ml.evaluate import load_model_results


def get_confusion_matrices() -> Dict[str, Any]:
    """Retrieve test confusion matrices and classification reports for both models."""
    results = load_model_results()
    return {
        "SVM": {
            "confusion_matrix": results["models"]["SVM"]["confusion_matrix"],
            "classification_report": results["models"]["SVM"]["classification_report"],
            "accuracy": results["models"]["SVM"]["accuracy"],
            "f1": results["models"]["SVM"]["f1"],
        },
        "Decision Tree": {
            "confusion_matrix": results["models"]["Decision Tree"]["confusion_matrix"],
            "classification_report": results["models"]["Decision Tree"]["classification_report"],
            "accuracy": results["models"]["Decision Tree"]["accuracy"],
            "f1": results["models"]["Decision Tree"]["f1"],
        },
    }


def get_roc_data() -> Dict[str, Any]:
    """Retrieve test ROC curve coordinates and AUC scores for both models."""
    results = load_model_results()
    return {
        "SVM": results["models"]["SVM"]["roc_curve"],
        "Decision Tree": results["models"]["Decision Tree"]["roc_curve"],
    }


def get_precision_recall_data() -> Dict[str, Any]:
    """Retrieve test Precision-Recall curve coordinates and F1 scores for both models."""
    results = load_model_results()
    return {
        "SVM": results["models"]["SVM"]["precision_recall_curve"],
        "Decision Tree": results["models"]["Decision Tree"]["precision_recall_curve"],
        "baseline_rate": results["dataset_summary"]["positive_percentage"] / 100.0,
    }
