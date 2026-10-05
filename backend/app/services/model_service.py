from typing import Dict, Any, List
from backend.app.ml.evaluate import load_model_results
from backend.app.ml.train import run_full_training_and_evaluation
from backend.app.ml.predict import reload_models


def get_all_model_results() -> Dict[str, Any]:
    """Retrieve full results dictionary containing metrics, confusion matrices, and ROC."""
    return load_model_results()


def get_model_comparison() -> Dict[str, Any]:
    """Retrieve model comparison table, best model, and academic guidance."""
    results = load_model_results()
    return {
        "best_model_by_f1": results["best_model_by_f1"],
        "comparison_table": results["comparison_table"],
        "models": results["models"],
        "objective_guidance": results["objective_guidance"],
    }


def get_feature_importances() -> Dict[str, Any]:
    """Retrieve permutation importances for SVM and tree-based importances for Decision Tree."""
    results = load_model_results()
    return results["feature_importance"]


def retrain_models() -> Dict[str, Any]:
    """Trigger retraining of both SVM and Decision Tree models and reload cache."""
    results = run_full_training_and_evaluation()
    reload_models()
    return {
        "status": "success",
        "message": "Models successfully retrained, cross-validated, and re-evaluated.",
        "best_model_by_f1": results["best_model_by_f1"],
        "comparison_table": results["comparison_table"],
    }
