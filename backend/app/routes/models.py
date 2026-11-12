from fastapi import APIRouter, BackgroundTasks
from backend.app.services.model_service import (
    get_all_model_results,
    get_model_comparison,
    get_feature_importances,
    retrain_models,
)

router = APIRouter(prefix="/api", tags=["Models"])


@router.get("/models/results")
def api_models_results():
    """Retrieve full test evaluation metrics, hyperparameters, and cross-validation stats."""
    return get_all_model_results()


@router.get("/models/comparison")
def api_models_comparison():
    """Retrieve comparative metrics table, best model selector, and objective rationale."""
    return get_model_comparison()


@router.get("/features/importance")
def api_features_importance():
    """Retrieve Permutation Feature Importance for SVM and Gini Importance for Decision Tree."""
    return get_feature_importances()


@router.post("/retrain")
def api_retrain_models():
    """Trigger retraining of both SVM and Decision Tree pipelines."""
    return retrain_models()
