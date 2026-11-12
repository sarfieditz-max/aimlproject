from fastapi import APIRouter
from backend.app.services.evaluation_service import (
    get_confusion_matrices,
    get_roc_data,
    get_precision_recall_data,
)

router = APIRouter(prefix="/api/evaluation", tags=["Evaluation"])


@router.get("/confusion-matrix")
def api_evaluation_confusion_matrix():
    """Retrieve test confusion matrices and classification reports for SVM and Decision Tree."""
    return get_confusion_matrices()


@router.get("/roc")
def api_evaluation_roc():
    """Retrieve ROC coordinates (FPR, TPR) and AUC scores for both models."""
    return get_roc_data()


@router.get("/precision-recall")
def api_evaluation_precision_recall():
    """Retrieve Precision-Recall curve coordinates, test F1, and minority class baseline."""
    return get_precision_recall_data()
