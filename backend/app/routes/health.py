from fastapi import APIRouter
from backend.app.schemas import HealthResponse
from backend.app.config import DATASET_PATH, SVM_MODEL_PATH, DT_MODEL_PATH, METRICS_JSON_PATH

router = APIRouter(prefix="/api", tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_health_status():
    dataset_ok = DATASET_PATH.exists()
    svm_ok = SVM_MODEL_PATH.exists()
    dt_ok = DT_MODEL_PATH.exists()
    metrics_ok = METRICS_JSON_PATH.exists()

    all_ready = dataset_ok and svm_ok and dt_ok and metrics_ok

    return HealthResponse(
        status="healthy" if all_ready else "degraded",
        dataset_present=dataset_ok,
        svm_model_loaded=svm_ok,
        dt_model_loaded=dt_ok,
        metrics_present=metrics_ok,
        message="System fully operational with trained models and validated dataset."
        if all_ready
        else "Models or dataset need preparation.",
    )
