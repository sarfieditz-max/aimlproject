from fastapi import APIRouter
from backend.app.schemas import ShopperFeatures, PredictionResponse
from backend.app.services.prediction_service import execute_prediction

router = APIRouter(prefix="/api", tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
def api_predict(payload: ShopperFeatures):
    """
    Accept visitor session features, validate through Pydantic,
    and predict purchase intention probability using the chosen model pipeline.
    """
    return execute_prediction(payload)
