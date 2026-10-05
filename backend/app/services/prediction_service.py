from typing import Dict, Any
from backend.app.schemas import ShopperFeatures, PredictionResponse
from backend.app.ml.predict import predict_single


def execute_prediction(features_input: ShopperFeatures) -> PredictionResponse:
    """Validate and execute prediction via pipeline with genuine probability."""
    features_dict = features_input.model_dump()
    model_name = features_dict.pop("model", "SVM") or "SVM"

    result = predict_single(features=features_dict, model_name=model_name)

    return PredictionResponse(
        prediction=result["prediction"],
        prediction_label=result["prediction_label"],
        probability=result["probability"],
        model=result["model"],
        features_received=result["features_received"],
    )
