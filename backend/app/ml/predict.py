import joblib
import pandas as pd
from typing import Dict, Any, Tuple
from backend.app.config import SVM_MODEL_PATH, DT_MODEL_PATH, ALL_FEATURES

# Cache models in memory once loaded
_CACHED_MODELS: Dict[str, Any] = {}


def get_model(model_name: str = "SVM"):
    """Retrieve requested model pipeline from memory cache or disk."""
    normalized = "SVM" if "svm" in model_name.lower() else "Decision Tree"
    
    if normalized in _CACHED_MODELS:
        return _CACHED_MODELS[normalized], normalized

    model_path = SVM_MODEL_PATH if normalized == "SVM" else DT_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} not found. Please train models first.")

    model = joblib.load(model_path)
    _CACHED_MODELS[normalized] = model
    return model, normalized


def reload_models():
    """Clear cached models to reload newly trained versions."""
    _CACHED_MODELS.clear()


def predict_single(features: Dict[str, Any], model_name: str = "SVM") -> Dict[str, Any]:
    """
    Execute prediction for single session feature payload.
    Uses predict_proba() to get genuine positive class probability.
    """
    model, selected_name = get_model(model_name)

    # Convert single dictionary to DataFrame matching expected pipeline format
    # Ensure Weekend is cast to int (0/1) as expected by pipeline
    row = {k: features.get(k) for k in ALL_FEATURES}
    if "Weekend" in row:
        row["Weekend"] = int(row["Weekend"])

    input_df = pd.DataFrame([row])

    # Predict probability and class
    probabilities = model.predict_proba(input_df)[0]
    positive_proba = float(probabilities[1])
    predicted_class = int(model.predict(input_df)[0])

    is_purchase = bool(predicted_class == 1)
    label = "Purchase" if is_purchase else "No Purchase"

    return {
        "prediction": is_purchase,
        "prediction_label": label,
        "probability": round(positive_proba, 4),
        "model": selected_name,
        "features_received": features,
    }
