import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from backend.app.config import (
    DATASET_PATH,
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    BOOLEAN_FEATURES,
    ALL_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
)


def load_dataset(filepath=DATASET_PATH) -> pd.DataFrame:
    """Load the online shoppers intention dataset."""
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    if df.shape != (12330, 18):
        raise ValueError(f"Expected shape (12330, 18), but got {df.shape}")
    return df


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and target, casting Revenue and Weekend to numeric integers."""
    df_copy = df.copy()
    
    # Cast target variable: False -> 0, True -> 1
    if TARGET_COLUMN in df_copy.columns:
        y = df_copy[TARGET_COLUMN].astype(int)
        X = df_copy.drop(columns=[TARGET_COLUMN])
    else:
        y = None
        X = df_copy

    # Cast Weekend boolean column to int: False -> 0, True -> 1
    if "Weekend" in X.columns:
        X["Weekend"] = X["Weekend"].astype(int)

    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into 80% training and 20% testing sets using stratified splitting.
    The test set remains completely unseen during training and tuning.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def get_preprocessor_for_svm() -> ColumnTransformer:
    """
    Construct ColumnTransformer for SVM:
    - StandardScaler for 14 numerical features
    - OneHotEncoder(handle_unknown='ignore') for 2 categorical features
    - Passthrough for numeric Weekend (0/1)
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
        ],
        remainder="drop",
    )


def get_preprocessor_for_dt() -> ColumnTransformer:
    """
    Construct ColumnTransformer for Decision Tree:
    - Passthrough for 14 numerical features (Decision Trees are scale-invariant)
    - OneHotEncoder(handle_unknown='ignore') for 2 categorical features
    - Passthrough for numeric Weekend (0/1)
    """
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
        ],
        remainder="drop",
    )


def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> List[str]:
    """Extract output feature names from a fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback manual reconstruction if get_feature_names_out is unavailable
        names = list(NUMERICAL_FEATURES)
        cat_encoder = preprocessor.named_transformers_["cat"]
        names.extend(list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)))
        names.extend(list(BOOLEAN_FEATURES))
        return names
