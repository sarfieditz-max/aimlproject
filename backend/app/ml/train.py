import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.inspection import permutation_importance

from backend.app.config import (
    SVM_MODEL_PATH,
    DT_MODEL_PATH,
    METRICS_JSON_PATH,
    ROOT_METRICS_JSON_PATH,
    RANDOM_STATE,
    CV_SPLITS,
    PRIMARY_SCORING,
    NUMERICAL_FEATURES,
)
from backend.app.ml.preprocessing import (
    load_dataset,
    prepare_features_and_target,
    split_data,
    get_preprocessor_for_svm,
    get_preprocessor_for_dt,
    get_feature_names_from_preprocessor,
)
from backend.app.utils.metrics import calculate_test_metrics, perform_kfold_cv
from backend.app.utils.plotting import (
    generate_all_eda_plots,
    plot_13_confusion_matrices,
    plot_14_roc_curves,
    plot_15_precision_recall_curves,
    plot_16_model_comparison_chart,
)


def train_and_tune_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    c_values=None,
    gamma_values=None,
    cv_splits=CV_SPLITS,
    random_state=RANDOM_STATE,
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, float]]:
    """
    Train and tune SVM with RBF kernel and class_weight='balanced'
    using Stratified 5-Fold Cross-Validation on the training set.
    """
    if c_values is None:
        c_values = [0.1, 1.0, 10.0, 100.0]
    if gamma_values is None:
        gamma_values = ["scale", 0.01, 0.001]

    preprocessor = get_preprocessor_for_svm()
    base_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)),
    ])

    param_grid = {
        "classifier__C": c_values,
        "classifier__gamma": gamma_values,
    }

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring=PRIMARY_SCORING,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    print("Fitting SVM GridSearch...")
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params = {k.replace("classifier__", ""): v for k, v in grid_search.best_params_.items()}
    best_params["kernel"] = "rbf"
    print(f"Best SVM Parameters: {best_params}, Best CV F1: {grid_search.best_score_:.4f}")

    # Comprehensive CV metrics with best pipeline
    cv_metrics = perform_kfold_cv(best_pipeline, X_train, y_train, n_splits=cv_splits, random_state=random_state)

    return best_pipeline, best_params, cv_metrics


def train_and_tune_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits=CV_SPLITS,
    random_state=RANDOM_STATE,
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, float]]:
    """
    Train and tune Decision Tree Classifier with class_weight='balanced'
    using Stratified 5-Fold Cross-Validation on the training set.
    """
    preprocessor = get_preprocessor_for_dt()
    base_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(class_weight="balanced", random_state=random_state)),
    ])

    param_grid = {
        "classifier__max_depth": [3, 5, 7, 10, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__criterion": ["gini", "entropy"],
    }

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring=PRIMARY_SCORING,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    print("Fitting Decision Tree GridSearch...")
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params = {k.replace("classifier__", ""): v for k, v in grid_search.best_params_.items()}
    print(f"Best Decision Tree Parameters: {best_params}, Best CV F1: {grid_search.best_score_:.4f}")

    # Comprehensive CV metrics with best pipeline
    cv_metrics = perform_kfold_cv(best_pipeline, X_train, y_train, n_splits=cv_splits, random_state=random_state)

    return best_pipeline, best_params, cv_metrics


def calculate_feature_importances(
    svm_pipeline: Pipeline,
    dt_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Calculate feature importances:
    - Permutation Importance for SVM (evaluated on test set)
    - Native feature_importances_ for Decision Tree
    """
    # Extract feature names out
    fitted_preprocessor = dt_pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names_from_preprocessor(fitted_preprocessor)

    # 1. Decision Tree native importances
    dt_classifier = dt_pipeline.named_steps["classifier"]
    dt_importances = dt_classifier.feature_importances_
    dt_feat_imp = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in zip(feature_names, dt_importances)
    ]
    dt_feat_imp = sorted(dt_feat_imp, key=lambda x: x["importance"], reverse=True)

    # 2. SVM Permutation Importance
    print("Calculating Permutation Importance for SVM (n_repeats=5)...")
    perm_result = permutation_importance(
        svm_pipeline,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    
    # Original raw features importance for user understanding
    raw_feature_names = list(X_test.columns)
    svm_feat_imp = [
        {
            "feature": name,
            "importance": round(float(mean_imp), 4),
            "std": round(float(std_imp), 4),
        }
        for name, mean_imp, std_imp in zip(
            raw_feature_names, perm_result.importances_mean, perm_result.importances_std
        )
    ]
    svm_feat_imp = sorted(svm_feat_imp, key=lambda x: x["importance"], reverse=True)

    return {
        "svm_permutation_importance": svm_feat_imp,
        "decision_tree_feature_importance": dt_feat_imp,
        "derived_feature_names": feature_names,
        "note": "For SVM, permutation feature importance measures decrease in test F1 when each feature column is randomly shuffled. For Decision Tree, Gini/Entropy impurity reduction is utilized.",
    }


def run_full_training_and_evaluation() -> Dict[str, Any]:
    """
    Orchestrate complete pipeline:
    1. Load raw dataset (12,330 rows, 18 columns)
    2. Generate all 12 EDA plots
    3. Stratified 80/20 train/test split
    4. Train & Tune SVM
    5. Train & Tune Decision Tree
    6. Evaluate on untouched test set
    7. Generate evaluation plots (Confusion Matrices, ROC, PR, Comparison)
    8. Compute feature importances
    9. Save models and metrics
    """
    print("Step 1: Loading dataset...")
    df = load_dataset()
    print(f"Dataset shape: {df.shape}")

    print("Step 2: Generating EDA plots...")
    generate_all_eda_plots(df, NUMERICAL_FEATURES)

    print("Step 3: Preparing features and stratified splitting...")
    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Train positive rate: {y_train.mean():.4f}, Test positive rate: {y_test.mean():.4f}")

    print("Step 4: Training & tuning SVM...")
    svm_pipe, svm_params, svm_cv = train_and_tune_svm(X_train, y_train)

    print("Step 5: Training & tuning Decision Tree...")
    dt_pipe, dt_params, dt_cv = train_and_tune_decision_tree(X_train, y_train)

    print("Step 6: Evaluating models on untouched test set...")
    svm_test_metrics = calculate_test_metrics(svm_pipe, X_test, y_test)
    dt_test_metrics = calculate_test_metrics(dt_pipe, X_test, y_test)

    # Combine test & CV metrics
    svm_full = {**svm_test_metrics, **svm_cv, "best_params": svm_params}
    dt_full = {**dt_test_metrics, **dt_cv, "best_params": dt_params}

    print("Step 7: Generating Evaluation plots...")
    plot_13_confusion_matrices(svm_full["confusion_matrix"], dt_full["confusion_matrix"])
    plot_14_roc_curves(svm_full["roc_curve"], dt_full["roc_curve"])
    plot_15_precision_recall_curves(svm_full["precision_recall_curve"], dt_full["precision_recall_curve"], baseline_rate=float(y.mean()))

    # Build comparison table
    comparison_table = [
        {
            "Model": "SVM",
            "Accuracy": svm_full["accuracy"],
            "Precision": svm_full["precision"],
            "Recall": svm_full["recall"],
            "F1 Score": svm_full["f1"],
            "ROC-AUC": svm_full["roc_auc"],
            "CV Mean F1": svm_full["cv_mean_f1"],
            "CV Std F1": svm_full["cv_std_f1"],
        },
        {
            "Model": "Decision Tree",
            "Accuracy": dt_full["accuracy"],
            "Precision": dt_full["precision"],
            "Recall": dt_full["recall"],
            "F1 Score": dt_full["f1"],
            "ROC-AUC": dt_full["roc_auc"],
            "CV Mean F1": dt_full["cv_mean_f1"],
            "CV Std F1": dt_full["cv_std_f1"],
        },
    ]

    plot_16_model_comparison_chart(comparison_table)

    print("Step 8: Computing feature importances...")
    feature_importance_data = calculate_feature_importances(svm_pipe, dt_pipe, X_test, y_test)

    # Determine best model dynamically based on test F1
    best_model_name = "SVM" if svm_full["f1"] >= dt_full["f1"] else "Decision Tree"

    final_results = {
        "dataset_summary": {
            "total_rows": len(df),
            "total_columns": df.shape[1],
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "positive_cases": int(y.sum()),
            "negative_cases": int((1 - y).sum()),
            "positive_percentage": round(float(y.mean() * 100), 2),
        },
        "best_model_by_f1": best_model_name,
        "models": {
            "SVM": svm_full,
            "Decision Tree": dt_full,
        },
        "comparison_table": comparison_table,
        "feature_importance": feature_importance_data,
        "objective_guidance": "While F1-score balances precision and recall on the imbalanced minority class (Revenue=True), the optimal model choice depends on business trade-offs: prioritize Precision if false purchase interventions are costly, or Recall if capturing all prospective buyers is paramount.",
    }

    print("Step 9: Saving models and metric artifacts...")
    joblib.dump(svm_pipe, SVM_MODEL_PATH)
    joblib.dump(dt_pipe, DT_MODEL_PATH)

    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(final_results, f, indent=2)

    with open(ROOT_METRICS_JSON_PATH, "w") as f:
        json.dump(final_results, f, indent=2)

    print("Model training and evaluation successfully completed!")
    return final_results


if __name__ == "__main__":
    run_full_training_and_evaluation()
