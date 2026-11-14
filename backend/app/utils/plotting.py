import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from typing import Dict, Any, List

from backend.app.config import PLOTS_DIR, REPORTS_FIGURES_DIR

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True,
})

PALETTE_PRIMARY = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4"]
COLOR_REV_FALSE = "#3B82F6"  # Blue for No Purchase
COLOR_REV_TRUE = "#10B981"   # Emerald Green for Purchase


def save_figure(fig, filename: str):
    """Save figure to backend/artifacts/plots and sync to reports/figures."""
    out_path = PLOTS_DIR / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Sync to reports figures directory
    shutil.copy(out_path, REPORTS_FIGURES_DIR / filename)
    return str(out_path)


def plot_01_revenue_class_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df["Revenue"].value_counts()
    bars = ax.bar(["No Purchase (False)", "Purchase (True)"], counts, color=[COLOR_REV_FALSE, COLOR_REV_TRUE], width=0.5)
    ax.set_title("1. Revenue Class Distribution (Total Sessions = 12,330)", fontweight="bold", pad=12)
    ax.set_xlabel("Purchase Outcome (Revenue)")
    ax.set_ylabel("Number of Sessions")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 150, f"{h:,}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.15)
    return save_figure(fig, "01_revenue_class_distribution.png")


def plot_02_revenue_percentage(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df["Revenue"].value_counts()
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=["No Purchase (False)", "Purchase (True)"],
        autopct="%1.2f%%",
        startangle=140,
        colors=[COLOR_REV_FALSE, COLOR_REV_TRUE],
        explode=(0, 0.08),
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_color("white")
    ax.set_title("2. Revenue Class Imbalance Proportion", fontweight="bold", pad=12)
    return save_figure(fig, "02_revenue_percentage.png")


def plot_03_monthly_visitor_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    month_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ordered_months = [m for m in month_order if m in df["Month"].unique()]
    counts = df["Month"].value_counts().reindex(ordered_months)
    bars = ax.bar(counts.index, counts.values, color="#2563EB", width=0.6)
    ax.set_title("3. Monthly Session Distribution", fontweight="bold", pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Visitor Sessions")
    for bar in bars:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, v + 40, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(counts.values) * 1.12)
    return save_figure(fig, "03_monthly_visitor_distribution.png")


def plot_04_revenue_by_month(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    month_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    df_plot = df.copy()
    df_plot["Revenue_Label"] = df_plot["Revenue"].map({False: "No Purchase", True: "Purchase"})
    sns.countplot(data=df_plot, x="Month", hue="Revenue_Label", order=month_order, ax=ax, palette=[COLOR_REV_FALSE, COLOR_REV_TRUE])
    ax.set_title("4. Session Revenue Outcomes by Month", fontweight="bold", pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Sessions")
    ax.legend(title="Outcome", frameon=True)
    return save_figure(fig, "04_revenue_by_month.png")


def plot_05_visitortype_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["VisitorType"].value_counts()
    colors = ["#2563EB", "#10B981", "#F59E0B"]
    bars = ax.bar(counts.index, counts.values, color=colors[:len(counts)], width=0.5)
    ax.set_title("5. Visitor Type Distribution", fontweight="bold", pad=12)
    ax.set_xlabel("Visitor Type")
    ax.set_ylabel("Total Sessions")
    for bar in bars:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, v + 120, f"{v:,}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, max(counts.values) * 1.12)
    return save_figure(fig, "05_visitortype_distribution.png")


def plot_06_revenue_by_visitortype(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    df_plot = df.copy()
    df_plot["Revenue_Label"] = df_plot["Revenue"].map({False: "No Purchase", True: "Purchase"})
    sns.countplot(data=df_plot, x="VisitorType", hue="Revenue_Label", ax=ax, palette=[COLOR_REV_FALSE, COLOR_REV_TRUE])
    ax.set_title("6. Purchase Conversion by Visitor Type", fontweight="bold", pad=12)
    ax.set_xlabel("Visitor Type")
    ax.set_ylabel("Number of Sessions")
    ax.legend(title="Outcome", frameon=True)
    return save_figure(fig, "06_revenue_by_visitortype.png")


def plot_07_weekend_vs_revenue(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    df_plot = df.copy()
    df_plot["Weekend_Label"] = df_plot["Weekend"].map({False: "Weekday", True: "Weekend"})
    df_plot["Revenue_Label"] = df_plot["Revenue"].map({False: "No Purchase", True: "Purchase"})
    sns.countplot(data=df_plot, x="Weekend_Label", hue="Revenue_Label", ax=ax, palette=[COLOR_REV_FALSE, COLOR_REV_TRUE])
    ax.set_title("7. Weekend vs Weekday Sessions by Purchase Status", fontweight="bold", pad=12)
    ax.set_xlabel("Session Day Category")
    ax.set_ylabel("Number of Sessions")
    ax.legend(title="Outcome", frameon=True)
    return save_figure(fig, "07_weekend_vs_revenue.png")


def plot_08_pagevalues_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="PageValues", hue="Revenue", bins=40, kde=True, ax=ax, palette=[COLOR_REV_FALSE, COLOR_REV_TRUE], element="step")
    ax.set_title("8. PageValues Distribution (Key Predictor Feature)", fontweight="bold", pad=12)
    ax.set_xlabel("PageValues (Google Analytics Value)")
    ax.set_ylabel("Session Frequency (Log-scaled)")
    ax.set_yscale("log")
    ax.legend(labels=["Purchase (True)", "No Purchase (False)"], title="Revenue")
    return save_figure(fig, "08_pagevalues_distribution.png")


def plot_09_bouncerates_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="BounceRates", hue="Revenue", bins=40, kde=True, ax=ax, palette=[COLOR_REV_FALSE, COLOR_REV_TRUE], element="step")
    ax.set_title("9. BounceRates Distribution by Revenue Status", fontweight="bold", pad=12)
    ax.set_xlabel("BounceRates")
    ax.set_ylabel("Session Frequency (Log-scaled)")
    ax.set_yscale("log")
    ax.legend(labels=["Purchase (True)", "No Purchase (False)"], title="Revenue")
    return save_figure(fig, "09_bouncerates_distribution.png")


def plot_10_exitrates_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="ExitRates", hue="Revenue", bins=40, kde=True, ax=ax, palette=[COLOR_REV_FALSE, COLOR_REV_TRUE], element="step")
    ax.set_title("10. ExitRates Distribution by Revenue Status", fontweight="bold", pad=12)
    ax.set_xlabel("ExitRates")
    ax.set_ylabel("Session Frequency (Log-scaled)")
    ax.set_yscale("log")
    ax.legend(labels=["Purchase (True)", "No Purchase (False)"], title="Revenue")
    return save_figure(fig, "10_exitrates_distribution.png")


def plot_11_productrelated_duration_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    # Cap visualization at 99th percentile for readability with footnote
    p99 = df["ProductRelated_Duration"].quantile(0.99)
    df_filtered = df[df["ProductRelated_Duration"] <= p99]
    sns.histplot(data=df_filtered, x="ProductRelated_Duration", hue="Revenue", bins=40, kde=True, ax=ax, palette=[COLOR_REV_FALSE, COLOR_REV_TRUE])
    ax.set_title("11. ProductRelated_Duration Distribution (Truncated to 99th Percentile)", fontweight="bold", pad=12)
    ax.set_xlabel("ProductRelated_Duration (Seconds)")
    ax.set_ylabel("Session Count")
    ax.legend(labels=["Purchase (True)", "No Purchase (False)"], title="Revenue")
    return save_figure(fig, "11_productrelated_duration_distribution.png")


def plot_12_correlation_heatmap(df: pd.DataFrame, num_cols: List[str]):
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f", ax=ax, cbar_kws={"label": "Pearson Correlation Coefficient"})
    ax.set_title("12. Pearson Correlation Heatmap for Numerical Predictors", fontweight="bold", pad=12)
    return save_figure(fig, "12_correlation_heatmap.png")


def plot_13_confusion_matrices(svm_cm: List[List[int]], dt_cm: List[List[int]]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # SVM Confusion Matrix
    sns.heatmap(svm_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False,
                xticklabels=["Pred: False", "Pred: True"], yticklabels=["Actual: False", "Actual: True"])
    axes[0].set_title("SVM Confusion Matrix (Test Set, N=2,466)", fontweight="bold")
    axes[0].set_ylabel("Actual Class")
    axes[0].set_xlabel("Predicted Class")

    # DT Confusion Matrix
    sns.heatmap(dt_cm, annot=True, fmt="d", cmap="Greens", ax=axes[1], cbar=False,
                xticklabels=["Pred: False", "Pred: True"], yticklabels=["Actual: False", "Actual: True"])
    axes[1].set_title("Decision Tree Confusion Matrix (Test Set, N=2,466)", fontweight="bold")
    axes[1].set_ylabel("Actual Class")
    axes[1].set_xlabel("Predicted Class")

    plt.suptitle("13. Model Confusion Matrices Comparison", fontweight="bold", y=1.02)
    return save_figure(fig, "13_confusion_matrices.png")


def plot_14_roc_curves(svm_roc: Dict[str, Any], dt_roc: Dict[str, Any]):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(svm_roc["fpr"], svm_roc["tpr"], label=f"SVM (AUC = {svm_roc['auc']:.4f})", color="#2563EB", lw=2)
    ax.plot(dt_roc["fpr"], dt_roc["tpr"], label=f"Decision Tree (AUC = {dt_roc['auc']:.4f})", color="#10B981", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.5000)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.set_title("14. Receiver Operating Characteristic (ROC) Curves", fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    return save_figure(fig, "14_roc_curves.png")


def plot_15_precision_recall_curves(svm_pr: Dict[str, Any], dt_pr: Dict[str, Any], baseline_rate: float = 0.1547):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(svm_pr["recall"], svm_pr["precision"], label=f"SVM (F1 = {svm_pr['f1']:.4f})", color="#2563EB", lw=2)
    ax.plot(dt_pr["recall"], dt_pr["precision"], label=f"Decision Tree (F1 = {dt_pr['f1']:.4f})", color="#10B981", lw=2)
    ax.axhline(y=baseline_rate, color="gray", linestyle="--", lw=1.5, label=f"No-Skill Baseline ({baseline_rate:.2%})")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall (True Positive Rate)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("15. Precision-Recall Curves (Class-Imbalance Evaluation)", fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True)
    return save_figure(fig, "15_precision_recall_curves.png")


def plot_16_model_comparison_chart(comparison_data: List[Dict[str, Any]]):
    fig, ax = plt.subplots(figsize=(10, 5))
    df_comp = pd.DataFrame(comparison_data)
    
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "CV Mean F1"]
    svm_vals = [df_comp.loc[df_comp["Model"] == "SVM", m].values[0] for m in metrics]
    dt_vals = [df_comp.loc[df_comp["Model"] == "Decision Tree", m].values[0] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    rects1 = ax.bar(x - width / 2, svm_vals, width, label="SVM (Primary)", color="#2563EB")
    rects2 = ax.bar(x + width / 2, dt_vals, width, label="Decision Tree (Comparison)", color="#10B981")

    ax.set_ylabel("Score (0.0 - 1.0)")
    ax.set_title("16. Quantitative Model Performance Metric Comparison", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=True)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)

    return save_figure(fig, "16_model_comparison_chart.png")


def generate_all_eda_plots(df: pd.DataFrame, num_cols: List[str]):
    """Generate all 12 EDA plots from the dataset."""
    plot_01_revenue_class_distribution(df)
    plot_02_revenue_percentage(df)
    plot_03_monthly_visitor_distribution(df)
    plot_04_revenue_by_month(df)
    plot_05_visitortype_distribution(df)
    plot_06_revenue_by_visitortype(df)
    plot_07_weekend_vs_revenue(df)
    plot_08_pagevalues_distribution(df)
    plot_09_bouncerates_distribution(df)
    plot_10_exitrates_distribution(df)
    plot_11_productrelated_duration_distribution(df)
    plot_12_correlation_heatmap(df, num_cols)
