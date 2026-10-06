from pathlib import Path
import os
import tempfile
import warnings

matplotlib_cache_dir = Path(tempfile.gettempdir()) / "heart_disease_matplotlib_cache"
matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")

POSITIVE_CLASS = 1

#Chuyển tên mô hình hoặc tên metric thành tên file 
def make_filename_slug(text: str) -> str:

    return text.lower().replace(" ", "_").replace("-", "_")

#Tính xác suất hoặc decision score để tính ROC-AUC
def predict_scores(model, X):

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, POSITIVE_CLASS]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    return model.predict(X)

#Tính Accuracy, Precision, Recall, F1 và ROC-AUC cho từng mô hình
def evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_score = predict_scores(model, X_test)
    return {
        "Model": model_name,
        "Train Accuracy": accuracy_score(y_train, y_train_pred),
        "Test Accuracy": accuracy_score(y_test, y_test_pred),
        "Train Precision": precision_score(y_train, y_train_pred, zero_division=0),
        "Test Precision": precision_score(y_test, y_test_pred, zero_division=0),
        "Train Recall": recall_score(y_train, y_train_pred, zero_division=0),
        "Test Recall": recall_score(y_test, y_test_pred, zero_division=0),
        "Train F1": f1_score(y_train, y_train_pred, zero_division=0),
        "Test F1": f1_score(y_test, y_test_pred, zero_division=0),
        "Test ROC AUC": roc_auc_score(y_test, y_test_score),
    }



#Vẽ confusion matrix cho từng model
def save_confusion_matrices(trained_models, X_test, y_test, output_dir: Path) -> None:

    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test)
        matrix = confusion_matrix(y_test, y_pred)

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["No Disease", "Disease"],
        )
        display.plot(cmap="YlGnBu", values_format="d")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        plt.savefig(
            output_dir / "result model"/ f"confusion_matrix_{make_filename_slug(model_name)}.png",
            dpi=150,
        )
        plt.close()

#Tạo classification report của từng model
def save_classification_reports(trained_models, X_test, y_test, output_dir: Path) -> None:
    rows = []
    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test)
        report = classification_report(
            y_test,
            y_pred,
            target_names=["No Disease", "Disease"],
            output_dict=True,
            zero_division=0,
        )
        for label in ["No Disease", "Disease", "macro avg", "weighted avg"]:
            rows.append(
                {
                    "Model": model_name,
                    "Label": label,
                    "Precision": report[label]["precision"],
                    "Recall": report[label]["recall"],
                    "F1": report[label]["f1-score"],
                    "Support": report[label]["support"],
                }
            )

    pd.DataFrame(rows).to_csv(output_dir /  "result model"/"classification_reports.csv", index=False)

#Vẽ ROC curve của tất cả mô hình trên cùng một hình của 1 tập Test cụ thể
def save_roc_curves(trained_models, X_test, y_test, output_dir: Path) -> None:

    plt.figure(figsize=(8, 6))

    for model_name, model in trained_models.items():
        y_score = predict_scores(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc = roc_auc_score(y_test, y_score)
        plt.plot(fpr, tpr, label=f"{model_name} AUC={auc:.3f}")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "result model"/ "roc_curves.png", dpi=150)
    plt.close()

#Vẽ biểu đồ so sánh Accuracy, Precision, Recall, F1 và ROC-AUC của từng mô hình với nhau
def save_model_comparison_charts(results_df: pd.DataFrame, output_dir: Path) -> None:

    metrics = ["CV Accuracy Mean", "CV Precision Mean", "CV Recall Mean", "CV F1 Mean", "CV ROC AUC Mean"]

    for metric in metrics:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=results_df, x="Model", y=metric, palette="viridis", hue="Model")
        plt.title(f"So sanh {metric} ")
        plt.ylim(0, 1)
        plt.xticks(rotation=18)
        plt.legend([], [], frameon=False)
        plt.tight_layout()
        plt.savefig(output_dir / "result model"/ f"model_comparison_{make_filename_slug(metric)}.png", dpi=150)
        plt.close()

    # Tạo learning curve để kiểm tra mô hình có overfit hoặc underfit (Có dùng StratifiedKFold)
def save_learning_curve_charts(
    estimators,
    pipeline_factory,
    X_train,
    y_train,
    X_test,
    y_test,
    numeric_features,
    categorical_features,
    binary_features,
    output_dir: Path,
) -> None:
    from sklearn.model_selection import StratifiedKFold, learning_curve
    rows = []
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, estimator in estimators.items():
        print(f"Đang tạo learning curve (CV): {model_name}")

        model = pipeline_factory(
            estimator,
            numeric_features,
            categorical_features,
            binary_features,
        )

        train_sizes_abs, train_scores_f1, test_scores_f1 = learning_curve(
            model, X_train, y_train, cv=cv, scoring='f1', 
            train_sizes=np.linspace(0.15, 1.0, 8), random_state=42, n_jobs=-1
        )
        
        train_sizes_abs, train_scores_acc, test_scores_acc = learning_curve(
            model, X_train, y_train, cv=cv, scoring='accuracy', 
            train_sizes=np.linspace(0.15, 1.0, 8), random_state=42, n_jobs=-1
        )

        for i, size in enumerate(train_sizes_abs):
            rows.append(
                {
                    "Model": model_name,
                    "Train Size": size,
                    "Train F1": train_scores_f1[i].mean(),
                    "CV Val F1": test_scores_f1[i].mean(),
                    "Train Accuracy": train_scores_acc[i].mean(),
                    "CV Val Accuracy": test_scores_acc[i].mean(),
                }
            )

    learning_curve_df = pd.DataFrame(rows)
    learning_curve_df.to_csv(output_dir / "result model"/ "learning_curves_all_models.csv", index=False)

    for model_name in learning_curve_df["Model"].unique():
        model_df = learning_curve_df[learning_curve_df["Model"] == model_name]
        slug = make_filename_slug(model_name)

        for metric in ["F1", "Accuracy"]:
            plt.figure(figsize=(10, 5))
            plt.plot(model_df["Train Size"], model_df[f"Train {metric}"], marker="o")
            plt.plot(model_df["Train Size"], model_df[f"CV Val {metric}"], marker="s")
            plt.xlabel("Số lượng mẫu train (Train Size)")
            plt.ylabel(metric)
            plt.ylim(0, 1)
            plt.title(f"Learning Curve (Stratified 5-Fold) {metric} - {model_name}")
            plt.legend([f"Train {metric}", f"CV Val {metric}"])
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(output_dir / "result model"/ f"learning_curve_{slug}_{metric.lower()}.png", dpi=150)
            plt.close()

# Vẽ validation curve cho Decision Tree để kiểm tra ảnh hưởng theo max_depth (Có dùng StratifiedKFold)
def save_decision_tree_validation_curve(
    pipeline_factory,
    X_train,
    y_train,
    X_test,
    y_test,
    numeric_features,
    categorical_features,
    binary_features,
    output_dir: Path,
) -> None:
    from sklearn.model_selection import StratifiedKFold, cross_validate
    rows = []
    max_depth_values = [1, 2, 3, 4, 5, 7, 10, None]
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Đang tạo validation curve cho Decision Tree (CV)...")
    for max_depth in max_depth_values:
        model = pipeline_factory(
            DecisionTreeClassifier(max_depth=max_depth, random_state=42),
            numeric_features,
            categorical_features,
            binary_features,
        )
        
        scores = cross_validate(
            model, X_train, y_train, cv=cv, 
            scoring={'f1': 'f1', 'accuracy': 'accuracy'},
            return_train_score=True, n_jobs=-1
        )

        rows.append(
            {
                "max_depth": "None" if max_depth is None else max_depth,
                "complexity_index": len(rows) + 1,
                "Train F1": scores["train_f1"].mean(),
                "CV Val F1": scores["test_f1"].mean(),
                "Train Accuracy": scores["train_accuracy"].mean(),
                "CV Val Accuracy": scores["test_accuracy"].mean(),
            }
        )

    curve_df = pd.DataFrame(rows)
    curve_df.to_csv(output_dir / "result model"/"validation_curve_decision_tree.csv", index=False)

    for metric in ["F1", "Accuracy"]:
        plt.figure(figsize=(9, 5))
        plt.plot(curve_df["complexity_index"], curve_df[f"Train {metric}"], marker="o")
        plt.plot(curve_df["complexity_index"], curve_df[f"CV Val {metric}"], marker="s")
        plt.xticks(curve_df["complexity_index"], curve_df["max_depth"])
        plt.xlabel("max_depth")
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.title(f"Validation Curve (Stratified 5-Fold) Decision Tree - {metric}")
        plt.legend([f"Train {metric}", f"CV Val {metric}"])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / "result model"/f"validation_curve_decision_tree_{metric.lower()}.png", dpi=150)
        plt.close()
