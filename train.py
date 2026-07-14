"""
Heart Disease Prediction - Training Pipeline
==============================================
Loads the UCI Heart Disease dataset, performs EDA, preprocesses the data,
trains multiple classification models, compares their performance, and
saves the best-performing model + scaler to disk for later inference.

Run:
    python src/train.py
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "heart.csv")
IMAGES_DIR = os.path.join(ROOT, "images")
MODELS_DIR = os.path.join(ROOT, "models")
RANDOM_STATE = 42

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def run_eda(df):
    """Generate and save exploratory data analysis plots."""
    print("\nRunning EDA...")

    # Target distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(x="target", data=df, palette="Set2")
    plt.title("Target Class Distribution (0 = No Disease, 1 = Disease)")
    plt.xlabel("Heart Disease")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "target_distribution.png"))
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(11, 9))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True,
                cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "correlation_heatmap.png"))
    plt.close()

    # Age distribution by target
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="age", hue="target", multiple="stack", bins=20, palette="Set2")
    plt.title("Age Distribution by Heart Disease Status")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "age_distribution.png"))
    plt.close()

    # Chest pain type vs target
    plt.figure(figsize=(6, 4))
    sns.countplot(x="cp", hue="target", data=df, palette="Set2")
    plt.title("Chest Pain Type vs Heart Disease")
    plt.xlabel("Chest Pain Type (cp)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "chest_pain_vs_target.png"))
    plt.close()

    print(f"EDA plots saved to {IMAGES_DIR}")


def preprocess(df):
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
    }


def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = []
    roc_data = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "CV Accuracy (5-fold)": round(cv_score, 4),
        })

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, auc)

        print(f"{name:<25} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | CV Acc: {cv_score:.4f}")

    results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False).reset_index(drop=True)
    return results_df, roc_data


def plot_model_comparison(results_df):
    plt.figure(figsize=(9, 5))
    plot_df = results_df.melt(id_vars="Model", value_vars=["Accuracy", "F1-Score", "ROC-AUC"],
                               var_name="Metric", value_name="Score")
    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette="Set2")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.title("Model Performance Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "model_comparison.png"))
    plt.close()


def plot_roc_curves(roc_data):
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "roc_curves.png"))
    plt.close()


def tune_best_model(best_name, X_train, y_train):
    """Hyperparameter tuning for the best-performing model family."""
    print(f"\nTuning hyperparameters for: {best_name}")

    param_grids = {
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10], "min_samples_split": [2, 5]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3, 4]},
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {"C": [0.01, 0.1, 1, 10]},
        ),
        "Support Vector Machine": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7, 9, 11]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"max_depth": [3, 5, 7, None], "min_samples_split": [2, 5, 10]},
        ),
        "Naive Bayes": (GaussianNB(), {}),
    }

    model, grid = param_grids.get(best_name, (LogisticRegression(max_iter=1000), {}))
    if not grid:
        model.fit(X_train, y_train)
        return model

    search = GridSearchCV(model, grid, cv=5, scoring="f1", n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    return search.best_estimator_


def plot_confusion_matrix(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title(f"Confusion Matrix - {name} (Tuned)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "confusion_matrix_best_model.png"))
    plt.close()
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


def main():
    df = load_data()
    run_eda(df)

    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

    models = get_models()
    print("\nTraining and evaluating models...\n")
    results_df, roc_data = evaluate_models(models, X_train, X_test, y_train, y_test)

    print("\n=== Model Comparison (sorted by F1-Score) ===")
    print(results_df.to_string(index=False))

    plot_model_comparison(results_df)
    plot_roc_curves(roc_data)

    results_df.to_csv(os.path.join(ROOT, "models", "model_comparison_results.csv"), index=False)

    best_name = results_df.iloc[0]["Model"]
    print(f"\nBest baseline model: {best_name}")

    best_model = tune_best_model(best_name, X_train, y_train)

    plot_confusion_matrix(best_model, X_test, y_test, best_name)

    final_acc = accuracy_score(y_test, best_model.predict(X_test))
    final_f1 = f1_score(y_test, best_model.predict(X_test))
    final_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

    print(f"\nFinal Tuned Model ({best_name}) -> Accuracy: {final_acc:.4f} | "
          f"F1: {final_f1:.4f} | ROC-AUC: {final_auc:.4f}")

    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    metadata = {
        "best_model": best_name,
        "accuracy": round(final_acc, 4),
        "f1_score": round(final_f1, 4),
        "roc_auc": round(final_auc, 4),
        "feature_names": feature_names,
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel, scaler, and metadata saved to {MODELS_DIR}/")
    print("Training complete.")


if __name__ == "__main__":
    main()
