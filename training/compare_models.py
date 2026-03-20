import json
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from src.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    CLASSIFIER_PATH,
    CLASS_NAMES_PATH,
)
from src.feature_engineering import FEATURE_COLUMNS


ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(exist_ok=True)

MODEL_COMPARISON_CSV = ASSETS_DIR / "model_comparison.csv"
MODEL_COMPARISON_PNG = ASSETS_DIR / "model_comparison.png"
CONFUSION_MATRIX_PNG = ASSETS_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_TXT = ASSETS_DIR / "classification_report.txt"


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    X_train = full_train_df[FEATURE_COLUMNS]
    y_train = full_train_df["target"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target"]

    return X_train, y_train, X_test, y_test


def build_models():
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=42))
        ]),
        "linear_svm": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearSVC(random_state=42, dual="auto"))
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=2000,
                random_state=42
            ))
        ]),
    }


def save_model_comparison_plot(results_df: pd.DataFrame):
    plot_df = results_df.sort_values("cv_f1_macro_mean", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["model"], plot_df["cv_f1_macro_mean"])
    plt.xlabel("Cross-validation Macro F1")
    plt.ylabel("Model")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(MODEL_COMPARISON_PNG, dpi=200, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(cm, labels):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title("Confusion Matrix - Best Model")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PNG, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_data()

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "f1_macro": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
    }

    results = []

    for model_name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train_enc,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        results.append({
            "model": model_name,
            "cv_f1_macro_mean": scores["test_f1_macro"].mean(),
            "cv_f1_macro_std": scores["test_f1_macro"].std(),
            "cv_bal_acc_mean": scores["test_balanced_accuracy"].mean(),
            "cv_acc_mean": scores["test_accuracy"].mean(),
        })

    results_df = pd.DataFrame(results).sort_values("cv_f1_macro_mean", ascending=False)

    print("\n=== CROSS-VALIDATION RESULTS ===")
    print(results_df.to_string(index=False))

    # zapis tabeli wyników
    results_df.to_csv(MODEL_COMPARISON_CSV, index=False)

    # zapis wykresu porównania
    save_model_comparison_plot(results_df)

    best_model_name = results_df.iloc[0]["model"]
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train_enc)

    y_pred = best_model.predict(X_test)

    f1 = f1_score(y_test_enc, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_test_enc, y_pred)

    print(f"\n=== BEST MODEL: {best_model_name} ===")
    print(f"Test Macro F1: {f1:.4f}")
    print(f"Test Balanced Accuracy: {bal_acc:.4f}")

    report_text = classification_report(
        y_test_enc,
        y_pred,
        target_names=label_encoder.classes_
    )

    print("\n=== CLASSIFICATION REPORT ===")
    print(report_text)

    with open(CLASSIFICATION_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("Best model: " + best_model_name + "\n\n")
        f.write(f"Test Macro F1: {f1:.4f}\n")
        f.write(f"Test Balanced Accuracy: {bal_acc:.4f}\n\n")
        f.write(report_text)

    cm = confusion_matrix(y_test_enc, y_pred)
    save_confusion_matrix(cm, label_encoder.classes_)

    joblib.dump(best_model, CLASSIFIER_PATH)

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(label_encoder.classes_.tolist(), f, ensure_ascii=False, indent=2)

    print(f"\nSaved model to: {CLASSIFIER_PATH}")
    print(f"Saved class names to: {CLASS_NAMES_PATH}")
    print(f"Saved comparison table to: {MODEL_COMPARISON_CSV}")
    print(f"Saved comparison chart to: {MODEL_COMPARISON_PNG}")
    print(f"Saved confusion matrix to: {CONFUSION_MATRIX_PNG}")
    print(f"Saved classification report to: {CLASSIFICATION_REPORT_TXT}")


if __name__ == "__main__":
    main()