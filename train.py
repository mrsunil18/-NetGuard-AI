from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = MODEL_DIR / "intrusion_model.joblib"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

# A practical cap keeps training manageable on a normal laptop.
# Increase this later if your laptop has enough memory.
MAX_ROWS = 150_000
RANDOM_STATE = 42

def clean_column_name(name):
    return str(name).replace("\ufeff", "").strip()

def find_label_column(columns):
    for col in columns:
        normalized = clean_column_name(col).lower()
        if normalized in {"label", "class", "attack", "target"}:
            return col
    return None

def load_csvs():
    csv_paths = list(RAW_DIR.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}.\n"
            "Extract CIC-IDS2017 MachineLearningCSV.zip and put its CSV files "
            "inside the data/raw folder."
        )

    frames = []
    per_file_limit = max(1, MAX_ROWS // len(csv_paths))
    for path in csv_paths:
        print(f"Reading: {path.name}")
        try:
            df = pd.read_csv(path, low_memory=False, nrows=per_file_limit * 2)
        except Exception as exc:
            print(f"  Skipping unreadable file: {exc}")
            continue

        df.columns = [clean_column_name(c) for c in df.columns]
        label_col = find_label_column(df.columns)
        if label_col is None:
            print("  Skipping: could not find a Label/Class column.")
            continue

        df = df.rename(columns={label_col: "Label"})
        if len(df) > per_file_limit:
            # Keep the sample reproducible. Later, increase MAX_ROWS to train on more data.
            df = df.sample(n=per_file_limit, random_state=RANDOM_STATE)
        frames.append(df)
        print(f"  Kept {len(df):,} rows")

    if not frames:
        raise ValueError("No usable CSV files found. Check that the CIC-IDS2017 CSVs were extracted.")

    data = pd.concat(frames, ignore_index=True, sort=False)
    data.columns = [clean_column_name(c) for c in data.columns]
    data["Label"] = data["Label"].astype(str).str.strip()
    data = data.dropna(subset=["Label"])
    data = data[data["Label"] != "nan"]

    # Binary target: BENIGN vs ATTACK. This is a first, understandable IDS model.
    data["Target"] = data["Label"].str.upper().apply(
        lambda label: 0 if label == "BENIGN" else 1
    )

    # Drop identifiers / potential leakage fields. These should not be used as predictors.
    remove = []
    for col in data.columns:
        lower = col.lower().strip()
        if col in {"Label", "Target"}:
            continue
        if lower in {
            "flow id", "source ip", "src ip", "destination ip", "dst ip",
            "timestamp", "simillarhttp", "similarhttp"
        }:
            remove.append(col)
    data = data.drop(columns=remove, errors="ignore")

    # Convert all feature columns to numeric; CIC flow CSVs can contain infinity/strings.
    X = data.drop(columns=["Label", "Target"], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Remove features that are entirely missing or constant.
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1]

    # Remove rows with no useful numeric feature values.
    keep = X.notna().any(axis=1)
    X = X.loc[keep].reset_index(drop=True)
    y = data.loc[keep, "Target"].reset_index(drop=True)

    # Remove exact duplicate rows after feature preparation.
    combined = X.copy()
    combined["__target__"] = y
    combined = combined.drop_duplicates()
    y = combined.pop("__target__").astype(int).reset_index(drop=True)
    X = combined.reset_index(drop=True)

    if y.nunique() < 2:
        raise ValueError("Only one class remains. Ensure both BENIGN and attack CSVs are included.")

    return X, y, data.loc[keep, "Label"].reset_index(drop=True)

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, original_labels = load_csvs()
    print(f"\nDataset after cleaning: {len(X):,} rows, {X.shape[1]} features")
    print("Binary class counts:")
    print(y.value_counts().rename(index={0: "BENIGN", 1: "ATTACK"}))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            max_depth=22,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ])

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "rows_used": int(len(X)),
        "feature_count": int(X.shape[1]),
        "benign_rows": int((y == 0).sum()),
        "attack_rows": int((y == 1).sum()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_attack": float(precision_score(y_test, predictions, zero_division=0)),
        "recall_attack": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_attack": float(f1_score(y_test, predictions, zero_division=0)),
        "classification_report": classification_report(
            y_test, predictions, target_names=["BENIGN", "ATTACK"],
            output_dict=True, zero_division=0
        )
    }

    joblib.dump(model, MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps(list(X.columns), indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["BENIGN", "ATTACK"], yticklabels=["BENIGN", "ATTACK"]
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Network Intrusion Detection — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    classifier = model.named_steps["classifier"]
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": classifier.feature_importances_
    }).sort_values("importance", ascending=False).head(20)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    print("\n===== TEST RESULTS =====")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision_attack']:.4f} (ATTACK)")
    print(f"Recall:    {metrics['recall_attack']:.4f} (ATTACK)")
    print(f"F1-score:  {metrics['f1_attack']:.4f} (ATTACK)")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["BENIGN", "ATTACK"], zero_division=0))
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print("Note: these are offline dataset results, not proof of production security.")

if __name__ == "__main__":
    main()
