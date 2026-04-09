"""
Train XGBoost classifiers on symptom-disease dataset.
Usage: python scripts/train_classifier.py --data data/your_dataset.csv
"""
import argparse
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
from pathlib import Path

from app.ml.preprocessor import get_symptom_list, SYMPTOM_SYNONYMS


def load_and_detect_format(csv_path: str) -> pd.DataFrame:
    """Load CSV and auto-detect its format."""
    import time
    t0 = time.time()
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} cols in {time.time()-t0:.1f}s")
    return df


def extract_symptoms_from_row(row: pd.Series, symptom_columns: list[str]) -> list[str]:
    """Extract non-null symptom values from a row."""
    symptoms = []
    for col in symptom_columns:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            cleaned = str(val).strip().lower().replace(" ", "_")
            symptoms.append(cleaned)
    return symptoms


def build_feature_matrix(df: pd.DataFrame, symptom_columns: list[str]) -> np.ndarray:
    """Build binary feature matrix from dataframe.

    Fast path: if symptom columns are already numeric (int/float), read directly as numpy.
    Slow path: extract string symptom names and vectorize (for text-based CSVs).
    """
    import time
    t0 = time.time()

    # Check if columns are already numeric by dtype (avoids expensive pd.to_numeric scan)
    sample_dtypes = df[symptom_columns].dtypes
    all_numeric = all(np.issubdtype(dt, np.number) for dt in sample_dtypes)

    if all_numeric:
        X = df[symptom_columns].fillna(0).values.astype(np.float32)
        all_symptoms = [c.strip().lower().replace(" ", "_") for c in symptom_columns]
        print(f"Feature matrix: {X.shape} ({len(all_symptoms)} symptoms) built in {time.time()-t0:.1f}s")
        return X, all_symptoms

    # Mixed types: coerce non-numeric to 0
    print("Coercing mixed-type symptom columns to numeric...")
    numeric_df = df[symptom_columns].apply(pd.to_numeric, errors='coerce')
    bad_count = numeric_df.isna().any(axis=1).sum()
    if bad_count > 0:
        print(f"  NOTE: Coerced {bad_count} rows with non-numeric symptom values to 0")
    X = numeric_df.fillna(0).values.astype(np.float32)
    all_symptoms = [c.strip().lower().replace(" ", "_") for c in symptom_columns]
    print(f"Feature matrix: {X.shape} built in {time.time()-t0:.1f}s")
    return X, all_symptoms



def auto_detect_columns(df: pd.DataFrame) -> tuple[str, list[str], str | None]:
    """Auto-detect disease column, symptom columns, and urgency column."""
    cols_lower = {c.lower().strip(): c for c in df.columns}

    # Detect disease/prognosis column
    disease_col = None
    for candidate in ["disease", "prognosis", "diagnosis", "condition", "pathology", "label"]:
        if candidate in cols_lower:
            disease_col = cols_lower[candidate]
            break

    if disease_col is None:
        # Try first column
        disease_col = df.columns[0]
        print(f"WARNING: Could not detect disease column, using '{disease_col}'")

    # Detect urgency column
    urgency_col = None
    for candidate in ["urgency", "severity", "triage", "priority", "urgency_level"]:
        if candidate in cols_lower:
            urgency_col = cols_lower[candidate]
            break

    # Detect symptom columns (everything that's not disease or urgency)
    skip = {disease_col, urgency_col} if urgency_col else {disease_col}
    # Also skip common non-symptom columns
    skip_names = {"id", "index", "unnamed", "patient_id", "age", "gender", "sex", "description", "precaution"}

    symptom_cols = []
    for c in df.columns:
        if c in skip:
            continue
        if c.lower().strip() in skip_names:
            continue
        if c.lower().startswith("unnamed"):
            continue
        symptom_cols.append(c)

    print(f"Disease column: {disease_col}")
    print(f"Urgency column: {urgency_col}")
    print(f"Symptom columns ({len(symptom_cols)}): {symptom_cols[:5]}...")

    return disease_col, symptom_cols, urgency_col


def assign_urgency(disease: str) -> str:
    """Auto-assign urgency based on disease name if no urgency column exists."""
    disease_lower = str(disease).lower()

    emergency = ["heart attack", "stroke", "cardiac", "meningitis", "sepsis", "hemorrhage",
                 "anaphylaxis", "pulmonary embolism", "brain", "paralysis"]
    urgent = ["dengue", "malaria", "pneumonia", "typhoid", "hepatitis", "tuberculosis", "tb",
              "cholera", "appendicitis", "kidney", "diabetes", "hypertension", "jaundice",
              "chikungunya", "leptospirosis", "encephalitis"]
    routine = ["uti", "urinary", "infection", "bronchitis", "gastritis", "arthritis",
               "anemia", "thyroid", "asthma", "migraine", "sinusitis"]

    for word in emergency:
        if word in disease_lower:
            return "emergency"
    for word in urgent:
        if word in disease_lower:
            return "urgent"
    for word in routine:
        if word in disease_lower:
            return "routine"
    return "self_care"


def detect_gpu() -> bool:
    """Check if CUDA GPU is available for XGBoost."""
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Train triage classifiers")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--gpu", action="store_true", default=None,
                        help="Force GPU training (auto-detected if omitted)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Force CPU training")
    args = parser.parse_args()

    # Determine device
    if args.no_gpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = True
    else:
        use_gpu = detect_gpu()

    device = "cuda" if use_gpu else "cpu"
    print(f"Training device: {device.upper()}")

    # Load data
    df = load_and_detect_format(args.data)
    disease_col, symptom_cols, urgency_col = auto_detect_columns(df)

    # Drop rows with no disease
    df = df.dropna(subset=[disease_col])
    print(f"After dropping nulls: {len(df)} rows")

    # Build feature matrix
    X, all_symptoms = build_feature_matrix(df, symptom_cols)

    # Encode disease labels
    disease_encoder = LabelEncoder()
    y_disease = disease_encoder.fit_transform(df[disease_col].astype(str))
    disease_labels = list(disease_encoder.classes_)
    print(f"Disease classes: {len(disease_labels)}")

    # Urgency labels (copy to avoid PerformanceWarning on fragmented DataFrame)
    df = df.copy()
    if urgency_col and urgency_col in df.columns:
        df["_urgency"] = df[urgency_col].astype(str).str.lower().str.strip()
    else:
        print("No urgency column found — auto-assigning based on disease names")
        df["_urgency"] = df[disease_col].apply(assign_urgency)

    urgency_encoder = LabelEncoder()
    urgency_encoder.classes_ = np.array(["self_care", "routine", "urgent", "emergency"])

    # Map to valid labels
    valid_urgencies = {"self_care", "routine", "urgent", "emergency"}
    df["_urgency"] = df["_urgency"].apply(lambda x: x if x in valid_urgencies else "routine")
    y_urgency = urgency_encoder.transform(df["_urgency"])

    print(f"Urgency distribution:\n{pd.Series(df['_urgency']).value_counts()}")

    # Train/test split
    X_train, X_test, y_urg_train, y_urg_test, y_dis_train, y_dis_test = train_test_split(
        X, y_urgency, y_disease, test_size=0.2, random_state=42, stratify=y_urgency
    )

    # Re-map disease labels to contiguous 0..N-1 after split
    # (some rare diseases may only appear in test set due to urgency-stratified split)
    unique_train_classes = np.unique(y_dis_train)
    class_mapping = {old: new for new, old in enumerate(unique_train_classes)}
    y_dis_train = np.array([class_mapping[y] for y in y_dis_train])

    # Filter test set: keep only samples whose disease was seen in training
    test_mask = np.isin(y_dis_test, unique_train_classes)
    dropped = (~test_mask).sum()
    if dropped > 0:
        print(f"NOTE: Dropped {dropped} test samples with diseases not seen in training")
    X_test_dis = X_test[test_mask]
    y_dis_test = np.array([class_mapping[y] for y in y_dis_test[test_mask]])

    # Update disease_labels to match the new contiguous encoding
    disease_labels = [disease_labels[i] for i in unique_train_classes]
    num_disease_classes = len(disease_labels)
    print(f"Disease classes in training: {num_disease_classes}")

    # Train urgency classifier
    print("\n" + "="*60)
    print("Training URGENCY classifier...")
    print("="*60)
    urg_model = XGBClassifier(
        n_estimators=100,      # 4 classes doesn't need 200 trees
        max_depth=4,
        learning_rate=0.2,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=42,
        use_label_encoder=False,
        tree_method="hist",
        device=device,
    )
    urg_model.fit(X_train, y_urg_train)

    y_urg_pred = urg_model.predict(X_test)
    print(f"Urgency Accuracy: {accuracy_score(y_urg_test, y_urg_pred):.4f}")
    print(classification_report(y_urg_test, y_urg_pred,
          target_names=["self_care", "routine", "urgent", "emergency"], zero_division=0))

    # Train disease classifier
    print("\n" + "="*60)
    print("Training DISEASE classifier...")
    print("="*60)
    dis_model = XGBClassifier(
        n_estimators=50,       # Reduced from 200: 1,249 classes makes each iteration expensive
        max_depth=4,           # Reduced from 6: avoids overfitting on rare diseases
        learning_rate=0.2,     # Bumped from 0.1: compensate for fewer trees
        objective="multi:softprob",
        num_class=num_disease_classes,
        eval_metric="mlogloss",
        random_state=42,
        use_label_encoder=False,
        tree_method="hist",
        device=device,
    )
    dis_model.fit(X_train, y_dis_train)

    y_dis_pred = dis_model.predict(X_test_dis)
    print(f"Disease Accuracy: {accuracy_score(y_dis_test, y_dis_pred):.4f}")

    # Save models
    output_dir = Path("app/ml/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(urg_model, output_dir / "urgency_classifier.joblib")
    joblib.dump(dis_model, output_dir / "disease_classifier.joblib")

    with open(output_dir / "disease_labels.json", "w") as f:
        json.dump(disease_labels, f, indent=2)

    with open(output_dir / "symptom_list.json", "w") as f:
        json.dump(all_symptoms, f, indent=2)

    print(f"\nModels saved to {output_dir}/")
    print("Files: urgency_classifier.joblib, disease_classifier.joblib, disease_labels.json, symptom_list.json")


if __name__ == "__main__":
    main()
