"""
Merge multiple source datasets into a single clean training dataset with NAMED symptom columns.

Sources:
  1. symbipredict_2022.csv — 132 named symptoms, 41 diseases, 4,961 rows
  2. Disease and symptoms dataset.csv — 377 named symptoms, 773 diseases, 246,945 rows
  3. Indian-Healthcare-Symptom-Disease-Dataset.csv — severity/urgency metadata for mapping

Output: data/merged_symptoms_diseases.csv with columns:
  disease, urgency, symptom_name_1, symptom_name_2, ... (all named, binary 0/1)
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def normalize_col(name: str) -> str:
    """Normalize a column/symptom name to a consistent format."""
    return name.lower().strip().replace(" ", "_").replace("-", "_")


def load_symbipredict(path: str) -> pd.DataFrame:
    """Load symbipredict_2022.csv (132 named symptoms + prognosis)."""
    df = pd.read_csv(path)
    print(f"Loaded symbipredict: {len(df)} rows, {len(df.columns)} cols")

    # Rename columns
    rename = {}
    symptom_cols = []
    for c in df.columns:
        if c == "prognosis":
            rename[c] = "disease"
        else:
            norm = normalize_col(c)
            rename[c] = norm
            symptom_cols.append(norm)

    df = df.rename(columns=rename)
    df["disease"] = df["disease"].str.lower().str.strip()

    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.startswith("unnamed")]

    return df, symptom_cols


def load_disease_symptoms(path: str) -> pd.DataFrame:
    """Load Disease and symptoms dataset.csv (377 named symptoms + diseases)."""
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded Disease&Symptoms: {len(df)} rows, {len(df.columns)} cols")

    # Rename columns
    rename = {}
    symptom_cols = []
    for c in df.columns:
        if c.lower().strip() == "diseases":
            rename[c] = "disease"
        else:
            norm = normalize_col(c)
            rename[c] = norm
            symptom_cols.append(norm)

    df = df.rename(columns=rename)
    df["disease"] = df["disease"].str.lower().str.strip()

    # Drop non-numeric symptom columns (contamination check)
    for col in symptom_cols[:]:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        except Exception:
            print(f"  WARNING: Dropping non-convertible column: {col}")
            df = df.drop(columns=[col])
            symptom_cols.remove(col)

    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.startswith("unnamed")]

    return df, symptom_cols


def load_severity_metadata(path: str) -> dict:
    """Load Indian Healthcare metadata for severity -> urgency mapping."""
    df = pd.read_csv(path)
    print(f"Loaded severity metadata: {len(df)} entries")

    # Map severity to urgency
    severity_to_urgency = {
        "Mild": "self_care",
        "Moderate": "routine",
        "Severe": "emergency",
    }

    # Build symptom -> urgency hint mapping
    symptom_severity = {}
    for _, row in df.iterrows():
        symptom = normalize_col(str(row["Symptom"]))
        severity = str(row.get("Severity", "Moderate")).strip()
        urgency = severity_to_urgency.get(severity, "routine")
        symptom_severity[symptom] = urgency

    return symptom_severity


def assign_urgency_from_disease(disease: str) -> str:
    """Assign urgency based on disease name keywords."""
    d = disease.lower()

    emergency_kw = [
        "heart attack", "stroke", "cardiac", "meningitis", "sepsis",
        "hemorrhage", "anaphylaxis", "pulmonary embolism", "paralysis",
        "brain hemorrhage", "heart failure", "myocardial",
    ]
    urgent_kw = [
        "dengue", "malaria", "pneumonia", "typhoid", "hepatitis",
        "tuberculosis", "cholera", "appendicitis", "kidney", "diabetes",
        "hypertension", "jaundice", "chikungunya", "encephalitis",
        "hiv", "aids", "cancer", "tumor", "leukemia", "lymphoma",
    ]
    self_care_kw = [
        "common cold", "acne", "dandruff", "pimple",
    ]

    for kw in emergency_kw:
        if kw in d:
            return "emergency"
    for kw in urgent_kw:
        if kw in d:
            return "urgent"
    for kw in self_care_kw:
        if kw in d:
            return "self_care"
    return "routine"


def main():
    parser = argparse.ArgumentParser(description="Merge source datasets")
    parser.add_argument("--source-dir", default="d:/Navya Singhal/dattaset",
                        help="Directory containing source CSVs")
    parser.add_argument("--output", default="data/merged_symptoms_diseases.csv",
                        help="Output path for merged dataset")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="Drop diseases with fewer samples")
    args = parser.parse_args()

    src = Path(args.source_dir)

    # Load all sources
    print("=" * 70)
    print("LOADING SOURCE DATASETS")
    print("=" * 70)

    df1, syms1 = load_symbipredict(str(src / "symbipredict_2022.csv"))
    df2, syms2 = load_disease_symptoms(str(src / "Disease and symptoms dataset.csv"))
    severity_map = load_severity_metadata(
        str(src / "Indian-Healthcare-Symptom-Disease-Dataset - Sheet1 (2).csv")
    )

    # Build unified symptom list (union of both datasets)
    all_symptoms = sorted(set(syms1) | set(syms2))
    print(f"\nUnified symptom list: {len(all_symptoms)} unique symptoms")
    print(f"  From symbipredict: {len(syms1)}")
    print(f"  From Disease&Symptoms: {len(syms2)}")
    print(f"  Overlap: {len(set(syms1) & set(syms2))}")

    # Align both datasets to the unified symptom list
    print("\n" + "=" * 70)
    print("ALIGNING DATASETS TO UNIFIED SCHEMA")
    print("=" * 70)

    # For df1: add missing symptom columns as zeros
    for sym in all_symptoms:
        if sym not in df1.columns:
            df1[sym] = 0
    df1 = df1[["disease"] + all_symptoms]

    # For df2: add missing symptom columns as zeros
    for sym in all_symptoms:
        if sym not in df2.columns:
            df2[sym] = 0
    df2 = df2[["disease"] + all_symptoms]

    print(f"Dataset 1 aligned: {df1.shape}")
    print(f"Dataset 2 aligned: {df2.shape}")

    # Concatenate
    merged = pd.concat([df1, df2], ignore_index=True)
    print(f"Merged: {len(merged)} rows")

    # Drop exact duplicates
    before = len(merged)
    merged = merged.drop_duplicates()
    print(f"After dedup: {len(merged)} rows (dropped {before - len(merged)})")

    # Assign urgency
    print("\n" + "=" * 70)
    print("ASSIGNING URGENCY LEVELS")
    print("=" * 70)
    merged["urgency"] = merged["disease"].apply(assign_urgency_from_disease)
    print(merged["urgency"].value_counts().to_string())

    # Filter rare diseases
    print(f"\n" + "=" * 70)
    print(f"FILTERING DISEASES WITH < {args.min_samples} SAMPLES")
    print("=" * 70)
    disease_counts = merged["disease"].value_counts()
    rare = disease_counts[disease_counts < args.min_samples].index
    print(f"Diseases before: {merged['disease'].nunique()}")
    print(f"Rare diseases (< {args.min_samples} samples): {len(rare)}")
    merged = merged[~merged["disease"].isin(rare)]
    print(f"Diseases after: {merged['disease'].nunique()}")
    print(f"Rows after: {len(merged)}")

    # Drop dead symptom columns
    print("\n" + "=" * 70)
    print("DROPPING DEAD SYMPTOM COLUMNS")
    print("=" * 70)
    col_sums = merged[all_symptoms].sum()
    dead = col_sums[col_sums == 0].index.tolist()
    near_dead = col_sums[(col_sums > 0) & (col_sums < 5)].index.tolist()
    drop_cols = dead + near_dead
    print(f"Dead (all zeros): {len(dead)}")
    print(f"Near-dead (< 5 uses): {len(near_dead)}")
    merged = merged.drop(columns=drop_cols)
    remaining_symptoms = [c for c in merged.columns if c not in ["disease", "urgency"]]
    print(f"Remaining symptom columns: {len(remaining_symptoms)}")

    # Rebalance urgency
    print("\n" + "=" * 70)
    print("REBALANCING URGENCY CLASSES")
    print("=" * 70)
    print("Before:")
    print(merged["urgency"].value_counts().to_string())

    majority_count = merged["urgency"].value_counts().max()
    target_min = int(majority_count * 0.15)

    parts = []
    for urg_class in merged["urgency"].unique():
        subset = merged[merged["urgency"] == urg_class]
        if len(subset) < target_min:
            upsampled = subset.sample(n=target_min, replace=True, random_state=42)
            parts.append(upsampled)
            print(f"  Upsampled '{urg_class}': {len(subset)} -> {target_min}")
        else:
            parts.append(subset)
            print(f"  Kept '{urg_class}': {len(subset)}")

    merged = pd.concat(parts, ignore_index=True)
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nAfter:")
    print(merged["urgency"].value_counts().to_string())

    # Reorder columns: disease, urgency, then symptoms alphabetically
    remaining_symptoms = sorted([c for c in merged.columns if c not in ["disease", "urgency"]])
    merged = merged[["disease", "urgency"] + remaining_symptoms]

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL MERGED DATASET")
    print("=" * 70)
    print(f"Rows: {len(merged):,}")
    print(f"Columns: {len(merged.columns)}")
    print(f"Symptom features: {len(remaining_symptoms)} (ALL NAMED)")
    print(f"Unique diseases: {merged['disease'].nunique()}")
    print(f"Sample symptom names: {remaining_symptoms[:10]}")
    print(f"Urgency distribution:")
    for urg, count in merged["urgency"].value_counts().items():
        print(f"  {urg}: {count:,} ({count/len(merged)*100:.1f}%)")

    # Save
    merged.to_csv(args.output, index=False)
    import os
    size = os.path.getsize(args.output)
    print(f"\nSaved to: {args.output} ({size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
