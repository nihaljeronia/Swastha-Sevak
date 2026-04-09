"""
Preprocess and clean the symptoms_diseases dataset.
Fixes: contaminated rows, dead columns, rare diseases, duplicates, urgency imbalance.
Outputs a clean CSV ready for training.

Usage: python scripts/preprocess_dataset.py --input data/symptoms_diseases.csv --output data/symptoms_diseases_clean.csv
"""
import argparse
import pandas as pd
import numpy as np
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Clean and preprocess dataset")
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--output", required=True, help="Path to save cleaned CSV")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="Drop diseases with fewer than N samples (default: 10)")
    args = parser.parse_args()

    print("=" * 70)
    print("STEP 1: Load raw data")
    print("=" * 70)
    df = pd.read_csv(args.input, low_memory=False)
    print(f"Raw: {len(df):,} rows, {len(df.columns)} columns")

    symptom_cols = [c for c in df.columns if c.startswith('symptom_')]
    print(f"Symptom columns: {len(symptom_cols)}")

    # ---- STEP 2: Fix non-numeric contamination ----
    print("\n" + "=" * 70)
    print("STEP 2: Fix contaminated rows (non-numeric symptom values)")
    print("=" * 70)
    numeric_df = df[symptom_cols].apply(pd.to_numeric, errors='coerce')
    non_numeric_mask = df[symptom_cols].notna() & numeric_df.isna()
    bad_row_mask = non_numeric_mask.any(axis=1)
    bad_count = bad_row_mask.sum()
    print(f"Rows with non-numeric symptom values: {bad_count:,}")

    # Drop these rows entirely — they're corrupted data from a bad merge
    df = df[~bad_row_mask].copy()
    # Now safely convert symptom columns to numeric
    df[symptom_cols] = df[symptom_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    print(f"After dropping contaminated rows: {len(df):,} rows")

    # ---- STEP 3: Drop exact duplicates ----
    print("\n" + "=" * 70)
    print("STEP 3: Remove exact duplicate rows")
    print("=" * 70)
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    print(f"Dropped {dropped:,} exact duplicates -> {len(df):,} rows")

    # ---- STEP 4: Drop dead symptom columns (all zeros) ----
    print("\n" + "=" * 70)
    print("STEP 4: Remove dead symptom columns")
    print("=" * 70)
    col_sums = df[symptom_cols].sum()
    dead_cols = col_sums[col_sums == 0].index.tolist()
    rare_cols = col_sums[(col_sums > 0) & (col_sums < 10)].index.tolist()
    drop_cols = dead_cols + rare_cols
    print(f"Dead columns (all zeros): {len(dead_cols)}")
    print(f"Near-dead columns (<10 rows): {len(rare_cols)}")
    print(f"Dropping {len(drop_cols)} columns total")

    df = df.drop(columns=drop_cols)
    remaining_symptom_cols = [c for c in df.columns if c.startswith('symptom_')]
    print(f"Remaining symptom columns: {len(remaining_symptom_cols)}")

    # ---- STEP 5: Drop rows with no disease or null disease ----
    print("\n" + "=" * 70)
    print("STEP 5: Drop rows with no disease")
    print("=" * 70)
    before = len(df)
    df = df.dropna(subset=['disease'])
    df = df[df['disease'].str.strip() != '']
    print(f"Dropped {before - len(df)} rows with empty/null disease -> {len(df):,} rows")

    # ---- STEP 6: Filter rare diseases ----
    print("\n" + "=" * 70)
    print(f"STEP 6: Filter diseases with < {args.min_samples} samples")
    print("=" * 70)
    disease_counts = df['disease'].value_counts()
    rare_diseases = disease_counts[disease_counts < args.min_samples].index
    print(f"Total unique diseases: {len(disease_counts)}")
    print(f"Diseases with < {args.min_samples} samples: {len(rare_diseases)}")
    print(f"Rows in rare diseases: {df[df['disease'].isin(rare_diseases)].shape[0]:,}")

    before = len(df)
    df = df[~df['disease'].isin(rare_diseases)]
    print(f"After filtering: {len(df):,} rows, {df['disease'].nunique()} diseases")

    # ---- STEP 7: Rebalance urgency via upsampling minorities ----
    print("\n" + "=" * 70)
    print("STEP 7: Rebalance urgency classes")
    print("=" * 70)
    print("Before rebalancing:")
    print(df['urgency'].value_counts().to_string())

    # Upsample minority urgency classes to at least 10% of the majority
    majority_count = df['urgency'].value_counts().max()
    target_min = int(majority_count * 0.15)  # At least 15% of majority

    parts = []
    for urg_class in df['urgency'].unique():
        subset = df[df['urgency'] == urg_class]
        if len(subset) < target_min:
            # Upsample with replacement
            upsampled = subset.sample(n=target_min, replace=True, random_state=42)
            parts.append(upsampled)
            print(f"  Upsampled '{urg_class}': {len(subset):,} -> {target_min:,}")
        else:
            parts.append(subset)
            print(f"  Kept '{urg_class}': {len(subset):,}")

    df = pd.concat(parts, ignore_index=True)
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nAfter rebalancing:")
    print(df['urgency'].value_counts().to_string())

    # ---- STEP 8: Drop rows where all symptoms are zero ----
    print("\n" + "=" * 70)
    print("STEP 8: Drop rows with no symptoms active")
    print("=" * 70)
    remaining_symptom_cols = [c for c in df.columns if c.startswith('symptom_')]
    zero_mask = (df[remaining_symptom_cols].sum(axis=1) == 0)
    zero_count = zero_mask.sum()
    print(f"Rows with all-zero symptoms: {zero_count}")
    df = df[~zero_mask]

    # ---- FINAL SUMMARY ----
    remaining_symptom_cols = [c for c in df.columns if c.startswith('symptom_')]
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Symptom features: {len(remaining_symptom_cols)}")
    print(f"Unique diseases: {df['disease'].nunique()}")
    print(f"Urgency distribution:")
    urg_vc = df['urgency'].value_counts()
    for urg, count in urg_vc.items():
        print(f"  {urg}: {count:,} ({count/len(df)*100:.1f}%)")
    print(f"Sparsity: {(df[remaining_symptom_cols] == 0).sum().sum() / (len(df) * len(remaining_symptom_cols)) * 100:.1f}% zeros")

    # Save
    df.to_csv(args.output, index=False)
    print(f"\nSaved cleaned dataset to: {args.output}")
    print(f"File size: {pd.io.common.file_exists(args.output)}")


if __name__ == "__main__":
    main()
