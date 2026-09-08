import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')


class Tee:
    """کلاسی برای نوشتن هم‌زمان خروجی روی ترمینال و داخل فایل متنی"""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def analyze_dataset_changes():
    # ---------------------------------------------------------
    # 0. SETUP REPORT DIRECTORY
    # ---------------------------------------------------------
    report_dir = "../report"
    os.makedirs(report_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = os.path.join(report_dir, f"preprocessing_report_{timestamp}.txt")

    sys.stdout = Tee(report_path)

    input_path = "../data/raw/jm1.csv"

    # ---------------------------------------------------------
    # 1. BEFORE PREPROCESSING (Raw Data)
    # ---------------------------------------------------------
    print("=" * 60)
    print("--- BEFORE PREPROCESSING (RAW DATA) ---")
    print("=" * 60)

    df_raw = pd.read_csv(input_path)

    total_rows_raw = len(df_raw)
    total_cols_raw = len(df_raw.columns)

    print(f"Total Rows (Instances): {total_rows_raw}")
    print(f"Total Columns (Features + Target): {total_cols_raw}")

    print("\nClass Distribution (Raw):")
    print(df_raw['defects'].value_counts().to_string())

    # ---------------------------------------------------------
    # 2. APPLYING 5-STEP AUTOMATIC DATA CLEANING
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("--- EXECUTION OF 5 CLEANING STEPS ---")
    print("=" * 60)

    # STEP 1 & 3: Handling Missing & Invalid Values + Data Type Conversion
    df_clean = pd.read_csv(input_path, na_values='?')
    missing_before = df_clean.isna().sum().sum()
    df_clean.dropna(inplace=True)
    print(f"[STEP 1 & 3] Missing/Invalid values cleaned (Dropped rows with NaN/'?').")

    # STEP 2: Removing Duplicate Data
    duplicates_count = df_clean.duplicated().sum()
    df_clean.drop_duplicates(inplace=True)
    print(f"[STEP 2] Duplicate rows removed: {duplicates_count}")

    # STEP 4: Handling Inconsistent Data
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    inconsistent_mask = (df_clean[numeric_cols] < 0).any(axis=1)
    inconsistent_count = inconsistent_mask.sum()
    df_clean = df_clean[~inconsistent_mask]
    print(f"[STEP 4] Inconsistent rows removed (Negative values check): {inconsistent_count}")

    # STEP 5: Handling Outliers (Winsorization at 1st & 99th percentiles)
    feature_cols = [col for col in df_clean.columns if col != 'defects']
    print(f"[STEP 5] Outlier handling will be applied on training data only.")

    # ---------------------------------------------------------
    # 3. TRAIN-TEST SPLIT (BEFORE SCALING/BALANCING)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("--- TRAIN-TEST SPLIT (BEFORE SCALING/BALANCING) ---")
    print("=" * 60)

    X = df_clean.drop('defects', axis=1)
    y = df_clean['defects'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training set size: {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)")
    print(f"Testing set size: {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)")
    print(f"\nClass distribution in TRAIN (before SMOTE):")
    print(y_train.value_counts().to_string())
    print(f"\nClass distribution in TEST (before SMOTE):")
    print(y_test.value_counts().to_string())

    # ---------------------------------------------------------
    # 4. WINSORIZATION (ON TRAINING DATA ONLY)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("--- OUTLIER HANDLING (ON TRAINING DATA ONLY) ---")
    print("=" * 60)

    for col in feature_cols:
        lower_bound = X_train[col].quantile(0.01)
        upper_bound = X_train[col].quantile(0.99)

        X_train[col] = np.clip(X_train[col], lower_bound, upper_bound)
        X_test[col] = np.clip(X_test[col], lower_bound, upper_bound)

    print(f"Outliers capped on {len(feature_cols)} feature columns (using train percentiles).")

    # ---------------------------------------------------------
    # 5. NORMALIZATION (FIT ON TRAINING DATA ONLY)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("--- NORMALIZATION (ON TRAINING DATA) ---")
    print("=" * 60)

    scaler = MinMaxScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )

    print("Scaler fitted on TRAINING data only.")
    print("TRAINING data normalization (min/max):")
    print(f"  Min: {X_train_scaled.min().min():.4f}, Max: {X_train_scaled.max().max():.4f}")
    print("TEST data normalization (min/max, using train parameters):")
    print(f"  Min: {X_test_scaled.min().min():.4f}, Max: {X_test_scaled.max().max():.4f}")

    # ---------------------------------------------------------
    # 6. BALANCING (SMOTE ON BOTH TRAIN AND TEST DATA)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("--- SMOTE BALANCING (ON BOTH TRAIN AND TEST DATA) ---")
    print("=" * 60)

    # ✅ SMOTE برای TRAIN
    smote_train = SMOTE(random_state=42, sampling_strategy=0.8)
    X_train_balanced, y_train_balanced = smote_train.fit_resample(X_train_scaled, y_train)

    print(f"✅ SMOTE applied to TRAINING set.")
    print(f"\nClass distribution in TRAIN (before SMOTE):")
    print(y_train.value_counts().to_string())
    print(f"\nClass distribution in TRAIN (after SMOTE):")
    print(pd.Series(y_train_balanced).value_counts().to_string())

    # ✅ SMOTE برای TEST (جدید!)
    smote_test = SMOTE(random_state=42, sampling_strategy=0.8)
    X_test_balanced, y_test_balanced = smote_test.fit_resample(X_test_scaled, y_test)

    print(f"\n✅ SMOTE applied to TEST set.")
    print(f"\nClass distribution in TEST (before SMOTE):")
    print(y_test.value_counts().to_string())
    print(f"\nClass distribution in TEST (after SMOTE):")
    print(pd.Series(y_test_balanced).value_counts().to_string())

    # ---------------------------------------------------------
    # 7. SUMMARY & SAVING
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("--- SUMMARY OF CHANGES ---")
    print("=" * 60)

    print(f"Raw dataset: {total_rows_raw} rows")
    print(f"After cleaning: {len(X)} rows")
    print(f"Train set (before SMOTE): {len(X_train)} rows")
    print(f"Train set (after SMOTE): {len(X_train_balanced)} rows")
    print(f"Test set (before SMOTE): {len(X_test)} rows")
    print(f"Test set (after SMOTE): {len(X_test_balanced)} rows")
    print(f"Total rows saved: {len(X_train_balanced) + len(X_test_balanced)} rows")

    # Save train data
    X_train_balanced.to_csv("../data/processed/X_train.csv", index=False)
    y_train_balanced_df = pd.DataFrame(y_train_balanced, columns=['defects'])
    y_train_balanced_df.to_csv("../data/processed/y_train.csv", index=False)

    # Save test data
    X_test_balanced.to_csv("../data/processed/X_test.csv", index=False)
    y_test_balanced_df = pd.DataFrame(y_test_balanced, columns=['defects'])
    y_test_balanced_df.to_csv("../data/processed/y_test.csv", index=False)

    print("\n" + "=" * 60)
    print("--- FILES SAVED ---")
    print("=" * 60)
    print("✅ X_train.csv (balanced with SMOTE)")
    print("✅ y_train.csv (balanced with SMOTE)")
    print("✅ X_test.csv (balanced with SMOTE)")
    print("✅ y_test.csv (balanced with SMOTE)")

    print(f"\n[INFO] Report successfully saved to: {report_path}")


if __name__ == "__main__":
    analyze_dataset_changes()