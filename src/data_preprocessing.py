"""
FedShield — Phase 1: Data Preprocessing
========================================
Loads the PaySim dataset, engineers features, handles the extreme class
imbalance, performs a stratified train/test split, and saves the processed
arrays to data/processed/ for downstream phases.

Assumptions documented:
  - Fraud in PaySim only occurs in CASH_OUT and TRANSFER transactions.
    All other types are guaranteed non-fraudulent by the simulator.
  - isFlaggedFraud is a system-level flag (very few flagged); we drop it
    and use isFraud as the sole ground-truth label.
  - To address the 0.13% fraud rate without discarding all fraud signal,
    we undersample the non-fraud class so the train set ratio is ~10:1
    (non-fraud:fraud), giving the GCN a learnable signal.
  - Numerical features are scaled with StandardScaler fitted ONLY on the
    training set, then applied to test — no data leakage.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Paths
RAW_PATH = os.path.join("data", "raw", "PS_20174392719_1491204439457_log.csv")
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Constants
RANDOM_SEED = 42
TEST_SIZE = 0.2
UNDERSAMPLE_RATIO = 10  # non-fraud:fraud ratio in balanced training set

FRAUD_TYPES = {"CASH_OUT", "TRANSFER"}  # Only these can contain fraud in PaySim


def load_raw(path: str) -> pd.DataFrame:
    """Load the full PaySim CSV."""
    print(f"[1/7] Loading raw dataset from {path} ...")
    df = pd.read_csv(path)
    print(f"      Loaded {len(df):,} rows x {len(df.columns)} columns.")
    return df


def drop_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the isFlaggedFraud column (system flag, not our label).
    Keep only CASH_OUT and TRANSFER rows because PaySim only generates
    fraud in these two transaction types.
    """
    print("[2/7] Filtering to fraud-relevant transaction types (CASH_OUT, TRANSFER) ...")
    df = df.drop(columns=["isFlaggedFraud"])
    df = df[df["type"].isin(FRAUD_TYPES)].copy()
    fraud = df["isFraud"].sum()
    total = len(df)
    print(f"      Retained {total:,} rows | Fraud: {fraud:,} ({fraud/total*100:.2f}%)")
    return df


def engineer_features(df: pd.DataFrame):
    """
    Create informative derived features:
      - balance_change_orig  : how much the sender's balance changed
      - balance_change_dest  : how much the receiver's balance changed
      - orig_balance_mismatch: flag when debit != amount (common in fraud)
      - dest_balance_mismatch: flag when credit != amount (common in fraud)
      - type_encoded         : integer-encode the transaction type
    """
    print("[3/7] Engineering features ...")

    df["balance_change_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_change_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    df["orig_balance_mismatch"] = (
        (df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]).abs() > 1
    ).astype(int)

    df["dest_balance_mismatch"] = (
        (df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]).abs() > 1
    ).astype(int)

    le = LabelEncoder()
    df["type_encoded"] = le.fit_transform(df["type"])

    print(f"      Feature engineering complete. Shape: {df.shape}")
    return df, le


def define_feature_columns():
    """Return the final ordered list of model-input feature columns."""
    return [
        "step",
        "type_encoded",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "balance_change_orig",
        "orig_balance_mismatch",
        "oldbalanceDest",
        "newbalanceDest",
        "balance_change_dest",
        "dest_balance_mismatch",
    ]


def split_and_balance(df: pd.DataFrame, feature_cols: list):
    """
    1. Stratified train/test split (80/20) on the full filtered dataset.
    2. Undersample non-fraud in the TRAINING set only.
       Test set is left intact for realistic imbalanced evaluation.
    """
    print(f"[4/7] Splitting data (test_size={TEST_SIZE}, stratified) ...")
    X = df[feature_cols].values
    y = df["isFraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")

    print(f"[5/7] Undersampling non-fraud at {UNDERSAMPLE_RATIO}:1 ratio in training set ...")
    fraud_idx = np.where(y_train == 1)[0]
    nonfraud_idx = np.where(y_train == 0)[0]
    n_keep = min(len(fraud_idx) * UNDERSAMPLE_RATIO, len(nonfraud_idx))

    rng = np.random.default_rng(RANDOM_SEED)
    sampled_nonfraud = rng.choice(nonfraud_idx, size=n_keep, replace=False)
    kept_idx = np.concatenate([fraud_idx, sampled_nonfraud])
    rng.shuffle(kept_idx)

    X_train_bal = X_train[kept_idx]
    y_train_bal = y_train[kept_idx]

    fraud_train = y_train_bal.sum()
    print(f"      Balanced train: {len(X_train_bal):,} rows | "
          f"Fraud: {fraud_train:,} ({fraud_train/len(X_train_bal)*100:.1f}%)")
    print(f"      Test (imbalanced, realistic): {len(X_test):,} rows | "
          f"Fraud: {y_test.sum():,} ({y_test.sum()/len(y_test)*100:.2f}%)")

    return X_train_bal, X_test, y_train_bal, y_test


def scale_features(X_train, X_test):
    """Fit StandardScaler on train only, transform both sets."""
    print("[6/7] Scaling numerical features (fit on train only) ...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_test_sc, scaler


def save_processed(X_train, X_test, y_train, y_test, scaler, le, feature_cols, df):
    """Save all processed artifacts to data/processed/."""
    print("[7/7] Saving processed artifacts to data/processed/ ...")

    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    with open(os.path.join(PROCESSED_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(PROCESSED_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    with open(os.path.join(PROCESSED_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

    # Save full filtered+engineered DataFrame for Phase 2 (client partition)
    processed_csv = os.path.join(PROCESSED_DIR, "paysim_filtered.csv")
    df.to_csv(processed_csv, index=False)

    print("      Saved:")
    print(f"        X_train.npy        : {X_train.shape}")
    print(f"        X_test.npy         : {X_test.shape}")
    print(f"        y_train.npy        : {y_train.shape}")
    print(f"        y_test.npy         : {y_test.shape}")
    print(f"        scaler.pkl")
    print(f"        label_encoder.pkl")
    print(f"        feature_cols.pkl")
    print(f"        paysim_filtered.csv : {len(df):,} rows (for Phase 2)")


def run():
    """Execute Phase 1 end-to-end."""
    print("=" * 60)
    print("FedShield - Phase 1: Data Preprocessing")
    print("=" * 60)

    df = load_raw(RAW_PATH)
    df = drop_and_filter(df)
    df, le = engineer_features(df)
    feature_cols = define_feature_columns()
    X_train, X_test, y_train, y_test = split_and_balance(df, feature_cols)
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
    save_processed(X_train_sc, X_test_sc, y_train, y_test, scaler, le, feature_cols, df)

    print()
    print("=" * 60)
    print("Phase 1 COMPLETE. Proceed to Phase 2: Client Partitioning.")
    print("=" * 60)


if __name__ == "__main__":
    run()
