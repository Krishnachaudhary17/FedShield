"""
FedShield — Phase 2: Client Partitioning
==========================================
Partitions the filtered PaySim dataset into 3 logically isolated simulated
banking clients (Bank A, Bank B, Bank C) using a non-IID strategy.

Non-IID rationale:
  Real banks differ in transaction volume, customer base size, and fraud
  exposure. To simulate this realistically, we partition by the originating
  account (nameOrig). Each account is assigned to exactly one bank, so:
    - No transaction appears in more than one client's dataset.
    - The distribution of transaction amounts and fraud rates varies
      naturally between banks (non-IID) because account behaviour varies.
    - Bank A gets 50% of accounts, Bank B gets 30%, Bank C gets 20%.
      This mimics a large, medium, and small institution.

Privacy guarantee:
  After partitioning, each client holds only its own slice. No client
  can access another client's raw transactions. Only model parameters
  are shared with the server in Phase 6.
"""

import os
import numpy as np
import pandas as pd
import pickle

# Paths
PROCESSED_DIR = os.path.join("data", "processed")
FILTERED_CSV = os.path.join(PROCESSED_DIR, "paysim_filtered.csv")
CLIENTS_DIR = os.path.join(PROCESSED_DIR, "clients")
os.makedirs(CLIENTS_DIR, exist_ok=True)

# Non-IID account split ratios (Bank A: large, B: medium, C: small)
CLIENT_RATIOS = {"client1_bankA": 0.50, "client2_bankB": 0.30, "client3_bankC": 0.20}
RANDOM_SEED = 42


def load_filtered(path: str) -> pd.DataFrame:
    print(f"[1/4] Loading filtered dataset from {path} ...")
    df = pd.read_csv(path)
    print(f"      Loaded {len(df):,} rows.")
    return df


def partition_by_account(df: pd.DataFrame) -> dict:
    """
    Assign each unique originating account (nameOrig) to exactly one client.
    All transactions from that account belong to that client only.
    This ensures:
      1. Zero overlap between clients (privacy isolation).
      2. Natural non-IID distribution (accounts vary in behaviour).
    """
    print("[2/4] Partitioning accounts across 3 clients (non-IID by nameOrig) ...")

    unique_accounts = np.array(df["nameOrig"].unique())
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(unique_accounts)

    n = len(unique_accounts)
    splits = {}
    start = 0
    for client_name, ratio in CLIENT_RATIOS.items():
        end = start + int(n * ratio)
        splits[client_name] = set(unique_accounts[start:end])
        start = end
    # Assign any remaining accounts (due to int rounding) to last client
    last = list(CLIENT_RATIOS.keys())[-1]
    splits[last].update(unique_accounts[start:])

    # Build per-client DataFrames
    client_dfs = {}
    for client_name, accounts in splits.items():
        client_df = df[df["nameOrig"].isin(accounts)].copy()
        client_dfs[client_name] = client_df

    return client_dfs


def verify_isolation(client_dfs: dict):
    """Assert zero overlap of transactions between clients."""
    print("[3/4] Verifying client isolation (zero data leakage) ...")
    client_names = list(client_dfs.keys())
    for i in range(len(client_names)):
        for j in range(i + 1, len(client_names)):
            a = set(client_dfs[client_names[i]]["nameOrig"].unique())
            b = set(client_dfs[client_names[j]]["nameOrig"].unique())
            overlap = a & b
            assert len(overlap) == 0, (
                f"ISOLATION FAILURE: {client_names[i]} and {client_names[j]} "
                f"share {len(overlap)} accounts!"
            )
    print("      PASSED — no account appears in more than one client.")


def save_clients(client_dfs: dict):
    """Save each client's partition to data/processed/clients/."""
    print("[4/4] Saving client partitions ...")
    summary = {}
    for client_name, df in client_dfs.items():
        path = os.path.join(CLIENTS_DIR, f"{client_name}.csv")
        df.to_csv(path, index=False)

        fraud = df["isFraud"].sum()
        total = len(df)
        accounts = df["nameOrig"].nunique()
        summary[client_name] = {
            "rows": total,
            "accounts": accounts,
            "fraud": int(fraud),
            "fraud_rate_pct": round(fraud / total * 100, 3),
        }
        print(f"      {client_name:22s} | rows={total:>8,} | "
              f"accounts={accounts:>6,} | fraud={fraud:>5,} "
              f"({fraud/total*100:.2f}%)")

    # Save summary metadata
    summary_path = os.path.join(CLIENTS_DIR, "client_summary.pkl")
    with open(summary_path, "wb") as f:
        pickle.dump(summary, f)
    print(f"      Summary saved to {summary_path}")
    return summary


def run():
    print("=" * 60)
    print("FedShield - Phase 2: Client Partitioning")
    print("=" * 60)

    df = load_filtered(FILTERED_CSV)
    client_dfs = partition_by_account(df)
    verify_isolation(client_dfs)
    summary = save_clients(client_dfs)

    print()
    total_rows = sum(v["rows"] for v in summary.values())
    print(f"      Total rows across all clients: {total_rows:,}")
    print()
    print("=" * 60)
    print("Phase 2 COMPLETE. Proceed to Phase 3: Graph Construction.")
    print("=" * 60)


if __name__ == "__main__":
    run()
