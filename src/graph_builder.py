"""
FedShield — Phase 3: Graph Construction
=========================================
Converts each client's tabular transaction data into a PyTorch Geometric
(PyG) graph suitable for node-level fraud classification with a GCN.

Graph construction assumptions (documented as required by the brief):
  - NODES  : Every unique account (nameOrig OR nameDest) becomes a node.
             Each account appears at most once regardless of how many
             transactions it appears in.
  - EDGES  : Every transaction becomes a DIRECTED edge from the originating
             account (nameOrig) to the destination account (nameDest).
             Self-loops are retained if they occur.
  - NODE FEATURES (x): Aggregated statistics computed from each account's
             transaction history within this client:
               [0] total_sent          — total amount sent as originator
               [1] total_received      — total amount received as destination
               [2] num_sent            — count of outgoing transactions
               [3] num_received        — count of incoming transactions
               [4] avg_sent            — average outgoing amount
               [5] avg_received        — average incoming amount
               [6] balance_change_orig — mean balance change as originator
               [7] balance_change_dest — mean balance change as destination
               [8] orig_mismatch_rate  — fraction of outgoing txns with balance mismatch
               [9] dest_mismatch_rate  — fraction of incoming txns with balance mismatch
  - LABELS (y): A node is labelled FRAUD (1) if ANY transaction where it
             appears as the originator is fraudulent. Non-originator accounts
             that are never the source of fraud are labelled 0.
             Assumption: a fraudulent originator is the primary actor;
             destination accounts are not labelled as fraudsters here.
  - EDGE ATTRIBUTES (edge_attr): Per-transaction features:
               [0] amount
               [1] step (time)
               [2] balance_change_orig
               [3] balance_change_dest
               [4] orig_balance_mismatch
               [5] dest_balance_mismatch

Privacy guarantee: Each client's graph is built exclusively from its own
local partition. No cross-client data is accessed.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# Paths
CLIENTS_DIR = os.path.join("data", "processed", "clients")
GRAPHS_DIR = os.path.join("data", "processed", "graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)

CLIENT_FILES = {
    "client1_bankA": os.path.join(CLIENTS_DIR, "client1_bankA.csv"),
    "client2_bankB": os.path.join(CLIENTS_DIR, "client2_bankB.csv"),
    "client3_bankC": os.path.join(CLIENTS_DIR, "client3_bankC.csv"),
}


def build_graph_for_client(client_name: str, df: pd.DataFrame) -> Data:
    """
    Build a PyG Data object from a single client's transaction DataFrame.
    Returns a PyG Data object with:
        data.x          — node feature matrix  [num_nodes, 10]
        data.edge_index — edge connectivity     [2, num_edges]
        data.edge_attr  — edge feature matrix   [num_edges, 6]
        data.y          — node labels           [num_nodes]
        data.num_nodes  — total node count
    """
    # ── 1. Build account → node ID mapping ───────────────────────────────
    all_accounts = pd.concat([df["nameOrig"], df["nameDest"]]).unique()
    account_to_id = {acc: idx for idx, acc in enumerate(all_accounts)}
    num_nodes = len(account_to_id)

    # ── 2. Edge index (directed: nameOrig → nameDest) ─────────────────────
    src = df["nameOrig"].map(account_to_id).values
    dst = df["nameDest"].map(account_to_id).values
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    # ── 3. Edge attributes ────────────────────────────────────────────────
    edge_attr = torch.tensor(
        df[["amount", "step", "balance_change_orig", "balance_change_dest",
            "orig_balance_mismatch", "dest_balance_mismatch"]].values,
        dtype=torch.float32
    )

    # ── 4. Node features — aggregate per account (vectorized) ────────────
    node_feats = np.zeros((num_nodes, 10), dtype=np.float32)

    # --- Originator (sender) aggregations ---
    orig_stats = df.groupby("nameOrig").agg(
        total_sent=("amount", "sum"),
        num_sent=("amount", "count"),
        avg_sent=("amount", "mean"),
        mean_balance_change_orig=("balance_change_orig", "mean"),
        orig_mismatch_rate=("orig_balance_mismatch", "mean"),
    ).reset_index()

    orig_ids = orig_stats["nameOrig"].map(account_to_id).values
    node_feats[orig_ids, 0] = orig_stats["total_sent"].values
    node_feats[orig_ids, 2] = orig_stats["num_sent"].values
    node_feats[orig_ids, 4] = orig_stats["avg_sent"].values
    node_feats[orig_ids, 6] = orig_stats["mean_balance_change_orig"].values
    node_feats[orig_ids, 8] = orig_stats["orig_mismatch_rate"].values

    # --- Destination (receiver) aggregations ---
    dest_stats = df.groupby("nameDest").agg(
        total_received=("amount", "sum"),
        num_received=("amount", "count"),
        avg_received=("amount", "mean"),
        mean_balance_change_dest=("balance_change_dest", "mean"),
        dest_mismatch_rate=("dest_balance_mismatch", "mean"),
    ).reset_index()

    dest_mask = dest_stats["nameDest"].isin(account_to_id)
    dest_stats = dest_stats[dest_mask]
    dest_ids = dest_stats["nameDest"].map(account_to_id).values
    node_feats[dest_ids, 1] = dest_stats["total_received"].values
    node_feats[dest_ids, 3] = dest_stats["num_received"].values
    node_feats[dest_ids, 5] = dest_stats["avg_received"].values
    node_feats[dest_ids, 7] = dest_stats["mean_balance_change_dest"].values
    node_feats[dest_ids, 9] = dest_stats["dest_mismatch_rate"].values

    x = torch.tensor(node_feats, dtype=torch.float32)

    # ── 5. Node labels ────────────────────────────────────────────────────
    # A node is labelled 1 if it originated ANY fraudulent transaction
    node_labels = np.zeros(num_nodes, dtype=np.int64)
    fraud_txns = df[df["isFraud"] == 1]
    for acc in fraud_txns["nameOrig"].unique():
        if acc in account_to_id:
            node_labels[account_to_id[acc]] = 1

    y = torch.tensor(node_labels, dtype=torch.long)

    # ── 6. Build PyG Data object ──────────────────────────────────────────
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.num_nodes = num_nodes

    # Store account mapping for traceability
    data.account_to_id = account_to_id

    return data


def run():
    print("=" * 60)
    print("FedShield - Phase 3: Graph Construction")
    print("=" * 60)

    graphs = {}
    for client_name, csv_path in CLIENT_FILES.items():
        print(f"\n[{client_name}] Loading data ...")
        df = pd.read_csv(csv_path)
        print(f"  Rows: {len(df):,} | Fraud txns: {df['isFraud'].sum():,}")

        print(f"  Building transaction graph ...")
        graph = build_graph_for_client(client_name, df)
        graphs[client_name] = graph

        fraud_nodes = int(graph.y.sum())
        print(f"  Graph built:")
        print(f"    Nodes (accounts) : {graph.num_nodes:,}")
        print(f"    Edges (txns)     : {graph.edge_index.shape[1]:,}")
        print(f"    Node features    : {graph.x.shape[1]}")
        print(f"    Edge features    : {graph.edge_attr.shape[1]}")
        print(f"    Fraud nodes      : {fraud_nodes:,} "
              f"({fraud_nodes/graph.num_nodes*100:.3f}%)")
        print(f"    x.shape          : {tuple(graph.x.shape)}")
        print(f"    edge_index.shape : {tuple(graph.edge_index.shape)}")
        print(f"    y.shape          : {tuple(graph.y.shape)}")

        # Save individual graph
        graph_path = os.path.join(GRAPHS_DIR, f"{client_name}_graph.pt")
        torch.save(graph, graph_path)
        print(f"  Saved -> {graph_path}")

    # Save all graphs together
    all_graphs_path = os.path.join(GRAPHS_DIR, "all_graphs.pkl")
    with open(all_graphs_path, "wb") as f:
        pickle.dump(graphs, f)
    print(f"\n  All graphs saved -> {all_graphs_path}")

    print()
    print("=" * 60)
    print("Phase 3 COMPLETE. Proceed to Phase 4: GCN Model (Single Client).")
    print("=" * 60)


if __name__ == "__main__":
    run()
