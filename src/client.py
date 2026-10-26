"""
FedShield — Phase 4/5: Local Client Training
=============================================
Implements local GCN training for a single simulated banking client.
Used in Phase 4 (single-client test) and Phase 5 (all 3 clients).

Sampling strategy (no torch-sparse required):
  The full graphs have 800k–1.8M nodes, making full-batch GCN impractical
  on CPU. Instead, we extract a balanced induced subgraph per training run:
    1. Take ALL fraud nodes from the client's graph.
    2. Sample NORMAL_MULTIPLIER × fraud_count normal nodes randomly.
    3. Use torch_geometric.utils.subgraph() to extract the induced subgraph
       (edges between sampled nodes only).
    4. Run full-batch GCN on this manageable subgraph.
  This approach preserves the graph topology for the sampled nodes and
  requires no optional C++ dependencies (torch-sparse / pyg-lib).

Privacy guarantee:
  Each client trains ONLY on its local subgraph data. No raw data leaves
  this function. Only the model state_dict is returned to the server.
"""

import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph as pyg_subgraph
from sklearn.metrics import f1_score, precision_score, recall_score

from src.model import FraudGCN

# Default hyperparameters
DEFAULT_CONFIG = {
    "hidden_dim": 64,
    "dropout": 0.3,
    "lr": 0.01,
    "epochs": 10,
    "normal_multiplier": 20,   # Normal nodes sampled per fraud node
    "random_seed": 42,
}


# ── Balanced subgraph sampling ────────────────────────────────────────────────

def sample_balanced_subgraph(graph, normal_multiplier: int = 20, seed: int = 42):
    """
    Extract a balanced induced subgraph:
      - All fraud nodes
      - normal_multiplier × fraud_count normal nodes (random)
    Returns a new PyG Data object with relabelled node indices and
    train / val / test boolean masks (70 / 15 / 15 split).
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    fraud_mask = graph.y == 1
    normal_mask = graph.y == 0

    fraud_nodes = fraud_mask.nonzero(as_tuple=True)[0]
    normal_nodes = normal_mask.nonzero(as_tuple=True)[0]

    n_fraud = len(fraud_nodes)
    n_normal_sample = min(n_fraud * normal_multiplier, len(normal_nodes))

    perm = torch.randperm(len(normal_nodes), generator=rng)
    sampled_normal = normal_nodes[perm[:n_normal_sample]]

    selected_nodes = torch.cat([fraud_nodes, sampled_normal])

    # Induced subgraph (edges among selected nodes only)
    edge_index_sub, edge_attr_sub = pyg_subgraph(
        selected_nodes,
        graph.edge_index,
        graph.edge_attr,
        relabel_nodes=True,
        num_nodes=graph.num_nodes,
    )

    x_sub = graph.x[selected_nodes]
    y_sub = graph.y[selected_nodes]
    n_sub = len(selected_nodes)

    # ── Stratified 70/15/15 masks ─────────────────────────────────────────
    np_rng = np.random.default_rng(seed)

    def stratified_split(idx_tensor):
        idx = idx_tensor.numpy()
        np_rng.shuffle(idx)
        n_train = int(0.70 * len(idx))
        n_val   = int(0.15 * len(idx))
        return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    fraud_sub_idx = (y_sub == 1).nonzero(as_tuple=True)[0]
    normal_sub_idx = (y_sub == 0).nonzero(as_tuple=True)[0]

    f_tr, f_v, f_te = stratified_split(fraud_sub_idx)
    n_tr, n_v, n_te = stratified_split(normal_sub_idx)

    train_idx = np.concatenate([f_tr, n_tr])
    val_idx   = np.concatenate([f_v,  n_v])
    test_idx  = np.concatenate([f_te, n_te])

    train_mask = torch.zeros(n_sub, dtype=torch.bool)
    val_mask   = torch.zeros(n_sub, dtype=torch.bool)
    test_mask  = torch.zeros(n_sub, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    from torch_geometric.data import Data
    sub_data = Data(
        x=x_sub,
        edge_index=edge_index_sub,
        edge_attr=edge_attr_sub,
        y=y_sub,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    sub_data.num_nodes = n_sub
    return sub_data


# ── Class weight computation ──────────────────────────────────────────────────

def compute_class_weights(y_train: torch.Tensor) -> torch.Tensor:
    n_total  = len(y_train)
    n_fraud  = y_train.sum().item()
    n_normal = n_total - n_fraud
    w_normal = n_total / (2.0 * n_normal) if n_normal > 0 else 1.0
    w_fraud  = n_total / (2.0 * n_fraud)  if n_fraud  > 0 else 1.0
    return torch.tensor([w_normal, w_fraud], dtype=torch.float32)


# ── Training & evaluation ─────────────────────────────────────────────────────

def train_one_epoch(model, sub_data, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    out  = model(sub_data.x.to(device), sub_data.edge_index.to(device))
    mask = sub_data.train_mask.to(device)
    loss = criterion(out[mask], sub_data.y[mask].to(device))
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, sub_data, mask_name: str, criterion, device):
    model.eval()
    out  = model(sub_data.x.to(device), sub_data.edge_index.to(device))
    mask = getattr(sub_data, mask_name).to(device)
    y    = sub_data.y[getattr(sub_data, mask_name)]

    loss  = criterion(out[mask], y.to(device)).item()
    preds = out[mask].argmax(dim=1).cpu().numpy()
    labels = y.cpu().numpy()

    return {
        "loss":      loss,
        "f1":        f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
    }


# ── Main local training function ──────────────────────────────────────────────

def local_train(
    client_name: str,
    graph,
    config: dict = None,
    global_weights: dict = None,
    device: torch.device = None,
) -> tuple:
    """
    Train the GCN locally on one client's graph.

    Args:
        client_name   : identifier, e.g. "client1_bankA"
        graph         : PyG Data object (client's local graph)
        config        : hyperparameter dict
        global_weights: state_dict from federated server (None = fresh init)
        device        : torch device

    Returns:
        (state_dict, val_metrics, model)
        state_dict  -- trained weights to send to server (no raw data)
        val_metrics -- dict with loss, f1, precision, recall
        model       -- trained FraudGCN instance
    """
    if config is None:
        config = DEFAULT_CONFIG
    if device is None:
        device = torch.device("cpu")

    # ── 1. Sample balanced subgraph ───────────────────────────────────────
    sub = sample_balanced_subgraph(
        graph,
        normal_multiplier=config["normal_multiplier"],
        seed=config["random_seed"],
    )

    n_fraud  = sub.y.sum().item()
    n_total  = sub.num_nodes
    n_train  = sub.train_mask.sum().item()
    print(f"  [{client_name}] Subgraph: {n_total:,} nodes "
          f"(fraud={int(n_fraud):,}, normal={n_total - int(n_fraud):,}) | "
          f"Edges: {sub.edge_index.shape[1]:,} | Train nodes: {n_train:,}")

    # ── 2. Model ──────────────────────────────────────────────────────────
    model = FraudGCN(
        in_channels=sub.x.shape[1],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    ).to(device)

    if global_weights is not None:
        model.load_state_dict(global_weights)

    # ── 3. Weighted loss ──────────────────────────────────────────────────
    class_weights = compute_class_weights(sub.y[sub.train_mask]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ── 4. Optimizer ──────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # ── 5. Training loop ──────────────────────────────────────────────────
    print(f"  [{client_name}] Training for {config['epochs']} epoch(s) ...")
    for epoch in range(1, config["epochs"] + 1):
        loss = train_one_epoch(model, sub, optimizer, criterion, device)
        if epoch % 2 == 0 or epoch == config["epochs"]:
            print(f"    Epoch {epoch:>2}/{config['epochs']} | Loss: {loss:.4f}")

    # ── 6. Validation ─────────────────────────────────────────────────────
    val_metrics = evaluate(model, sub, "val_mask", criterion, device)
    print(f"  [{client_name}] Val -> Loss: {val_metrics['loss']:.4f} | "
          f"F1: {val_metrics['f1']:.4f} | "
          f"Prec: {val_metrics['precision']:.4f} | "
          f"Recall: {val_metrics['recall']:.4f}")

    # Return CPU state_dict (raw data stays local — privacy preserved)
    state_dict = {k: v.cpu() for k, v in copy.deepcopy(model.state_dict()).items()}
    return state_dict, val_metrics, model


# ── Phase 4 entry point ───────────────────────────────────────────────────────

def run_single_client_test(client_name: str = "client1_bankA"):
    """Phase 4: Train on a single client to verify the GCN pipeline end-to-end."""
    print("=" * 60)
    print(f"FedShield - Phase 4: GCN on Single Client ({client_name})")
    print("=" * 60)

    graph_path = os.path.join(
        "data", "processed", "graphs", f"{client_name}_graph.pt"
    )
    graph = torch.load(graph_path, weights_only=False)
    print(f"Graph loaded: {graph.num_nodes:,} nodes | "
          f"{graph.edge_index.shape[1]:,} edges | "
          f"Fraud nodes: {int(graph.y.sum()):,}")

    device = torch.device("cpu")
    state_dict, val_metrics, model = local_train(
        client_name=client_name,
        graph=graph,
        config=DEFAULT_CONFIG,
        global_weights=None,
        device=device,
    )

    print()
    print("  Final validation metrics:")
    for k, v in val_metrics.items():
        print(f"    {k:12s}: {v:.4f}")

    save_path = os.path.join("results", "logs", f"{client_name}_phase4_model.pt")
    torch.save(state_dict, save_path)
    print(f"\n  Model weights saved -> {save_path}")

    print()
    print("=" * 60)
    print("Phase 4 COMPLETE. GCN trains on single client successfully.")
    print("Proceed to Phase 5: Multi-client local training.")
    print("=" * 60)


if __name__ == "__main__":
    run_single_client_test()
