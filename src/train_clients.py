"""
FedShield — Phase 5: Multi-Client Local Training
==================================================
Runs independent local GCN training on all 3 simulated banking clients
(Bank A, Bank B, Bank C) and saves each client's model update (state_dict)
for the federated server in Phase 6.

Privacy guarantee:
  Each client trains strictly on its own local graph. No client accesses
  another client's raw transactions at any point. Only the model weight
  dictionaries (state_dicts) are written to disk — no raw graph data.
"""

import os
import pickle
import torch
from src.client import local_train, DEFAULT_CONFIG

# Paths
GRAPHS_DIR = os.path.join("data", "processed", "graphs")
UPDATES_DIR = os.path.join("results", "logs", "client_updates")
os.makedirs(UPDATES_DIR, exist_ok=True)

# All 3 clients
CLIENTS = ["client1_bankA", "client2_bankB", "client3_bankC"]


def run():
    print("=" * 60)
    print("FedShield - Phase 5: Multi-Client Local Training")
    print("=" * 60)
    print()

    all_updates = {}   # {client_name: state_dict}
    all_metrics = {}   # {client_name: val_metrics}

    for client_name in CLIENTS:
        print(f"[{client_name}]")
        graph_path = os.path.join(GRAPHS_DIR, f"{client_name}_graph.pt")
        graph = torch.load(graph_path, weights_only=False)

        print(f"  Graph: {graph.num_nodes:,} nodes | "
              f"{graph.edge_index.shape[1]:,} edges | "
              f"Fraud nodes: {int(graph.y.sum()):,}")

        state_dict, val_metrics, _ = local_train(
            client_name=client_name,
            graph=graph,
            config=DEFAULT_CONFIG,
            global_weights=None,    # Round 0: fresh init
            device=torch.device("cpu"),
        )

        all_updates[client_name] = state_dict
        all_metrics[client_name] = val_metrics

        # Save individual update
        update_path = os.path.join(UPDATES_DIR, f"{client_name}_update.pt")
        torch.save(state_dict, update_path)
        print(f"  Update saved -> {update_path}")
        print()

    # Save all updates together for federated server
    all_updates_path = os.path.join(UPDATES_DIR, "all_client_updates.pkl")
    with open(all_updates_path, "wb") as f:
        pickle.dump(all_updates, f)

    # Print summary
    print("=" * 60)
    print("Phase 5 Summary — Validation Metrics per Client")
    print("=" * 60)
    print(f"  {'Client':<22} {'F1':>6} {'Prec':>7} {'Recall':>8}")
    print("  " + "-" * 45)
    for cname, m in all_metrics.items():
        print(f"  {cname:<22} {m['f1']:>6.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>8.4f}")

    print()
    print(f"  All client updates saved -> {all_updates_path}")
    print()
    print("=" * 60)
    print("Phase 5 COMPLETE. All 3 clients trained independently.")
    print("Model updates ready for Phase 6: Federated Averaging (FedAvg).")
    print("=" * 60)


if __name__ == "__main__":
    run()
