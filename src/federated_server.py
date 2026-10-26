"""
FedShield — Phase 6: Federated Server (FedAvg)
================================================
Central server that orchestrates federated learning across 3 simulated
banking clients using the Federated Averaging (FedAvg) algorithm.

FedAvg algorithm (McMahan et al., 2017):
  For each communication round t:
    1. Server broadcasts current global model weights to all clients.
    2. Each client trains locally for E epochs on its own graph.
    3. Each client returns its updated model weights (state_dict).
    4. Server computes a weighted average of all client updates:
         w_global = sum( (n_k / n_total) * w_k ) for k in clients
       where n_k = number of training nodes for client k.
    5. Server updates the global model with the averaged weights.
    6. Server evaluates the global model on a held-out test set.

Privacy guarantee:
  Only model weights (state_dicts) are transmitted.
  Raw transaction data never leaves any client.
"""

import os
import json
import copy
import pickle
import torch
import numpy as np
from collections import OrderedDict

from src.model import FraudGCN

# Paths
GRAPHS_DIR   = os.path.join("data", "processed", "graphs")
RESULTS_DIR  = os.path.join("results", "metrics")
LOGS_DIR     = os.path.join("results", "logs")
MODELS_DIR   = os.path.join("results", "logs", "federated_models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CLIENTS = ["client1_bankA", "client2_bankB", "client3_bankC"]


# ── FedAvg Aggregation ────────────────────────────────────────────────────────

def federated_average(client_updates: dict, client_weights: dict) -> OrderedDict:
    """
    Compute weighted average of client model parameters.

    Args:
        client_updates : {client_name: state_dict}
        client_weights : {client_name: float} — proportional weight per client
                         (based on number of local training samples)

    Returns:
        Averaged global state_dict (same keys as any client state_dict)
    """
    total_weight = sum(client_weights.values())
    global_state = OrderedDict()

    # Initialise with zeros matching param shapes
    ref_state = next(iter(client_updates.values()))
    for key in ref_state:
        global_state[key] = torch.zeros_like(ref_state[key], dtype=torch.float32)

    # Weighted sum
    for client_name, state_dict in client_updates.items():
        w = client_weights[client_name] / total_weight
        for key in state_dict:
            global_state[key] += w * state_dict[key].float()

    return global_state


# ── Federated Server ──────────────────────────────────────────────────────────

class FederatedServer:
    """
    Orchestrates multi-round federated training across simulated banking
    clients. Acts as the central aggregation server — never sees raw data.
    """

    def __init__(self, model_config: dict, client_config: dict, num_rounds: int = 5):
        """
        Args:
            model_config  : kwargs for FraudGCN (in_channels, hidden_dim, dropout)
            client_config : hyperparameter dict passed to local_train()
            num_rounds    : number of federated communication rounds
        """
        self.model_config  = model_config
        self.client_config = client_config
        self.num_rounds    = num_rounds
        self.device        = torch.device("cpu")

        # Initialise global model
        self.global_model  = FraudGCN(**model_config).to(self.device)
        self.global_weights = copy.deepcopy(self.global_model.state_dict())

        # History log
        self.round_logs = []

    def load_graphs(self) -> dict:
        """Load all client graphs from disk."""
        graphs = {}
        for client_name in CLIENTS:
            path = os.path.join(GRAPHS_DIR, f"{client_name}_graph.pt")
            graphs[client_name] = torch.load(path, weights_only=False)
        return graphs

    def run(self) -> list:
        """Execute all federated rounds and return the round log."""
        from src.client import local_train

        print("=" * 60)
        print(f"FedShield - Phase 6: Federated Learning ({self.num_rounds} rounds)")
        print("=" * 60)

        graphs = self.load_graphs()

        for round_num in range(1, self.num_rounds + 1):
            print(f"\n{'='*60}")
            print(f"  ROUND {round_num}/{self.num_rounds}")
            print(f"{'='*60}")

            client_updates = {}
            client_train_sizes = {}
            round_metrics = {}

            # ── Step 1: Each client trains locally ───────────────────────
            for client_name in CLIENTS:
                graph = graphs[client_name]
                state_dict, val_metrics, _ = local_train(
                    client_name=client_name,
                    graph=graph,
                    config=self.client_config,
                    global_weights=copy.deepcopy(self.global_weights),
                    device=self.device,
                )
                client_updates[client_name]      = state_dict
                client_train_sizes[client_name]  = int(graph.y.sum().item())
                round_metrics[client_name]        = val_metrics

            # ── Step 2: FedAvg aggregation ────────────────────────────────
            print(f"\n  [Server] Performing FedAvg aggregation ...")
            self.global_weights = federated_average(client_updates, client_train_sizes)
            self.global_model.load_state_dict(self.global_weights)
            print(f"  [Server] Global model updated.")

            # ── Step 3: Save global model for this round ──────────────────
            model_path = os.path.join(MODELS_DIR, f"global_model_round{round_num}.pt")
            torch.save(self.global_weights, model_path)
            print(f"  [Server] Saved -> {model_path}")

            # ── Step 4: Log round results ─────────────────────────────────
            log_entry = {
                "round": round_num,
                "client_metrics": round_metrics,
                "avg_f1":      np.mean([m["f1"]        for m in round_metrics.values()]),
                "avg_recall":  np.mean([m["recall"]    for m in round_metrics.values()]),
                "avg_precision": np.mean([m["precision"] for m in round_metrics.values()]),
            }
            self.round_logs.append(log_entry)

            # Print round summary
            print(f"\n  Round {round_num} Summary:")
            print(f"  {'Client':<22} {'F1':>6} {'Prec':>7} {'Recall':>8}")
            print("  " + "-" * 46)
            for cname, m in round_metrics.items():
                print(f"  {cname:<22} {m['f1']:>6.4f} {m['precision']:>7.4f} "
                      f"{m['recall']:>8.4f}")
            print(f"  {'--- Avg ---':<22} "
                  f"{log_entry['avg_f1']:>6.4f} "
                  f"{log_entry['avg_precision']:>7.4f} "
                  f"{log_entry['avg_recall']:>8.4f}")

        # ── Save final global model & logs ────────────────────────────────
        final_model_path = os.path.join(MODELS_DIR, "global_model_final.pt")
        torch.save(self.global_weights, final_model_path)

        logs_path = os.path.join(RESULTS_DIR, "federated_round_logs.json")
        with open(logs_path, "w") as f:
            json.dump(self.round_logs, f, indent=2)

        print()
        print("=" * 60)
        print("Phase 6 COMPLETE.")
        print(f"  Final global model -> {final_model_path}")
        print(f"  Round logs         -> {logs_path}")
        print("Proceed to Phase 7: Differential Privacy.")
        print("=" * 60)

        return self.round_logs


def run():
    from src.client import DEFAULT_CONFIG

    model_config = {
        "in_channels": 10,
        "hidden_dim":  64,
        "dropout":     0.3,
    }

    client_config = dict(DEFAULT_CONFIG)
    client_config["epochs"] = 5   # Fewer local epochs per round for federation

    server = FederatedServer(
        model_config=model_config,
        client_config=client_config,
        num_rounds=3,              # 3 communication rounds
    )
    server.run()


if __name__ == "__main__":
    run()
