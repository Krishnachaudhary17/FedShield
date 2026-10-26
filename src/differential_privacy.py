"""
FedShield — Phase 7: Differential Privacy
==========================================
Implements aggregation-level Differential Privacy (DP) at the federated
server, inspired by the GAP (Graph Aggregation Privacy) mechanism.

IMPORTANT DISCLAIMER (as required by the project brief):
  This implementation is INSPIRED BY GAP (Sajadmanesh & Gatica-Perez, 2021)
  but is NOT an exact reproduction. GAP applies DP to individual GNN
  aggregation steps in a local DP setting. Here we apply DP at the
  SERVER-SIDE FedAvg aggregation stage (a central/global DP setting),
  which protects the aggregated model updates from membership inference
  and model inversion attacks.

Mechanism:
  After FedAvg produces the averaged global weights w_global, we add
  calibrated Gaussian noise to each parameter tensor:
      w_dp = w_global + N(0, sigma^2 * sensitivity^2)
  where:
    - sigma (noise_multiplier) is configurable.
    - sensitivity is estimated as the L2 norm of the parameter update
      (clipped to clip_norm to bound sensitivity).
  Clipping before noise ensures the noise is properly calibrated.

Privacy accounting:
  We track the empirical noise multiplier sigma applied per round.
  Full (epsilon, delta)-DP accounting requires Renyi-DP composition
  (e.g. via Opacus or autodp), which depends on the number of rounds,
  the sampling ratio, and sigma. For this prototype we report sigma
  as a proxy for the privacy level (lower sigma = less private but
  more accurate; higher sigma = more private but lower accuracy).

Configurable settings:
  - noise_multiplier (sigma): controls noise magnitude.
  - clip_norm: maximum L2 norm for parameter clipping.
  - delta: target delta for (epsilon, delta)-DP (default 1e-5).
"""

import math
import copy
import torch
import numpy as np
from collections import OrderedDict


# ── Clipping ──────────────────────────────────────────────────────────────────

def clip_weights(state_dict: OrderedDict, clip_norm: float) -> OrderedDict:
    """
    Clip the global L2 norm of all parameters to clip_norm.
    This bounds the sensitivity of the update to clip_norm,
    ensuring the noise addition is properly calibrated.
    """
    clipped = copy.deepcopy(state_dict)

    # Compute global L2 norm across all parameter tensors
    total_norm_sq = sum(
        p.float().norm(2).item() ** 2
        for p in clipped.values()
        if torch.is_tensor(p)
    )
    total_norm = math.sqrt(total_norm_sq)

    if total_norm > clip_norm:
        scale = clip_norm / total_norm
        for key in clipped:
            if torch.is_tensor(clipped[key]):
                clipped[key] = clipped[key].float() * scale

    return clipped, min(total_norm, clip_norm)


# ── Gaussian Noise Injection ──────────────────────────────────────────────────

def add_gaussian_noise(
    state_dict: OrderedDict,
    noise_multiplier: float,
    clip_norm: float,
) -> OrderedDict:
    """
    Add Gaussian noise calibrated to noise_multiplier * clip_norm
    to every parameter tensor in the state_dict.

    Noise std = noise_multiplier * clip_norm (standard Gaussian mechanism).
    """
    noisy = copy.deepcopy(state_dict)
    noise_std = noise_multiplier * clip_norm

    for key in noisy:
        if torch.is_tensor(noisy[key]) and noisy[key].is_floating_point():
            noise = torch.randn_like(noisy[key].float()) * noise_std
            noisy[key] = noisy[key].float() + noise

    return noisy


# ── Main DP application ───────────────────────────────────────────────────────

def apply_dp_to_aggregation(
    global_weights: OrderedDict,
    noise_multiplier: float,
    clip_norm: float = 1.0,
) -> tuple:
    """
    Apply DP to the globally aggregated model weights:
      1. Clip global weight vector norm to clip_norm.
      2. Add Gaussian noise scaled by noise_multiplier * clip_norm.

    Args:
        global_weights   : FedAvg-aggregated model state_dict
        noise_multiplier : sigma — controls privacy/utility trade-off
                           0.0 = no noise (no DP)
                           0.1 = low noise (less private, more accurate)
                           0.5 = medium noise
                           1.0 = high noise (more private, less accurate)
        clip_norm        : L2 norm clipping threshold (default 1.0)

    Returns:
        (dp_weights, dp_info)
        dp_weights : noisy state_dict ready to broadcast to clients
        dp_info    : dict with noise_multiplier, clip_norm, actual_norm
    """
    if noise_multiplier == 0.0:
        return copy.deepcopy(global_weights), {
            "noise_multiplier": 0.0,
            "clip_norm": clip_norm,
            "actual_norm": 0.0,
            "noise_std": 0.0,
            "dp_applied": False,
        }

    clipped_weights, actual_norm = clip_weights(global_weights, clip_norm)
    dp_weights = add_gaussian_noise(clipped_weights, noise_multiplier, clip_norm)

    dp_info = {
        "noise_multiplier": noise_multiplier,
        "clip_norm": clip_norm,
        "actual_norm": round(actual_norm, 6),
        "noise_std": round(noise_multiplier * clip_norm, 6),
        "dp_applied": True,
    }
    return dp_weights, dp_info


# ── Privacy accounting (approximate) ─────────────────────────────────────────

def estimate_epsilon(
    noise_multiplier: float,
    num_rounds: int,
    delta: float = 1e-5,
) -> float:
    """
    Approximate epsilon using the simple strong composition theorem:
        epsilon ~ sqrt(2 * T * ln(1/delta)) / sigma
    where T = num_rounds, sigma = noise_multiplier.

    NOTE: This is a rough upper bound. For tighter accounting use
    Renyi-DP composition (e.g., Opacus privacy engine).
    Returns float('inf') if noise_multiplier == 0.
    """
    if noise_multiplier == 0.0:
        return float("inf")
    epsilon = math.sqrt(2 * num_rounds * math.log(1.0 / delta)) / noise_multiplier
    return round(epsilon, 4)


# ── DP-enabled Federated Server ───────────────────────────────────────────────

class DPFederatedServer:
    """
    Extends FedAvg with aggregation-level Differential Privacy.
    Broadcasts DP-noised global model to clients each round.
    """

    def __init__(
        self,
        model_config: dict,
        client_config: dict,
        num_rounds: int = 3,
        noise_multiplier: float = 0.1,
        clip_norm: float = 1.0,
        delta: float = 1e-5,
        experiment_label: str = "dp_run",
    ):
        from src.model import FraudGCN
        import os

        self.model_config     = model_config
        self.client_config    = client_config
        self.num_rounds       = num_rounds
        self.noise_multiplier = noise_multiplier
        self.clip_norm        = clip_norm
        self.delta            = delta
        self.label            = experiment_label
        self.device           = torch.device("cpu")

        self.global_model   = FraudGCN(**model_config).to(self.device)
        self.global_weights = copy.deepcopy(self.global_model.state_dict())

        self.round_logs = []

        self.models_dir = os.path.join("results", "logs", "federated_models")
        self.results_dir = os.path.join("results", "metrics")
        os.makedirs(self.models_dir, exist_ok=True)

        estimated_eps = estimate_epsilon(noise_multiplier, num_rounds, delta)
        print(f"  DP Config: sigma={noise_multiplier} | "
              f"clip_norm={clip_norm} | "
              f"approx epsilon={estimated_eps} | delta={delta}")

    def run(self) -> list:
        import os, json
        import numpy as np
        from src.federated_server import federated_average, CLIENTS, GRAPHS_DIR
        from src.client import local_train
        import torch

        graphs = {}
        for client_name in CLIENTS:
            path = os.path.join(GRAPHS_DIR, f"{client_name}_graph.pt")
            graphs[client_name] = torch.load(path, weights_only=False)

        for round_num in range(1, self.num_rounds + 1):
            print(f"\n  --- Round {round_num}/{self.num_rounds} "
                  f"[sigma={self.noise_multiplier}] ---")

            client_updates = {}
            client_train_sizes = {}
            round_metrics = {}

            # Clients receive the DP-noised global model from last round
            for client_name in CLIENTS:
                graph = graphs[client_name]
                state_dict, val_metrics, _ = local_train(
                    client_name=client_name,
                    graph=graph,
                    config=self.client_config,
                    global_weights=copy.deepcopy(self.global_weights),
                    device=self.device,
                )
                client_updates[client_name]     = state_dict
                client_train_sizes[client_name] = int(graph.y.sum().item())
                round_metrics[client_name]       = val_metrics

            # FedAvg
            aggregated = federated_average(client_updates, client_train_sizes)

            # Apply DP noise at aggregation
            dp_weights, dp_info = apply_dp_to_aggregation(
                aggregated,
                noise_multiplier=self.noise_multiplier,
                clip_norm=self.clip_norm,
            )
            self.global_weights = dp_weights
            self.global_model.load_state_dict(self.global_weights)

            log_entry = {
                "round":          round_num,
                "experiment":     self.label,
                "noise_multiplier": self.noise_multiplier,
                "dp_info":        dp_info,
                "client_metrics": round_metrics,
                "avg_f1":         np.mean([m["f1"]        for m in round_metrics.values()]),
                "avg_recall":     np.mean([m["recall"]    for m in round_metrics.values()]),
                "avg_precision":  np.mean([m["precision"] for m in round_metrics.values()]),
            }
            self.round_logs.append(log_entry)

            print(f"    Avg F1={log_entry['avg_f1']:.4f} | "
                  f"Prec={log_entry['avg_precision']:.4f} | "
                  f"Recall={log_entry['avg_recall']:.4f} | "
                  f"noise_std={dp_info['noise_std']}")

        return self.round_logs


def run():
    """
    Phase 7: Run one DP-enabled federated training run (medium noise)
    to verify the DP mechanism integrates correctly with FedAvg.
    Full privacy-utility experiments are in Phase 8.
    """
    import json, os
    from src.client import DEFAULT_CONFIG

    print("=" * 60)
    print("FedShield - Phase 7: Differential Privacy Integration")
    print("=" * 60)

    model_config = {"in_channels": 10, "hidden_dim": 64, "dropout": 0.3}
    client_config = dict(DEFAULT_CONFIG)
    client_config["epochs"] = 5

    # Test with medium noise to verify DP pipeline
    server = DPFederatedServer(
        model_config=model_config,
        client_config=client_config,
        num_rounds=3,
        noise_multiplier=0.5,    # medium noise
        clip_norm=1.0,
        delta=1e-5,
        experiment_label="medium_noise_sigma0.5",
    )
    logs = server.run()

    # Save DP logs
    dp_logs_path = os.path.join("results", "metrics", "dp_integration_test.json")
    with open(dp_logs_path, "w") as f:
        json.dump(logs, f, indent=2)

    final_avg_f1 = logs[-1]["avg_f1"]
    eps = estimate_epsilon(0.5, 3, 1e-5)
    print()
    print("=" * 60)
    print("Phase 7 COMPLETE. DP integrated with FedAvg.")
    print(f"  Final Avg F1        : {final_avg_f1:.4f}")
    print(f"  Noise multiplier    : sigma=0.5")
    print(f"  Approx epsilon      : {eps} (delta=1e-5)")
    print(f"  DP logs saved       -> {dp_logs_path}")
    print("Proceed to Phase 8: Evaluation & Privacy-Utility Experiments.")
    print("=" * 60)


if __name__ == "__main__":
    run()
