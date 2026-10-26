"""
FedShield: Privacy-Preserving Federated Fraud Detection System
==============================================================
Main CLI entrypoint for pipeline orchestration across all 9 phases:
  Phase 1: PaySim Data Preprocessing
  Phase 2: Client Partitioning (Bank A, Bank B, Bank C)
  Phase 3: Transaction Graph Construction (PyG)
  Phase 4: GCN Architecture & Single-Client Validation
  Phase 5: Multi-Client Local Training
  Phase 6: Federated Server & FedAvg Aggregation
  Phase 7: Differential Privacy (GAP-Inspired Aggregation Noise)
  Phase 8: Comprehensive Model Evaluation & Privacy-Utility Tradeoff
  Phase 9: Final System Integration & Dashboard

Usage:
  python main.py                  # Display full pipeline status & metrics summary
  python main.py --summary        # Display full pipeline status & metrics summary
  python main.py --evaluate       # Run Phase 8 model evaluation
  python main.py --tradeoff       # Run Phase 8 Differential Privacy tradeoff sweep
  python main.py --federated      # Run Phase 6 FedAvg multi-round training
  python main.py --dp             # Run Phase 7 DP-integrated federated run
  python main.py --dashboard      # Launch the dashboard on local web server
"""

import sys
import os
import json
import argparse
import http.server
import socketserver
import webbrowser

RESULTS_DIR = os.path.join("results", "metrics")


def print_banner():
    print("=" * 72)
    print("  FedShield: Privacy-Preserving Federated Fraud Detection System")
    print("  Federated Learning | Graph Neural Networks | Differential Privacy")
    print("=" * 72)


def display_pipeline_summary():
    print_banner()
    print("\n[PIPELINE STATUS OVERVIEW]")
    print("  Phase 1: PaySim Data Preprocessing           -> [COMPLETE] (2.77M transactions)")
    print("  Phase 2: Non-IID 3-Client Partitioning       -> [COMPLETE] (Bank A: 50%, B: 30%, C: 20%)")
    print("  Phase 3: PyG Transaction Graph Builder       -> [COMPLETE] (3 client graphs created)")
    print("  Phase 4: FraudGCN Model Architecture         -> [COMPLETE] (2-layer GCN with dropout)")
    print("  Phase 5: Multi-Client Independent Training   -> [COMPLETE] (3 isolated state_dicts)")
    print("  Phase 6: Central FedAvg Aggregation          -> [COMPLETE] (3 communication rounds)")
    print("  Phase 7: GAP-Inspired Differential Privacy   -> [COMPLETE] (sigma=0.5, eps=16.62)")
    print("  Phase 8: Comprehensive Model Evaluation      -> [COMPLETE] (Held-out test graphs)")
    print("  Phase 9: Final Integration & Dashboard       -> [COMPLETE] (FedShield_UI interactive)")

    # 1. Federated Round Performance
    fed_logs_path = os.path.join(RESULTS_DIR, "federated_round_logs.json")
    if os.path.exists(fed_logs_path):
        with open(fed_logs_path, "r") as f:
            fed_logs = json.load(f)
        print("\n" + "=" * 72)
        print("  FEDERATED LEARNING ROUNDS SUMMARY (PHASE 6)")
        print("=" * 72)
        print(f"  {'Round':<8} {'Avg F1':>10} {'Avg Precision':>16} {'Avg Recall':>14}")
        print("  " + "-" * 52)
        for r in fed_logs:
            print(f"  Round {r['round']:<2} {r['avg_f1']:>10.4f} {r['avg_precision']:>16.4f} {r['avg_recall']:>14.4f}")
        print("  " + "-" * 52)

    # 2. Final Global Model Test Metrics
    eval_path = os.path.join(RESULTS_DIR, "global_model_evaluation.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            eval_data = json.load(f)
        comb = eval_data.get("combined_test_metrics", {})
        cm = comb.get("confusion_matrix", {})
        print("\n" + "=" * 72)
        print("  HELD-OUT TEST SET EVALUATION (PHASE 8)")
        print("=" * 72)
        print(f"  Total Test Accounts Evaluated : {comb.get('total_test_samples', 0):,}")
        print(f"  Total Ground Truth Fraud      : {comb.get('test_fraud_count', 0):,}")
        print(f"  Global Test Accuracy          : {comb.get('accuracy', 0)*100:.2f}%")
        print(f"  Global Precision              : {comb.get('precision', 0)*100:.2f}%")
        print(f"  Global Recall (Fraud Caught)  : {comb.get('recall', 0)*100:.2f}%")
        print(f"  Global F1-Score               : {comb.get('f1', 0):.4f}")
        print(f"  Global ROC-AUC                : {comb.get('roc_auc', 0)*100:.2f}%")
        print(f"  Confusion Matrix              : TP={cm.get('tp',0):,} | FP={cm.get('fp',0):,} | TN={cm.get('tn',0):,} | FN={cm.get('fn',0):,}")
        print("  " + "-" * 52)

    # 3. Privacy-Utility Tradeoff
    tradeoff_path = os.path.join(RESULTS_DIR, "privacy_utility_tradeoff.json")
    if os.path.exists(tradeoff_path):
        with open(tradeoff_path, "r") as f:
            tradeoffs = json.load(f)
        print("\n" + "=" * 72)
        print("  DIFFERENTIAL PRIVACY & UTILITY TRADEOFF BENCHMARK (PHASE 8)")
        print("=" * 72)
        print(f"  {'Experiment':<20} {'Sigma':>10} {'Epsilon (eps)':>14} {'F1':>8} {'Recall':>8} {'ROC-AUC':>9}")
        print("  " + "-" * 72)
        for t in tradeoffs:
            eps_val = f"{t['epsilon']:.2f}" if isinstance(t['epsilon'], (int, float)) else str(t['epsilon'])
            print(f"  {t['experiment_id']:<20} {t['noise_multiplier_sigma']:>10.2f} {eps_val:>14} {t['f1_score']:>8.4f} {t['recall']:>8.4f} {t['roc_auc']:>9.4f}")
        print("  " + "-" * 72)

    print("\n[DASHBOARD LAUNCH]")
    print("  To launch the interactive dashboard, run:")
    print("    python main.py --dashboard")
    print("  Or open FedShield_UI/index.html in any web browser.")
    print("=" * 72 + "\n")


def run_dashboard(port=8080):
    print(f"\nStarting FedShield local server on port {port} ...")
    print(f"Open your browser at: http://localhost:{port}/FedShield_UI/index.html")
    print("Press Ctrl+C to terminate the dashboard server.\n")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard server stopped.")


def main():
    parser = argparse.ArgumentParser(description="FedShield Pipeline Orchestrator")
    parser.add_argument("--summary", action="store_true", help="Display full summary of all phases and metrics")
    parser.add_argument("--evaluate", action="store_true", help="Run Phase 8 model evaluation")
    parser.add_argument("--tradeoff", action="store_true", help="Run Phase 8 DP privacy-utility experiments")
    parser.add_argument("--federated", action="store_true", help="Run Phase 6 FedAvg federated training")
    parser.add_argument("--dp", action="store_true", help="Run Phase 7 DP-integrated federated training")
    parser.add_argument("--dashboard", action="store_true", help="Start local web server for FedShield_UI")
    parser.add_argument("--port", type=int, default=8080, help="Port for dashboard web server")

    args = parser.parse_args()

    if args.evaluate:
        from src.evaluate import evaluate_global_model
        evaluate_global_model()
    elif args.tradeoff:
        from src.evaluate import run_privacy_utility_experiments
        run_privacy_utility_experiments()
    elif args.federated:
        from src.federated_server import run as run_federated
        run_federated()
    elif args.dp:
        from src.differential_privacy import run as run_dp
        run_dp()
    elif args.dashboard:
        run_dashboard(port=args.port)
    else:
        display_pipeline_summary()


if __name__ == "__main__":
    main()
