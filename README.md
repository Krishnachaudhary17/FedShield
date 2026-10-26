# FedShield: Privacy-Preserving Federated Fraud Detection System

**A Centralized Banking Data Management System for Fraud Detection and Prevention**  
Combining **Federated Learning (FedAvg)**, **Graph Convolutional Networks (GCN)**, and **Differential Privacy (GAP-inspired)** on the real-world **PaySim** transaction dataset.

---

## 📌 Executive Summary

Traditional centralized fraud detection mandates that financial institutions aggregate sensitive transaction and customer records into a central warehouse, exposing customer privacy and violating strict financial regulations (GDPR, PCI-DSS).

**FedShield** resolves this challenge through a privacy-preserving collaborative learning framework:
1. **Zero Raw Data Transmission:** Client banking institutions train locally on their private transaction graphs. Raw transaction records, account balances, and entity graphs **never leave the bank**.
2. **Relational Fraud Detection (GCN):** Fraudulent transactions naturally form complex networks and money laundering rings. FedShield models accounts as nodes and transactions as edges, using a 2-layer Graph Convolutional Network to capture structural topology.
3. **Federated Averaging (FedAvg):** A central federated server coordinates multi-round learning by aggregating client parameter updates weighted by institutional training volume.
4. **Differential Privacy (GAP-Inspired):** Model sensitivity is bounded via global L2 norm clipping, followed by calibrated Gaussian noise injection at the aggregation phase to guard against model inversion and membership inference attacks.
5. **100% Real Empirical Validation:** Built entirely on actual PaySim transaction data. Zero simulated or fabricated metrics.

---

## 🏛 Architecture & Data Flow

```text
                           PaySim Dataset (2.77M rows)
                                      |
                            Phase 1: Preprocessing
                          (Filtering, Scaling, Splits)
                                      |
                      Phase 2: Non-IID Client Partitions
                     /                |                \
                    v                 v                 v
             Client 1 (Bank A)  Client 2 (Bank B)  Client 3 (Bank C)
               [50% volume]       [30% volume]       [20% volume]
                    |                 |                 |
          Phase 3: Local Graph   Local Graph       Local Graph
           (1.79M nodes, PyG) (1.16M nodes, PyG) (831k nodes, PyG)
                    |                 |                 |
          Phase 4/5: Local GCN   Local GCN         Local GCN
           (Subgraph training) (Subgraph training) (Subgraph training)
                    |                 |                 |
              Model Update       Model Update      Model Update
             (state_dict 1)     (state_dict 2)    (state_dict 3)
                     \                |                /
                      +---------------+---------------+
                                      |
                                      v
                        Phase 6: Federated Server
                                      |
                             FedAvg Aggregation
                     w_global = sum( (n_k / n_tot) * w_k )
                                      |
                                      v
                        Phase 7: Differential Privacy
                       - L2 Norm Clipping (clip_norm=1.0)
                       - Gaussian Noise Addition (sigma=0.5)
                                      |
                                      v
                        Updated Global Model Broadcast
                     /                |                \
                    v                 v                 v
             Client 1 (Bank A)  Client 2 (Bank B)  Client 3 (Bank C)
```

---

## 📊 Empirical Results (Real Model Evaluations)

### 1. Client Partition & Graph Sizes (Phase 3)
| Institution | Raw Transactions | Graph Nodes (Accounts) | Graph Edges | Fraud Accounts | Fraud Rate |
|---|---|---|---|---|---|
| **Client 1 (Bank A)** | 1,385,189 (50%) | 1,798,099 | 1,385,189 | 4,034 | 0.224% |
| **Client 2 (Bank B)** | 831,143 (30%) | 1,167,885 | 831,143 | 2,514 | 0.215% |
| **Client 3 (Bank C)** | 554,077 (20%) | 831,667 | 554,077 | 1,665 | 0.200% |
| **Total** | **2,770,409** | **3,797,651** | **2,770,409** | **8,213** | **0.216%** |

### 2. Multi-Round Federated Learning Progression (Phase 6)
Aggregated over 3 communication rounds across all 3 banks:

| Federated Round | Average Loss | Average Precision | Average Recall | Average F1-Score |
|---|---|---|---|---|
| **Round 1** | 7,018.84 | 6.49% | **99.87%** | 0.1219 |
| **Round 2** | 1,549.33 | 20.38% | **99.25%** | 0.3380 |
| **Round 3** | **545.52** | **29.94%** | **99.25%** | **0.4599** |

*Key finding:* Loss dropped by **92.2%** from Round 1 to Round 3, while F1-score quadrupled from 0.12 to 0.46 through collaborative FedAvg learning without any raw data pooling.

### 3. Held-Out Test Set Performance (Phase 8)
Evaluated on **25,874** held-out test nodes across the banking institutions:

| Evaluation Split | Test Accounts | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **Client 1 (Bank A)** | 12,708 | 87.70% | 27.92% | **99.83%** | 0.4364 | 0.9348 |
| **Client 2 (Bank B)** | 7,920 | 88.72% | 29.71% | **99.74%** | 0.4578 | 0.9397 |
| **Client 3 (Bank C)** | 5,246 | 89.73% | 31.63% | **98.80%** | 0.4792 | 0.9419 |
| **COMBINED GLOBAL** | **25,874** | **88.42%** | **29.15%** | **99.60%** | **0.4510** | **0.9377** |

#### Confusion Matrix (Combined Test Set):
- **True Positives (Fraud Caught):** `1,230` (out of 1,235 fraud accounts — **99.6% recall**)
- **False Negatives (Missed Fraud):** `5`
- **True Negatives:** `21,649`
- **False Positives:** `2,990`

---

## 🛡 Differential Privacy & Utility Trade-Off (GAP-Inspired)

FedShield implements aggregation-level Differential Privacy inspired by the **GAP (Graph Aggregation Privacy)** mechanism. Sensitivity is bounded via parameter norm clipping ($L_2 \le 1.0$) followed by Gaussian perturbation:

$$w_{dp} = \text{clip}(w_{global}, C) + \mathcal{N}(0, \sigma^2 C^2 I)$$

Empirical sweep over 4 privacy regimes ($\delta = 10^{-5}$):

| Experiment | Noise Multiplier ($\sigma$) | Approx Privacy Bound ($\varepsilon$) | F1-Score | Precision | Recall | ROC-AUC | Privacy Level |
|---|---|---|---|---|---|---|---|
| **Exp A: Baseline** | 0.00 | $\infty$ (No DP) | 0.3557 | 21.72% | 98.22% | 0.9029 | None |
| **Exp B: Low Noise** | 0.10 | 83.11 | 0.0850 | 4.89% | 32.47% | 0.5054 | Weak |
| **Exp C: Medium Noise** | 0.50 | 16.62 | 0.1141 | 6.35% | 56.28% | 0.5731 | Balanced |
| **Exp D: High Noise** | 1.00 | 8.31 | 0.1336 | 7.18% | 96.03% | 0.6690 | Strong |

*Empirical Tradeoff Insight:* Higher noise ($\sigma = 1.0$) provides strong theoretical privacy ($\varepsilon = 8.31$) while the model relies on aggressive inductive bias to preserve high recall (96.03%).

---

## 💻 Installation & Usage

### 1. Requirements & Setup
```bash
# Clone the repository
git clone https://github.com/Krishnachaudhary17/SehatSaathi.git FedShield
cd FedShield

# Install core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric pandas numpy scikit-learn
```

### 2. Run the Orchestrator
```bash
# View complete pipeline summary & metrics
python main.py

# Run Phase 8 evaluation on the trained federated model
python main.py --evaluate

# Run Phase 8 Differential Privacy benchmark sweep
python main.py --tradeoff

# Launch the interactive web dashboard
python main.py --dashboard
```

### 3. Interactive Web Dashboard
You can also view the dashboard by directly opening `FedShield_UI/index.html` in any browser, or navigate to:
```
http://localhost:8080/FedShield_UI/index.html
```

---

## 📁 Repository Structure

```text
FedShield/
├── data/
│   ├── raw/                           # Raw PaySim dataset CSV
│   └── processed/
│       ├── clients/                   # Bank A, B, C partitioned CSVs
│       ├── graphs/                    # Client PyG transaction graphs (.pt)
│       ├── X_train.npy, X_test.npy   # Preprocessed numerical feature tensors
│       └── scaler.pkl, feature_cols.pkl
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py          # Phase 1: Cleaning, features, splits
│   ├── client_partition.py            # Phase 2: Non-IID 3-bank isolation
│   ├── graph_builder.py               # Phase 3: PyG account-transaction graphs
│   ├── model.py                       # Phase 4: FraudGCN 2-layer architecture
│   ├── client.py                      # Phase 5: Balanced subgraph local training
│   ├── train_clients.py               # Phase 5: Multi-client training runner
│   ├── federated_server.py            # Phase 6: Central FedAvg aggregation
│   ├── differential_privacy.py        # Phase 7: GAP-inspired DP noise injection
│   └── evaluate.py                    # Phase 8: Global evaluation & DP tradeoff
├── results/
│   ├── metrics/
│   │   ├── federated_round_logs.json  # Multi-round FedAvg history
│   │   ├── global_model_evaluation.json # Test split metrics (Acc, F1, ROC-AUC)
│   │   └── privacy_utility_tradeoff.json# DP noise benchmark results
│   └── logs/
│       ├── client_updates/            # Isolated state_dict parameter updates
│       └── federated_models/          # Global model checkpoints (Rounds 1-3)
├── FedShield_UI/                      # Modern interactive frontend
│   ├── index.html                     # Dashboard interface
│   ├── style.css                      # Modern dark/fintech styling
│   ├── real_data.js                   # Ground-truth metric data
│   └── script.js                      # Client simulation controller
├── requirements.txt                   # Dependency manifest
├── main.py                            # Unified CLI pipeline orchestrator
└── README.md                          # Comprehensive technical documentation
```

---

## ⚖️ Implementation Verification Checklist

- [x] **Phase 1: Dataset Loading & Preprocessing:** PaySim loaded, cleaned, engineered features, scaled, train/test split.
- [x] **Phase 2: Client Partitioning:** Non-IID distribution across 3 isolated simulated institutions (Bank A, B, C).
- [x] **Phase 3: Graph Construction:** PyTorch Geometric graph representations built with vectorized feature extraction.
- [x] **Phase 4: GCN Architecture:** 2-layer GCNConv with ReLU, dropout, and class-weighted loss.
- [x] **Phase 5: Multi-Client Training:** Independent local training; raw data never leaves client partitions.
- [x] **Phase 6: Federated Aggregation:** Central server with FedAvg weighting across multiple rounds.
- [x] **Phase 7: Differential Privacy:** Aggregation-level DP inspired by GAP with configurable $\sigma$ and norm clipping.
- [x] **Phase 8: Comprehensive Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, and 4-tier Privacy-Utility tradeoff.
- [x] **Phase 9: Final Integration:** CLI orchestrator (`main.py`), interactive UI dashboard connected to real model metrics, and documentation.
