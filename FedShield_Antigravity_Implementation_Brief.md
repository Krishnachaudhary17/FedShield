# FedShield — Antigravity Project & Implementation Brief

## 1. Project

**FedShield — Centralized Banking Data Management System for Fraud Detection and Prevention**

FedShield is a privacy-preserving federated fraud detection prototype. Multiple simulated financial institutions collaboratively train a fraud detection model without sharing their raw transaction data with a central server.

Core technologies:
- Federated Learning
- Graph Neural Network / GCN
- Differential Privacy
- PaySim transaction/fraud dataset

The Review 2 PPT already covers the problem, objectives, architecture, literature survey, technical stack, methodology and references. The current priority is the **actual backend implementation**, not the PPT or UI.

---

## 2. What We Are Building

Target pipeline:

```text
PaySim Dataset
      |
      v
Preprocessing
      |
      v
Partition into 3 simulated clients
      |
      +----------------+----------------+
      |                |                |
   Client 1         Client 2         Client 3
   Bank A           Bank B           Bank C
      |                |                |
 Local Graph       Local Graph      Local Graph
      |                |                |
 Local GCN         Local GCN        Local GCN
      |                |                |
 Model Update      Model Update     Model Update
      +----------------+----------------+
                       |
                       v
                Federated Server
                       |
                       v
                FedAvg Aggregation
                       |
                       v
             Differential Privacy
                       |
                       v
              Updated Global Model
                  /     |     \
                 v      v      v
             Client 1 Client 2 Client 3
```

**Critical privacy rule:** raw transaction/client data must never be sent to the server. Only model parameters/updates are communicated.

---

## 3. Problem Being Solved

Traditional centralized fraud detection requires institutions to pool sensitive transaction data, creating privacy and regulatory risks.

Fraud can also span multiple accounts/transactions, so relationships between entities matter. Graph-based modeling is therefore used to capture relational fraud patterns.

Plain Federated Learning avoids sharing raw data but model updates can still leak information. FedShield therefore combines federated learning, graph-based modeling and differential privacy.

---

## 4. Dataset

Use the **actual PaySim dataset**.

Implementation must:
1. Load the CSV using Pandas.
2. Inspect columns and data types.
3. Handle missing/invalid values.
4. Select useful transaction features.
5. Encode categorical fields where necessary.
6. Define the fraud target.
7. Scale/normalize numerical features where appropriate.
8. Create an appropriate train/test split without data leakage.

**Do not invent results.**

---

## 5. Create 3 Simulated Banking Clients

Simulate:

- Client 1 = Bank A
- Client 2 = Bank B
- Client 3 = Bank C

Partition the PaySim training data among them.

Requirements:
- Clients remain logically isolated.
- Client 1 must not access Client 2/3 local data.
- Client 2 must not access Client 1/3 local data.
- Client 3 must not access Client 1/2 local data.
- If practical, make the client distributions non-IID to better represent different institutions.

---

## 6. Build Transaction Graphs

The project uses graph-based modeling.

Conceptual representation:

- Accounts/entities = nodes
- Transactions = edges
- Transaction/account information = node or edge features

For every client:
1. Use only that client's local data.
2. Construct its local transaction graph.
3. Create node/edge features.
4. Create labels required for fraud detection.
5. Convert the graph into a PyTorch Geometric-compatible representation.

Document the exact graph construction assumptions in the code/README.

---

## 7. Build the GCN

Use **PyTorch Geometric**.

Start with a simple GCN rather than an unnecessarily complex architecture.

Conceptually:

```text
Input Features
      |
   GCN Layer
      |
  Activation
      |
   GCN Layer
      |
  Classifier
      |
Fraud / Non-Fraud
```

First make the GCN work on **one client** before attempting federated training.

---

## 8. Local Client Training

For every federated round:

1. Client receives the current global model.
2. Client trains locally using only its own graph/data.
3. Client produces updated model parameters.
4. Client sends the model update to the server.
5. Client does NOT send raw transaction records.

First verify that local training works independently for all three clients.

---

## 9. Federated Learning / FedAvg

Create a central federated server.

The server:
1. Receives model updates from the clients.
2. Performs Federated Averaging (FedAvg).
3. Creates a new global model.
4. Sends the updated global model back to **all three clients**.

Training loop:

```text
Global Model
    |
    +--> Client 1 --> Update 1
    |
    +--> Client 2 --> Update 2
    |
    +--> Client 3 --> Update 3
                     |
                  Server
                     |
                   FedAvg
                     |
              Global Model
                     |
             next round
```

Run multiple communication rounds.

---

## 10. Differential Privacy

Add Differential Privacy at the planned aggregation/update stage.

The PPT describes this component as **aggregation-level DP inspired by GAP**.

Important:
- Do NOT claim FedShield is exactly GAP.
- Say **"inspired by GAP"**.
- The purpose is to protect model information while maintaining useful fraud-detection performance.
- Make the privacy/noise setting configurable.

Possible implementation:
- Opacus where appropriate, OR
- a custom DP noise module.

The exact implementation should be technically defensible and documented.

---

## 11. Evaluation

Evaluate the actual global model using real predictions.

Calculate:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC if practical
- Confusion matrix

Because fraud data can be highly imbalanced, do not rely only on accuracy.

**Never put fake metrics into the dashboard or PPT.**

---

## 12. Privacy–Utility Experiment

Run the model with different privacy/noise settings.

Example:

```text
Experiment A -> No additional DP noise
Experiment B -> Lower noise
Experiment C -> Medium noise
Experiment D -> Higher noise
```

Compare model utility using:
- Precision
- Recall
- F1
- ROC-AUC if available

Also record the privacy parameter / epsilon where applicable.

Goal: demonstrate the **privacy–utility trade-off**.

---

## 13. Recommended Technology Stack

Use the stack already defined in the PPT:

- Python 3.9+
- Pandas
- NumPy
- scikit-learn
- PyTorch
- PyTorch Geometric
- Flower OR custom FedAvg simulation
- Opacus OR custom Differential Privacy module

Do not introduce unnecessary technologies unless there is a clear reason.

---

## 14. Implementation Order

### Phase 1 — Dataset
1. Load PaySim.
2. Inspect dataset.
3. Preprocess.
4. Create train/test split.

### Phase 2 — Clients
5. Partition training data into 3 clients.
6. Verify isolation.

### Phase 3 — Graph
7. Build each client's transaction graph.
8. Verify nodes, edges, features and labels.

### Phase 4 — GCN
9. Implement simple GCN.
10. Train on one client.
11. Verify predictions and metrics.

### Phase 5 — Multi-client
12. Train locally on all 3 clients.
13. Generate model updates.

### Phase 6 — Federated Learning
14. Implement central server.
15. Receive client updates.
16. Implement FedAvg.
17. Return updated global model to all clients.
18. Run multiple rounds.

### Phase 7 — Differential Privacy
19. Add DP/noise.
20. Make privacy settings configurable.
21. Verify that training still works.

### Phase 8 — Evaluation
22. Evaluate global model.
23. Generate actual metrics.
24. Run privacy–utility experiments.

### Phase 9 — Final Integration
25. Save logs/results.
26. Connect real metrics to UI later.
27. Prepare final tables/graphs.

---

## 15. Immediate Target

Do NOT try to build everything at once.

The first meaningful end-to-end milestone is:

```text
PaySim
 -> preprocessing
 -> 3 clients
 -> local graphs
 -> GCN
 -> local training
 -> model updates
 -> FedAvg
 -> updated global model
```

Once this works, add Differential Privacy.

**This is much more important right now than polishing the UI.**

---

## 16. Suggested Project Structure

```text
FedShield/
|
|-- data/
|   |-- raw/
|   |-- processed/
|
|-- src/
|   |-- data_preprocessing.py
|   |-- client_partition.py
|   |-- graph_builder.py
|   |-- model.py
|   |-- client.py
|   |-- federated_server.py
|   |-- differential_privacy.py
|   |-- train.py
|   |-- evaluate.py
|
|-- results/
|   |-- metrics/
|   |-- logs/
|
|-- requirements.txt
|-- README.md
|-- main.py
```

The structure may be changed if Antigravity has a better engineering reason, but keep the implementation modular.

---

## 17. Development Rules for Antigravity

1. Build a **real runnable backend**, not a visual simulation.
2. Use the real PaySim dataset.
3. Never invent experimental results.
4. Never send raw client transaction data to the server.
5. Keep the three clients logically isolated.
6. Only model updates/parameters should be communicated to the server.
7. The updated global model must return to all three clients.
8. Keep the GCN simple at first.
9. Test every phase before moving to the next.
10. Save experiment configuration and results.
11. Clearly document assumptions about graph construction.
12. Clearly distinguish simulated banks from real banks.
13. Describe the DP component as **inspired by GAP**, not an exact GAP reproduction.
14. Do not connect the UI to fake/random metrics.
15. Only show metrics produced by actual model runs.

---

## 18. Definition of Done

The core implementation is functionally complete when:

- [ ] PaySim loads successfully
- [ ] Preprocessing works
- [ ] Three isolated clients are created
- [ ] Each client has a transaction graph
- [ ] GCN trains locally
- [ ] All clients produce model updates
- [ ] Central server performs FedAvg
- [ ] Updated global model is returned to all clients
- [ ] Multiple federated rounds execute
- [ ] Differential Privacy is integrated
- [ ] Global model produces fraud predictions
- [ ] Actual evaluation metrics are generated
- [ ] Privacy–utility experiments can run
- [ ] Results are saved for later UI/PPT integration

---

## 19. Antigravity's Immediate Task

Start with the simplest working implementation.

### Step 1
Make PaySim load and preprocess correctly.

### Step 2
Create three simulated clients.

### Step 3
Build a graph for each client.

### Step 4
Build and test a simple GCN on one client.

### Step 5
Extend local GCN training to all three clients.

### Step 6
Implement FedAvg and multiple communication rounds.

### Step 7
Add Differential Privacy.

### Step 8
Evaluate and save real results.

Do not skip ahead to UI integration.

If a design choice is ambiguous, choose the simplest technically defensible solution, document the assumption, and keep it modular.

The immediate goal is a **working PaySim → 3 Clients → Graphs → GCN → Local Training → FedAvg → Global Model pipeline**.
