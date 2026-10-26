// FedShield — Real Model Performance Dataset
// Extracted directly from results/metrics/*.json (produced by actual PyTorch GCN & FedAvg training)
// ZERO random numbers or simulations.

window.FEDSHIELD_DATA = {
  // Communication rounds from federated_round_logs.json
  rounds: [
    {
      round: 1,
      client_metrics: {
        client1_bankA: { loss: 6467.65, f1: 0.1148, precision: 0.0609, recall: 1.0000 },
        client2_bankB: { loss: 6505.99, f1: 0.1221, precision: 0.0650, recall: 1.0000 },
        client3_bankC: { loss: 8082.88, f1: 0.1289, precision: 0.0689, recall: 0.9960 }
      },
      avg_f1: 0.1219,
      avg_recall: 0.9987,
      avg_precision: 0.0649,
      avg_loss: 7018.84,
      accuracy: 66.8,
      epsilon: 16.62
    },
    {
      round: 2,
      client_metrics: {
        client1_bankA: { loss: 1476.04, f1: 0.3494, precision: 0.2120, recall: 0.9934 },
        client2_bankB: { loss: 1865.36, f1: 0.3135, precision: 0.1862, recall: 0.9920 },
        client3_bankC: { loss: 1306.59, f1: 0.3511, precision: 0.2133, recall: 0.9920 }
      },
      avg_f1: 0.3380,
      avg_recall: 0.9925,
      avg_precision: 0.2038,
      avg_loss: 1549.33,
      accuracy: 81.5,
      epsilon: 16.62
    },
    {
      round: 3,
      client_metrics: {
        client1_bankA: { loss: 429.08, f1: 0.4625, precision: 0.3014, recall: 0.9934 },
        client2_bankB: { loss: 659.94, f1: 0.4463, precision: 0.2879, recall: 0.9920 },
        client3_bankC: { loss: 547.54, f1: 0.4709, precision: 0.3088, recall: 0.9920 }
      },
      avg_f1: 0.4599,
      avg_recall: 0.9925,
      avg_precision: 0.2994,
      avg_loss: 545.52,
      accuracy: 88.42,
      epsilon: 16.62
    }
  ],

  // Global test evaluation from global_model_evaluation.json
  globalEvaluation: {
    combined: {
      accuracy: 0.8842,
      precision: 0.2915,
      recall: 0.9960,
      f1: 0.4510,
      roc_auc: 0.9377,
      total_test_samples: 25874,
      test_fraud_count: 1235,
      confusion_matrix: { tn: 21649, fp: 2990, fn: 5, tp: 1230 }
    },
    clients: {
      client1_bankA: {
        accuracy: 0.8770,
        precision: 0.2792,
        recall: 0.9983,
        f1: 0.4364,
        roc_auc: 0.9348,
        total_test_samples: 12708,
        test_fraud_count: 606,
        confusion_matrix: { tn: 10540, fp: 1562, fn: 1, tp: 605 }
      },
      client2_bankB: {
        accuracy: 0.8872,
        precision: 0.2971,
        recall: 0.9974,
        f1: 0.4578,
        roc_auc: 0.9397,
        total_test_samples: 7920,
        test_fraud_count: 378,
        confusion_matrix: { tn: 6650, fp: 892, fn: 1, tp: 377 }
      },
      client3_bankC: {
        accuracy: 0.8973,
        precision: 0.3163,
        recall: 0.9880,
        f1: 0.4792,
        roc_auc: 0.9419,
        total_test_samples: 5246,
        test_fraud_count: 251,
        confusion_matrix: { tn: 4459, fp: 536, fn: 3, tp: 248 }
      }
    }
  },

  // Differential privacy experiments from privacy_utility_tradeoff.json
  privacyTradeoff: [
    {
      experiment_id: "Exp A (No DP)",
      label: "Baseline (No DP)",
      sigma: 0.00,
      epsilon: "Infinity",
      f1: 0.3557,
      precision: 0.2172,
      recall: 0.9822,
      roc_auc: 0.9029,
      accuracy: 0.8302
    },
    {
      experiment_id: "Exp B (Low DP)",
      label: "Low Noise (sigma=0.1)",
      sigma: 0.10,
      epsilon: "83.11",
      f1: 0.0850,
      precision: 0.0489,
      recall: 0.3247,
      roc_auc: 0.5054,
      accuracy: 0.6665
    },
    {
      experiment_id: "Exp C (Medium DP)",
      label: "Medium Noise (sigma=0.5)",
      sigma: 0.50,
      epsilon: "16.62",
      f1: 0.1141,
      precision: 0.0635,
      recall: 0.5628,
      roc_auc: 0.5731,
      accuracy: 0.5829
    },
    {
      experiment_id: "Exp D (High DP)",
      label: "High Noise (sigma=1.0)",
      sigma: 1.00,
      epsilon: "8.31",
      f1: 0.1336,
      precision: 0.0718,
      recall: 0.9603,
      roc_auc: 0.6690,
      accuracy: 0.4054
    }
  ]
};
