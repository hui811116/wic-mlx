"""
Evaluation script for clustering.

This script provides utilities for evaluating clustering results, building upon the
core metrics defined in metric.py.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

# Re-export the core evaluation functions from metric.py
from metric import evaluate, cluster_acc, purity, ClusteringMetrics

def clustering_mapping(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Finds the best mapping between predicted clusters and true labels using the
    Hungarian algorithm.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted cluster assignments.

    Returns:
        A dictionary mapping predicted cluster indices to true label indices.
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
        
    row_idx, col_idx = linear_sum_assignment(w.max() - w)
    
    return {row_idx[idx]: col_idx[idx] for idx in range(len(row_idx))}

# The other functions from the user's file (`clustering_accuracy`, `cluster_acc`, 
# `purity`, `evaluate`) are already implemented in `metric.py`. 
# We can use the versions from `metric.py` for consistency.

if __name__ == '__main__':
    # Example usage:
    # This part is for demonstration. In a real scenario, you would load your
    # model's predictions and the true labels.
    
    # Example data
    y_true_example = np.array([0, 0, 1, 1, 2, 2])
    y_pred_example = np.array([1, 1, 0, 0, 2, 2]) # A permutation of the true labels
    
    print("Example Evaluation:")
    print("-------------------")
    
    # Using the evaluate function from metric.py
    metrics = evaluate(y_true_example, y_pred_example)
    print(f"Metrics: {metrics}")
    
    # Using the new clustering_mapping function
    mapping = clustering_mapping(y_true_example, y_pred_example)
    print(f"Cluster mapping (predicted -> true): {mapping}")
    
    # You can use this mapping to remap your predicted labels
    y_pred_remapped = np.copy(y_pred_example)
    for k, v in mapping.items():
        y_pred_remapped[y_pred_example == k] = v
        
    print(f"Original predictions: {y_pred_example}")
    print(f"Remapped predictions: {y_pred_remapped}")
    
    # Accuracy should be 1.0 after remapping
    remapped_acc = cluster_acc(y_true_example, y_pred_remapped)
    print(f"Accuracy after remapping: {remapped_acc:.4f}")
