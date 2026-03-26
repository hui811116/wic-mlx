"""
MLX Evaluation Metrics and Inference Functions

Provides clustering evaluation metrics (NMI, ARI, ACC, Purity) and
inference utilities for multi-view models.
"""

from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import mlx.core as mx
import mlx.nn as nn


@dataclass
class ClusteringMetrics:
    """Container for clustering evaluation metrics."""
    accuracy: float
    nmi: float
    ari: float
    purity: float

    def __str__(self) -> str:
        return (f"ACC = {self.accuracy:.4f} | NMI = {self.nmi:.4f} | "
                f"ARI = {self.ari:.4f} | PUR = {self.purity:.4f}")

    def to_dict(self) -> Dict[str, float]:
        return {
            'accuracy': self.accuracy,
            'nmi': self.nmi,
            'ari': self.ari,
            'purity': self.purity
        }


def cluster_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute clustering accuracy using Hungarian algorithm.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster assignments
        
    Returns:
        Clustering accuracy
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size, "Label sizes must match"
    
    # Build confusion matrix
    num_classes = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    # Find optimal assignment
    assignment = linear_sum_assignment(w.max() - w)
    indices = np.column_stack(assignment)
    
    # Calculate accuracy
    accuracy = sum(w[i, j] for i, j in indices) / y_pred.size
    return accuracy


def purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute clustering purity.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster assignments
        
    Returns:
        Purity score
    """
    # Normalize true labels
    y_true = y_true.copy()
    unique_labels = np.unique(y_true)
    ordered_labels = np.arange(len(unique_labels))
    
    for k, label in enumerate(unique_labels):
        y_true[y_true == label] = ordered_labels[k]
    
    # Vote for each cluster
    y_voted = np.zeros(y_true.shape)
    unique_clusters = np.unique(y_pred)
    bins = np.concatenate([unique_clusters, [unique_clusters.max() + 1]])
    
    for cluster in unique_clusters:
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted[y_pred == cluster] = winner
    
    return accuracy_score(y_true, y_voted)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> ClusteringMetrics:
    """
    Comprehensive evaluation of clustering results.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster assignments
        
    Returns:
        ClusteringMetrics object with all metrics
    """
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    acc = cluster_acc(y_true, y_pred)
    pur = purity(y_true, y_pred)
    
    return ClusteringMetrics(acc, nmi, ari, pur)


def _extract_predictions(logits: mx.array) -> np.ndarray:
    """Convert logits to hard predictions."""
    return np.argmax(np.array(logits), axis=1)


def _to_numpy(*arrays) -> tuple:
    """Convert MLX arrays to numpy."""
    return tuple(np.array(arr) if isinstance(arr, mx.array) else arr 
                 for arr in arrays)


class InferenceEngine:
    """Base inference engine for multi-view models."""

    def __init__(self, model: nn.Module, device_mode: str = 'cpu'):
        """
        Initialize inference engine.
        
        Args:
            model: MLX model with forward() method
            device_mode: Computation mode ('cpu', 'gpu')
        """
        self.model = model
        self.device_mode = device_mode

    def _prepare_batch(self, batch_views: List[mx.array], 
                      num_views: int) -> List[mx.array]:
        """Prepare batch data for inference."""
        return batch_views[:num_views]

    def _aggregate_predictions(self, logits_per_view: List[mx.array],
                              method: str = 'mean') -> mx.array:
        """Aggregate predictions across views."""
        if method == 'mean':
            stacked = mx.stack(logits_per_view)
            return mx.mean(stacked, axis=0)
        elif method == 'log_sum':
            log_probs = [mx.nn.log_softmax(logits, axis=1) for logits in logits_per_view]
            stacked = mx.stack(log_probs)
            return mx.mean(stacked, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


class BiMaskInferenceEngine(InferenceEngine):
    """Inference engine for BiMask model."""

    def infer_batch(self, batch_views: List[mx.array]) -> Tuple[mx.array, List[mx.array]]:
        """
        Perform inference on a batch.
        
        Returns:
            Tuple of (aggregated_logits, per_view_logits)
        """
        self.model.eval()
        
        with mx.no_grad():
            # Forward pass
            hs, qs, _, zs = self.model(batch_views)
        
        num_views = len(qs) // 2  # assumes paired predictions
        log_probs = [mx.nn.log_softmax(q, axis=1) for q in qs]
        
        # Aggregate paired predictions
        aggregated = []
        for i in range(num_views):
            pair_agg = mx.softmax(log_probs[2*i] + log_probs[2*i+1], axis=1)
            aggregated.append(pair_agg)
        
        final = mx.mean(mx.stack(aggregated), axis=0)
        
        return final, zs, qs

    def infer_dataset(self, loader, num_views: int, 
                     data_size: int) -> Tuple[np.ndarray, List[np.ndarray], 
                                             np.ndarray, List[np.ndarray]]:
        """
        Perform inference on entire dataset.
        
        Returns:
            Tuple of (predictions, per_view_predictions, labels, low_level_features)
        """
        self.model.eval()
        predictions = []
        per_view_preds = [[] for _ in range(len(num_views) * 2)]
        labels = []
        features = [[] for _ in range(num_views)]
        
        for batch_views, batch_labels, _ in loader:
            final_logits, zs, qs = self.infer_batch(batch_views)
            
            # Aggregate predictions
            preds = _extract_predictions(final_logits)
            predictions.extend(preds)
            
            # Per-view predictions
            for v in range(len(qs)):
                view_preds = _extract_predictions(qs[v])
                per_view_preds[v].extend(view_preds)
            
            # Low-level features
            for v in range(num_views):
                features[v].extend(_to_numpy(zs[v]))
            
            # Labels
            labels.extend(_to_numpy(batch_labels))
        
        predictions = np.array(predictions)
        labels = np.array(labels).reshape(data_size)
        features = [np.array(f) for f in features]
        per_view_preds = [np.array(p) for p in per_view_preds]
        
        return predictions, per_view_preds, labels, features


class WeightedInferenceEngine(InferenceEngine):
    """Inference engine for weighted multi-view model."""

    def infer_batch(self, batch_views: List[mx.array]) -> Tuple[mx.array, List[mx.array], mx.array]:
        """
        Perform inference with learned weights.
        
        Returns:
            Tuple of (aggregated_logits, per_view_logits, weights)
        """
        self.model.eval()
        
        with mx.no_grad():
            hs, qs, _, zs, kappas = self.model(batch_views)
        
        num_pairs = len(qs) // 2
        log_probs = [mx.nn.log_softmax(q, axis=1) for q in qs]
        
        # Aggregate with learned weights
        llrs = []
        for i in range(num_pairs):
            llr = log_probs[2*i] + log_probs[2*i+1]
            llrs.append(llr)
        
        llrs_stacked = mx.stack(llrs, axis=2)  # (batch, num_classes, num_pairs)
        weighted = (llrs_stacked * kappas.reshape(-1, 1, num_pairs)).sum(axis=2)
        final = mx.softmax(weighted, axis=1)
        
        return final, zs, qs, kappas

    def infer_dataset(self, loader, num_views: int,
                     data_size: int) -> Tuple[np.ndarray, List[np.ndarray],
                                             np.ndarray, List[np.ndarray]]:
        """Inference on entire dataset."""
        self.model.eval()
        predictions = []
        per_view_preds = [[] for _ in range(len(num_views) * 2)]
        labels = []
        features = [[] for _ in range(num_views)]
        
        for batch_views, batch_labels, _ in loader:
            final_logits, zs, qs, kappas = self.infer_batch(batch_views)
            
            preds = _extract_predictions(final_logits)
            predictions.extend(preds)
            
            for v in range(len(qs)):
                view_preds = _extract_predictions(qs[v])
                per_view_preds[v].extend(view_preds)
            
            for v in range(num_views):
                features[v].extend(_to_numpy(zs[v]))
            
            labels.extend(_to_numpy(batch_labels))
        
        predictions = np.array(predictions)
        labels = np.array(labels).reshape(data_size)
        features = [np.array(f) for f in features]
        per_view_preds = [np.array(p) for p in per_view_preds]
        
        return predictions, per_view_preds, labels, features


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray,
                       per_view_preds: Optional[List[np.ndarray]] = None,
                       per_view_features: Optional[List[np.ndarray]] = None,
                       num_classes: Optional[int] = None,
                       kmeans_n_init: int = 10,
                       verbose: bool = True) -> Dict[str, ClusteringMetrics]:
    """
    Comprehensive clustering evaluation with optional per-view metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predictions on semantic labels
        per_view_preds: Per-view predictions (optional)
        per_view_features: Per-view low-level features (optional)
        num_classes: Number of classes for K-means
        kmeans_n_init: K-means n_init parameter
        verbose: Print results
        
    Returns:
        Dictionary with metrics for different levels
    """
    results = {}
    
    # Main evaluation
    main_metrics = evaluate(y_true, y_pred)
    results['semantic'] = main_metrics
    
    if verbose:
        print(f"Semantic Labels: {main_metrics}")
    
    # Per-view predictions
    if per_view_preds is not None:
        results['per_view_pred'] = {}
        if verbose:
            print("\nPer-view Cluster Assignments:")
        
        for v, preds in enumerate(per_view_preds):
            metrics = evaluate(y_true, preds)
            results['per_view_pred'][v] = metrics
            if verbose:
                print(f"  View {v+1}: {metrics}")
    
    # Per-view features with K-means
    if per_view_features is not None and num_classes is not None:
        results['per_view_features'] = {}
        if verbose:
            print("\nPer-view Low-level Features (K-means):")
        
        for v, features in enumerate(per_view_features):
            kmeans = KMeans(n_clusters=num_classes, n_init=kmeans_n_init, random_state=42)
            preds = kmeans.fit_predict(features)
            metrics = evaluate(y_true, preds)
            results['per_view_features'][v] = metrics
            if verbose:
                print(f"  View {v+1}: {metrics}")
    
    return results
