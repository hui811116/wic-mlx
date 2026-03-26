"""
MLX Implementation of Wyner Loss for Contrastive Learning

Implements bipartite contrastive loss on features and cluster assignments,
optimized for multi-view learning scenarios.
"""

from typing import Tuple

import math
import mlx.core as mx
import mlx.nn as nn


class WynerLoss(nn.Module):
    """
    Wyner Loss for multi-view contrastive learning.
    
    Implements both feature-level and cluster-assignment-level contrastive losses
    with bipartite structure matching for paired views.
    """

    def __init__(self, batch_size: int, num_classes: int, 
                 temperature_features: float = 0.5,
                 temperature_clusters: float = 0.5):
        """
        Initialize Wyner Loss.
        
        Args:
            batch_size: Batch size
            num_classes: Number of semantic classes
            temperature_features: Temperature for feature contrastive loss
            temperature_clusters: Temperature for cluster assignment loss
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.temperature_features = temperature_features
        self.temperature_clusters = temperature_clusters
        
        # Create correlation masks for bipartite structure
        self._mask = self._create_bipartite_mask(batch_size)

    def _create_bipartite_mask(self, batch_size: int) -> mx.array:
        """
        Create mask for bipartite correlation structure.
        
        In bipartite setting: first batch_size samples from view 1,
        second batch_size samples from view 2.
        We want positive pairs on off-diagonals and mask diagonal + paired positions.
        
        Args:
            batch_size: Size of each view's batch
            
        Returns:
            Boolean mask for negative samples
        """
        n = 2 * batch_size
        mask = mx.ones((n, n))
        
        # Mask diagonal (self-similarity)
        mask = mx.diag(mx.zeros(n)) * (-1) + mask
        
        # Mask positive pairs (cross-view matches)
        for i in range(batch_size):
            mask[i, batch_size + i] = mx.array(0.0)
            mask[batch_size + i, i] = mx.array(0.0)
        
        return mask.astype(mx.bool_)

    def forward_feature_loss(self, h_view1: mx.array, 
                            h_view2: mx.array) -> mx.array:
        """
        Compute contrastive loss on high-level features.
        
        Args:
            h_view1: Features from view 1 (batch_size, feature_dim)
            h_view2: Features from view 2 (batch_size, feature_dim)
            
        Returns:
            Scalar loss value
        """
        # Concatenate features from both views
        batch_size = h_view1.shape[0]
        h_combined = mx.concatenate([h_view1, h_view2], axis=0)  # (2*batch_size, feat_dim)
        
        # Compute scaled cosine similarity
        sim_matrix = mx.matmul(h_combined, h_combined.T) / self.temperature_features
        
        # Extract positive pairs
        positive_sim_1 = mx.diagonal(sim_matrix, offset=batch_size)  # h_i with h_j
        positive_sim_2 = mx.diagonal(sim_matrix, offset=-batch_size)  # h_j with h_i
        positive_pair = mx.concatenate([positive_sim_1, positive_sim_2], axis=0)
        positive_pair = mx.reshape(positive_pair, (-1, 1))
        
        # Extract negative pairs
        mask_expanded = mx.broadcast_to(mx.expand_dims(self._mask, 0), 
                                       (1, 2*batch_size, 2*batch_size))
        negative_pairs = sim_matrix[self._mask]
        negative_pairs = mx.reshape(negative_pairs, (2*batch_size, -1))
        
        # Combine positive and negatives
        logits = mx.concatenate([positive_pair, negative_pairs], axis=1)
        labels = mx.zeros((2*batch_size,), dtype=mx.int32)
        
        # Cross-entropy loss
        loss = self._cross_entropy(logits, labels)
        loss = loss / (2 * batch_size)
        
        return loss

    def forward_cluster_loss(self, q_view1: mx.array, 
                            q_view2: mx.array) -> mx.array:
        """
        Compute contrastive loss on cluster assignments with entropy regularization.
        
        Args:
            q_view1: Cluster assignment probabilities from view 1 (batch_size, num_classes)
            q_view2: Cluster assignment probabilities from view 2 (batch_size, num_classes)
            
        Returns:
            Scalar loss value (contrastive + entropy term)
        """
        # Compute cluster priors and entropy
        p_view1 = mx.sum(q_view1, axis=0) / mx.sum(q_view1)
        entropy_view1 = self._entropy(p_view1)
        
        p_view2 = mx.sum(q_view2, axis=0) / mx.sum(q_view2)
        entropy_view2 = self._entropy(p_view2)
        
        total_entropy = entropy_view1 + entropy_view2
        
        # Transpose for view consistency
        q_view1_t = mx.transpose(q_view1)  # (num_classes, batch_size)
        q_view2_t = mx.transpose(q_view2)  # (num_classes, batch_size)
        
        # Concatenate cluster-level embeddings
        n_clusters = 2 * self.num_classes
        q_combined = mx.concatenate([q_view1_t, q_view2_t], axis=0)  # (2*num_classes, batch_size)
        
        # Compute pairwise cosine similarity
        q_norm = mx.sqrt(mx.sum(q_combined ** 2, axis=1, keepdims=True))
        q_normalized = q_combined / (q_norm + 1e-8)
        
        sim_matrix = mx.matmul(q_normalized, mx.transpose(q_normalized)) / self.temperature_clusters
        
        # Extract positive pairs
        positive_sim_1 = mx.diagonal(sim_matrix, offset=self.num_classes)
        positive_sim_2 = mx.diagonal(sim_matrix, offset=-self.num_classes)
        positive_pair = mx.concatenate([positive_sim_1, positive_sim_2], axis=0)
        positive_pair = mx.reshape(positive_pair, (-1, 1))
        
        # Extract negative pairs (mask bipartite structure)
        mask_clusters = self._create_bipartite_mask(self.num_classes)
        negative_pairs = sim_matrix[mask_clusters]
        negative_pairs = mx.reshape(negative_pairs, (2*self.num_classes, -1))
        
        # Combine and compute loss
        logits = mx.concatenate([positive_pair, negative_pairs], axis=1)
        labels = mx.zeros((2*self.num_classes,), dtype=mx.int32)
        
        loss = self._cross_entropy(logits, labels)
        loss = loss / (2 * self.num_classes)
        
        return loss + total_entropy

    def forward(self, h_view1: mx.array, h_view2: mx.array,
               q_view1: mx.array, q_view2: mx.array,
               weight_feature: float = 1.0,
               weight_cluster: float = 1.0) -> Tuple[mx.array, dict]:
        """
        Compute total Wyner loss combining feature and cluster losses.
        
        Args:
            h_view1: Features from view 1
            h_view2: Features from view 2
            q_view1: Cluster assignments from view 1
            q_view2: Cluster assignments from view 2
            weight_feature: Weight for feature loss
            weight_cluster: Weight for cluster loss
            
        Returns:
            Tuple of (total_loss, loss_dict with component losses)
        """
        feature_loss = self.forward_feature_loss(h_view1, h_view2)
        cluster_loss = self.forward_cluster_loss(q_view1, q_view2)
        
        total_loss = (weight_feature * feature_loss + 
                     weight_cluster * cluster_loss)
        
        loss_dict = {
            'total': total_loss,
            'feature': feature_loss,
            'cluster': cluster_loss,
        }
        
        return total_loss, loss_dict

    @staticmethod
    def _entropy(p: mx.array) -> mx.array:
        """
        Compute Shannon entropy of probability distribution.
        
        Args:
            p: Probability distribution (must sum to 1)
            
        Returns:
            Scalar entropy value
        """
        # Clip to avoid log(0)
        p_safe = mx.clip(p, a_min=1e-8)
        entropy = -mx.sum(p * mx.log(p_safe))
        return entropy / math.log(p.shape[0])  # Normalize by max entropy

    @staticmethod
    def _cross_entropy(logits: mx.array, labels: mx.array) -> mx.array:
        """
        Compute cross-entropy loss.
        
        Args:
            logits: Raw predictions (batch_size, num_classes)
            labels: Target class indices (batch_size,)
            
        Returns:
            Sum of cross-entropy losses
        """
        log_probs = mx.log_softmax(logits, axis=1)
        batch_size = logits.shape[0]
        loss = -mx.sum(log_probs[mx.arange(batch_size), labels])
        return loss


class MultiViewWynerLoss(nn.Module):
    """
    Multi-view extension of Wyner Loss supporting more than 2 views.
    """

    def __init__(self, batch_size: int, num_classes: int,
                 num_views: int = 2,
                 temperature_features: float = 0.5,
                 temperature_clusters: float = 0.5,
                 loss_weights: dict = None):
        """
        Initialize multi-view Wyner loss.
        
        Args:
            batch_size: Batch size
            num_classes: Number of semantic classes
            num_views: Number of views
            temperature_features: Temperature for feature loss
            temperature_clusters: Temperature for cluster loss
            loss_weights: Dict specifying weights between views
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_views = num_views
        self.loss_weights = loss_weights or {i: 1.0 / num_views for i in range(num_views)}
        
        # Create pairwise Wyner loss modules
        self.pairwise_losses = nn.ModuleList([
            WynerLoss(batch_size, num_classes, temperature_features, temperature_clusters)
            for _ in range(num_views * (num_views - 1) // 2)
        ])

    def forward(self, h_views: list, q_views: list) -> Tuple[mx.array, dict]:
        """
        Compute multi-view Wyner loss.
        
        Args:
            h_views: List of feature tensors for each view
            q_views: List of cluster assignment tensors for each view
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        total_loss = mx.array(0.0)
        loss_components = {}
        pair_idx = 0
        
        # Compute pairwise losses
        for i in range(self.num_views):
            for j in range(i + 1, self.num_views):
                pair_name = f"pair_{i}_{j}"
                pair_loss, pair_dict = self.pairwise_losses[pair_idx](
                    h_views[i], h_views[j],
                    q_views[i], q_views[j]
                )
                total_loss = total_loss + pair_loss
                loss_components[pair_name] = pair_loss.item()
                pair_idx += 1
        
        # Average over pairs
        total_loss = total_loss / pair_idx
        loss_components['total'] = total_loss.item()
        
        return total_loss, loss_components
