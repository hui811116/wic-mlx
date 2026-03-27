"""
Example: End-to-End Multi-View Clustering with MLX

Demonstrates loading data, setting up inference, and evaluating a multi-view model.
"""

from typing import Optional
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from dataloader import load_dataset, MultiViewDataLoader, DATASET_REGISTRY
from metric import BiMaskInferenceEngine, WeightedInferenceEngine, evaluate_clustering
from networks import NetworkWIC

from functools import partial

import argparse

# def _negative_masks(block_size):
#     n = 2 * block_size
#     mask = mx.ones((n, n), dtype=mx.bool_)
#     mask = mx.diag(mx.zeros(n, dtype=mx.bool_)) + mask
#     # off-diagonal blocks are False, diagonal blocks are True
#     for i in range(block_size):
#         mask[i, block_size + i] = False
#         mask[block_size + i, i] = False
#     return mask

def _cluster_mask(q1,q2,temperature=0.5):
    nclusters = q1.shape[1]
    p1 = mx.sum(q1,axis=0)/mx.sum(q1) # avg probability of each cluster across samples
    p2 = mx.sum(q2,axis=0)/mx.sum(q2)

    ent1 = -mx.sum(p1*mx.log(p1+1e-10))
    ent2 = -mx.sum(p2*mx.log(p2+1e-10))
    
    q1_t = mx.transpose(q1) # shape (class_num, batch_size)
    q2_t = mx.transpose(q2)

    qc = mx.concat([q1_t,q2_t],axis=0) # shape (2*class_num, batch_size)
    sim = mx.matmul(qc, mx.transpose(qc)) # shape (2*class_num, 2*class_num)
    # divide by temperature if needed
    sim = sim/temperature
    # focus on contrasting different clusters across views, so the labels are the same indices in the two halves of sim
    labels = mx.concat([mx.arange(nclusters)+nclusters, mx.arange(nclusters)], axis=0) # shape (2*class_num,)
    
    loss = nn.losses.cross_entropy(sim, labels, reduction='mean')

    return loss + ent1 + ent2

def loss_fn(model,Xs):
    # Xs is a list of views
    hs, qs, xrs, zs = model(Xs)
    loss = 0
    for i in range(len(Xs)):
        # Reconstruction loss
        loss += nn.losses.mse_loss(xrs[i], Xs[i])
        # Cluster consistency loss
        for j in range(i+1, len(Xs)):
           loss += _cluster_mask(qs[i], qs[j])
    return loss


def example_training(args):
    """
    Example training workflow for multi-view clustering.
    
    Args:
        dataset: Name of dataset to load
        datapath: Path to dataset directory
        batch_size: Batch size for training
        device_mode: Computation device ('cpu' or 'gpu')
    """
    
    # ============================================================================
    # 1. LOAD DATASETS
    # ============================================================================
    print(f"Loading {args.dataset} dataset...")
    
    train_dataset, config = load_dataset(args.dataset, args.datapath, train=True)
    test_dataset, _ = load_dataset(args.dataset, args.datapath, train=False)
    
    print(f"Dataset Configuration:")
    print(f"  - Views: {config.view}")
    print(f"  - Classes: {config.class_num}")
    print(f"  - Feature Dimensions: {config.dims}")
    print(f"  - Training Samples: {len(train_dataset)}")
    print(f"  - Test Samples: {len(test_dataset)}")
    
    # ============================================================================
    # 2. CREATE DATA LOADERS
    # ============================================================================
    print("\nCreating data loaders...")
    
    train_loader = MultiViewDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    test_loader = MultiViewDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # ============================================================================
    # 4. TRAINING LOOP PSEUDOCODE
    # ============================================================================
    print("\nExample training loop:")
    print("-" * 60)
    
    # Note: This requires a model implementation
    # Example structure:
    model = NetworkWIC(
        view=config.view,
        input_size=config.dims,
        feature_dim=256,
        high_feature_dim=512,
        class_num=config.class_num
    )
    optimizer = optim.Adam(learning_rate=args.lr)
    
    mx.eval(model.parameters())  # Set to evaluation mode for demonstration
    
    state = [model.state, optimizer.state]
    loss_and_grad_func = nn.value_and_grad(model,loss_fn)
    @partial(mx.compile,inputs=state,outputs=state)
    def train_step(Xs):
        loss, grads = loss_and_grad_func(model,Xs)
        optimizer.update(model,grads)
        return loss

    
    for e in range(args.epochs):
        loss_sum = 0
        #mx.train(model.state)
        for batch_views, batch_labels, batch_indices in train_loader:
            loss = train_step(batch_views)
            loss_sum += loss.item()
            mx.eval(model.state)
        print(f"Epoch {e+1}/{args.epochs}, Loss: {loss_sum:.4f}")
        
    print("-" * 60)
    
    # ============================================================================
    # 5. INFERENCE & EVALUATION EXAMPLE
    # ============================================================================
    print("\n\nExample inference workflow:")
    print("-" * 60)
    
    # Assuming you have a trained model:
    """
    model.eval()  # Set to evaluation mode
    
    # Choose appropriate inference engine
    inference_engine = BiMaskInferenceEngine(model, device_mode=device_mode)
    # OR
    # inference_engine = WeightedInferenceEngine(model, device_mode=device_mode)
    
    # Run inference on test set
    predictions, per_view_pred, labels, features = inference_engine.infer_dataset(
        test_loader,
        num_views=config.view,
        data_size=len(test_dataset)
    )
    
    # Comprehensive evaluation
    results = evaluate_clustering(
        labels,
        predictions,
        per_view_preds=per_view_pred,
        per_view_features=features,
        num_classes=config.class_num,
        kmeans_n_init=10,
        verbose=True
    )
    
    # Extract results
    main_metrics = results['semantic']
    print(f"\nSemantic Labels Performance: {main_metrics}")
    
    # Save results
    import json
    results_dict = {
        'semantic': main_metrics.to_dict(),
        'per_view_predictions': {
            str(k): v.to_dict() for k, v in results['per_view_pred'].items()
        }
    }
    with open('results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    """
    
    print("-" * 60)
    
    # ============================================================================
    # 6. BATCH PROCESSING EXAMPLE
    # ============================================================================
    print("\n\nExample batch processing:")
    print("-" * 60)
    
    print("\nIterating through first 3 batches:")
    for batch_idx, (batch_views, batch_labels, batch_indices) in enumerate(train_loader):
        if batch_idx >= 3:
            break
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Number of views: {len(batch_views)}")
        for v, view in enumerate(batch_views):
            print(f"    View {v+1} shape: {view.shape}")
        print(f"  Labels shape: {batch_labels.shape}")
        print(f"  Indices shape: {batch_indices.shape}")
    
    print("-" * 60)


def example_configuration():
    """Demonstrate dataset registry and configuration access."""
    
    print("Dataset Registry")
    print("=" * 60)
    
    
    for name, config in DATASET_REGISTRY.items():
        print(f"\n{name}:")
        print(f"  Views: {config.view}")
        print(f"  Classes: {config.class_num}")
        print(f"  Dimensions: {config.dims}")


def example_metrics():
    """Demonstrate metrics computation."""
    
    print("\nMetrics Examples")
    print("=" * 60)
    
    import numpy as np
    from metric import evaluate, cluster_acc, purity
    
    # Mock predictions
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 2, 2, 1])
    
    # Individual metric computation
    print(f"\nIndividual Metrics:")
    print(f"  Accuracy: {cluster_acc(y_true, y_pred):.4f}")
    print(f"  Purity: {purity(y_true, y_pred):.4f}")
    
    # Combined evaluation
    metrics = evaluate(y_true, y_pred)
    print(f"\nCombined Evaluation:")
    print(f"  {metrics}")
    
    # Access individual values
    print(f"\nAccess individual metrics:")
    print(f"  ACC: {metrics.accuracy:.4f}")
    print(f"  NMI: {metrics.nmi:.4f}")
    print(f"  ARI: {metrics.ari:.4f}")
    print(f"  PUR: {metrics.purity:.4f}")
    
    # Convert to dictionary
    print(f"\nAs dictionary:")
    print(f"  {metrics.to_dict()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath",type=str,default="./data",help="/path/to/dataset")
    parser.add_argument("--dataset",type=str,default="BDGP",choices=DATASET_REGISTRY.keys(),help="choose one of the dataset")
    parser.add_argument("--batch_size",type=int,default=32,help="mini batch size for training")
    parser.add_argument("--cpu",action="store_true",default=False,help="Force using CPU to run the code")
    parser.add_argument("--lr",type=float,default=1e-4,help="learning rate for training")
    parser.add_argument("--epochs",type=int,default=10,help="number of epochs for training")
    args = parser.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)
    print("=" * 70)
    print("WIC-MLX: Multi-View Clustering with MLX")
    print("=" * 70)
    
    # Run examples
    example_configuration()
    example_metrics()
    
    # Main training workflow example
    print("\n\n" + "=" * 70)
    print("MAIN EXAMPLE: Training Workflow")
    print("=" * 70)
    example_training(args)
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
