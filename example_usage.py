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
from wynerloss import WynerLoss
from networks import NetworkWIC


import argparse

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
    # 3. INITIALIZE LOSS FUNCTION
    # ============================================================================
    print("\nInitializing WynerLoss...")
    
    loss_fn = WynerLoss(
        batch_size=args.batch_size,
        num_classes=config.class_num,
        temperature_features=0.5,
        temperature_clusters=0.5
    )
    
    print("WynerLoss ready")
    
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
