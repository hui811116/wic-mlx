"""
Example: End-to-End Multi-View Clustering with MLX

Demonstrates loading data, setting up inference, and evaluating a multi-view model.
"""

from typing import Optional
import mlx.core as mx
import mlx.nn as nn

from dataloader import load_dataset, MultiViewDataLoader
from metric import BiMaskInferenceEngine, WeightedInferenceEngine, evaluate_clustering
from wynerloss import WynerLoss


def example_training(
    dataset_name: str = 'BDGP',
    data_path: str = './data/',
    batch_size: int = 32,
    device_mode: str = 'cpu'
):
    """
    Example training workflow for multi-view clustering.
    
    Args:
        dataset_name: Name of dataset to load
        data_path: Path to dataset directory
        batch_size: Batch size for training
        device_mode: Computation device ('cpu' or 'gpu')
    """
    
    # ============================================================================
    # 1. LOAD DATASETS
    # ============================================================================
    print(f"Loading {dataset_name} dataset...")
    
    train_dataset, config = load_dataset(dataset_name, data_path, train=True)
    test_dataset, _ = load_dataset(dataset_name, data_path, train=False)
    
    print(f"✓ Dataset Configuration:")
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
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = MultiViewDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # ============================================================================
    # 3. INITIALIZE LOSS FUNCTION
    # ============================================================================
    print("\nInitializing WynerLoss...")
    
    loss_fn = WynerLoss(
        batch_size=batch_size,
        num_classes=config.class_num,
        temperature_features=0.5,
        temperature_clusters=0.5
    )
    
    print("✓ WynerLoss ready")
    
    # ============================================================================
    # 4. TRAINING LOOP PSEUDOCODE
    # ============================================================================
    print("\nExample training loop:")
    print("-" * 60)
    
    # Note: This requires a model implementation
    # Example structure:
    """
    model = YourMultiViewModel(
        input_dims=config.dims,
        latent_dim=256,
        num_classes=config.class_num,
        num_views=config.view
    )
    
    optimizer = mx.optimizers.Adam(learning_rate=1e-3)
    
    for epoch in range(num_epochs):
        for batch_idx, (batch_views, batch_labels, _) in enumerate(train_loader):
            
            # Forward pass
            h_views, q_views, _, z_views = model(batch_views)
            
            # Compute loss (pairing views)
            h_view1, h_view2 = h_views[0], h_views[1]
            q_view1, q_view2 = q_views[0], q_views[1]
            
            loss, loss_dict = loss_fn(
                h_view1, h_view2,
                q_view1, q_view2,
                weight_feature=1.0,
                weight_cluster=1.0
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
    """
    
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
    
    from dataloader import DATASET_REGISTRY
    
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
    example_training(dataset_name='BDGP')
    
    print("\n" + "=" * 70)
    print("✓ All examples completed!")
    print("=" * 70)
