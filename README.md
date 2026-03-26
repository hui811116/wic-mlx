# WIC-MLX: MLX Implementation of Multi-View Clustering

MLX versions of core WIC (Weighted Information Clustering) modules with significant improvements in code organization, efficiency, and maintainability.

## Overview

This folder contains optimized implementations of three key modules converted from PyTorch to MLX:

- **dataloader.py** - Multi-view dataset loading and batch processing
- **metric.py** - Clustering evaluation metrics and inference engines
- **wynerloss.py** - Bipartite contrastive learning loss

## Key Improvements

### 1. **dataloader.py**

**Organization:**
- Introduced `BaseMultiViewDataset` abstract base class eliminating code duplication
- Created `DatasetConfig` dataclass for centralized dataset metadata
- Implemented `MultiViewDataLoader` with efficient batching and shuffling
- Dataset registry pattern for scalable dataset management

**Efficiency:**
- Unified train/test splitting logic with `_split_data()` methods
- Uniform class-aware splitting via `_split_uniform_indices()`
- MLX array conversion on demand via `to_mx_arrays()`
- Type hints throughout for better IDE support

**Features:**
- Support for datasets: BDGP, MNIST-USPS, CCV, Fashion, Caltech (2-5 views)
- Path handling with `pathlib.Path`
- Consistent reshape operations for feature dimensions
- MinMaxScaler integration for data normalization

### 2. **metric.py**

**Organization:**
- Introduced `ClusteringMetrics` dataclass consolidating 4 metrics (ACC, NMI, ARI, Purity)
- Created `InferenceEngine` base class with specialized subclasses:
  - `BiMaskInferenceEngine` - Paired view handling
  - `WeightedInferenceEngine` - Learned weight aggregation
- Unified evaluation via `evaluate_clustering()` function

**Efficiency:**
- Helper functions for MLX↔NumPy conversion
- Batch processing in inference loops
- Reduced memory footprint with `mx.no_grad()` context

**Features:**
- Metrics object with `to_dict()` and `__str__()` methods
- Per-view metrics and K-means clustering on features
- Comprehensive reporting with optional verbose output
- Support for different prediction aggregation strategies

### 3. **wynerloss.py**

**Organization:**
- Enhanced class docstrings with detailed parameter documentation
- `MultiViewWynerLoss` for arbitrary number of views
- Separated concerns: feature loss, cluster loss, entropy regularization

**Improvements:**
- Bipartite mask computation now clearer and more robust
- Cross-entropy computation extracted as static method
- Entropy calculation with normalization by max entropy
- Loss dictionary returns for debugging and analysis
- Type hints for all parameters and returns

**Efficiency:**
- Lazy mask creation on initialization
- Vectorized operations using MLX primitives
- Numerical stability (1e-8 epsilon for division)

## API Usage

### Loading Data

```python
from wic_mlx.dataloader import load_dataset, MultiViewDataLoader

# Load dataset
dataset, config = load_dataset('BDGP', './data/', train=True)
print(f"Data size: {config.data_size}, Views: {config.view}, Classes: {config.class_num}")

# Create data loader
loader = MultiViewDataLoader(dataset, batch_size=32, shuffle=True)
for batch_views, labels, indices in loader:
    # Process batch
    pass
```

### Evaluation

```python
from wic_mlx.metric import evaluate_clustering, BiMaskInferenceEngine

# Create inference engine
engine = BiMaskInferenceEngine(model)
preds, per_view_preds, labels, features = engine.infer_dataset(
    loader, num_views=2, data_size=len(dataset)
)

# Evaluate
results = evaluate_clustering(
    labels, preds,
    per_view_preds=per_view_preds,
    per_view_features=features,
    num_classes=config.class_num,
    verbose=True
)
```

### Loss Computation

```python
from wic_mlx.wynerloss import WynerLoss

loss_fn = WynerLoss(
    batch_size=32,
    num_classes=5,
    temperature_features=0.5,
    temperature_clusters=0.5
)

# Forward pass
total_loss, loss_dict = loss_fn(
    h_view1, h_view2, q_view1, q_view2,
    weight_feature=1.0, weight_cluster=1.0
)
```

## Dataset Details

| Dataset | Views | Classes | Samples | Dominant Modality |
|---------|-------|---------|---------|------------------|
| BDGP | 2 | 5 | 2500 | Gene expression |
| MNIST-USPS | 2 | 10 | 5000 | Image |
| CCV | 3 | 20 | 6773 | Video (STIP+SIFT+MFCC) |
| Fashion | 3 | 10 | 10000 | Image (multiple CNNs) |
| Caltech-2V to 5V | 2-5 | 7 | 1400 | Image (multiple features) |

## Comparison: PyTorch vs MLX

### Advantages of MLX Implementation

1. **Device Agnostic**: Automatically optimizes for available hardware (CPU/GPU/Metal)
2. **Composable APIs**: Clear separation of concerns with modular design
3. **Better Type Safety**: Extensive type hints aid development
4. **Centralized Config**: Dataset registry and dataclasses reduce config duplication
5. **Memory Efficient**: Lazy array conversion and no-grad contexts
6. **Cleaner Inference**: Dedicated engine classes vs scattered functions

### Key Architectural Changes

- Replaced inheritance from `torch.utils.data.Dataset` with custom base class
- Removed DataLoader dependency; implemented efficient replacement
- Class-based loss module with clearer parameter documentation
- Dataclass-based configuration management
- Engine pattern for inference abstraction

## Dependencies

```
mlx>=0.0.1
numpy>=1.20
scipy>=1.7
scikit-learn>=0.24
```

## Notes

- MLX arrays are evaluated lazily; use `np.array()` to eagerly compute results
- Model forward passes should be decorated with `mx.no_grad()` during inference
- Batch sizes should consider available hardware memory
- For large datasets, use streaming loading with smaller batch sizes
