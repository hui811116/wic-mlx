"""
MLX Data Loaders for Multi-view Clustering

This module provides efficient data loading utilities for multi-view datasets
with support for train/test splitting and normalization.
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scipy.io
import mlx.core as mx


# Configuration
TRAIN_TEST_SPLIT = 0.9


@dataclass
class DatasetConfig:
    """Configuration for dataset parameters."""
    name: str
    dims: List[int]
    view: int
    class_num: int
    data_size: Optional[int] = None


class BaseMultiViewDataset:
    """Base class for multi-view datasets."""

    def __init__(self, num_views: int):
        self.num_views = num_views
        self.data_views: List[np.ndarray] = []
        self.labels: np.ndarray = None
        self.indices: np.ndarray = None

    def __len__(self) -> int:
        """Return dataset size."""
        if self.labels is not None:
            return len(self.labels)
        return self.data_views[0].shape[0]

    def __getitem__(self, idx: int) -> Tuple[List[mx.array], mx.array, mx.array]:
        """Return sample as (views, label, index)."""
        views = [mx.array(view[idx]) for view in self.data_views]
        label = mx.array(self.labels[idx])
        index = mx.array(np.array(idx))
        return views, label, index

    def to_mx_arrays(self) -> None:
        """Convert data to MLX arrays."""
        self.data_views = [mx.array(view) for view in self.data_views]
        if self.labels is not None:
            self.labels = mx.array(self.labels)


class BDGP(BaseMultiViewDataset):
    """BDGP multi-view dataset (2 views)."""

    def __init__(self, path: str, train: bool = False):
        super().__init__(num_views=2)
        path = Path(path)
        
        data1 = scipy.io.loadmat(str(path / 'BDGP.mat'))['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(str(path / 'BDGP.mat'))['X2'].astype(np.float32)
        labels = scipy.io.loadmat(str(path / 'BDGP.mat'))['Y'].transpose()
        
        ntrain = int(data1.shape[0] * TRAIN_TEST_SPLIT)
        self._split_data(data1, data2, labels, ntrain, train)

    def _split_data(self, data1, data2, labels, split_idx, train):
        """Split data into train/test."""
        if train:
            self.data_views = [data1[:split_idx], data2[:split_idx]]
            self.labels = labels[:split_idx]
        else:
            self.data_views = [data1[split_idx:], data2[split_idx:]]
            self.labels = labels[split_idx:]


class MNIST_USPS(BaseMultiViewDataset):
    """MNIST-USPS multi-view dataset (2 views)."""

    def __init__(self, path: str, train: bool = False):
        super().__init__(num_views=2)
        path = Path(path)
        
        mat_file = scipy.io.loadmat(str(path / 'MNIST_USPS.mat'))
        labels = mat_file['Y'].astype(np.int32).reshape(5000,)
        v1 = mat_file['X1'].astype(np.float32)
        v2 = mat_file['X2'].astype(np.float32)
        
        ntrain = int(v1.shape[0] * TRAIN_TEST_SPLIT)
        self._split_data(v1, v2, labels, ntrain, train)

    def _split_data(self, v1, v2, labels, split_idx, train):
        """Split data into train/test."""
        if train:
            self.data_views = [v1[:split_idx].reshape(-1, 784), 
                              v2[:split_idx].reshape(-1, 784)]
            self.labels = labels[:split_idx]
        else:
            self.data_views = [v1[split_idx:].reshape(-1, 784), 
                              v2[split_idx:].reshape(-1, 784)]
            self.labels = labels[split_idx:]


class CCV(BaseMultiViewDataset):
    """CCV multi-view dataset (3 views)."""

    def __init__(self, path: str, train: bool = False):
        super().__init__(num_views=3)
        path = Path(path)
        
        # Load and normalize
        data1 = np.load(path / 'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        data1 = scaler.fit_transform(data1)
        
        data2 = np.load(path / 'SIFT.npy').astype(np.float32)
        data3 = np.load(path / 'MFCC.npy').astype(np.float32)
        labels = np.squeeze(np.load(path / 'label.npy'))
        
        train_idx, test_idx = _split_uniform_indices(labels, TRAIN_TEST_SPLIT)
        self._split_data(data1, data2, data3, labels, train_idx, test_idx, train)

    def _split_data(self, data1, data2, data3, labels, train_idx, test_idx, train):
        """Split data using uniform indices."""
        if train:
            self.data_views = [data1[train_idx], data2[train_idx], data3[train_idx]]
            self.labels = labels[train_idx]
        else:
            self.data_views = [data1[test_idx], data2[test_idx], data3[test_idx]]
            self.labels = labels[test_idx]


class Fashion(BaseMultiViewDataset):
    """Fashion multi-view dataset (3 views)."""

    def __init__(self, path: str, train: bool = False):
        super().__init__(num_views=3)
        path = Path(path)
        
        mat_file = scipy.io.loadmat(str(path / 'Fashion.mat'))
        labels = mat_file['Y'].astype(np.int32).reshape(10000,)
        v1 = mat_file['X1'].astype(np.float32)
        v2 = mat_file['X2'].astype(np.float32)
        v3 = mat_file['X3'].astype(np.float32)
        
        ntrain = int(v1.shape[0] * TRAIN_TEST_SPLIT)
        self._split_data(v1, v2, v3, labels, ntrain, train)

    def _split_data(self, v1, v2, v3, labels, split_idx, train):
        """Split data into train/test."""
        if train:
            self.data_views = [v1[:split_idx].reshape(-1, 784),
                              v2[:split_idx].reshape(-1, 784),
                              v3[:split_idx].reshape(-1, 784)]
            self.labels = labels[:split_idx]
        else:
            self.data_views = [v1[split_idx:].reshape(-1, 784),
                              v2[split_idx:].reshape(-1, 784),
                              v3[split_idx:].reshape(-1, 784)]
            self.labels = labels[split_idx:]


class Caltech(BaseMultiViewDataset):
    """Caltech multi-view dataset (2-5 views)."""

    def __init__(self, path: str, num_views: int = 2, train: bool = False):
        super().__init__(num_views=num_views)
        path = Path(path)
        
        mat_file = scipy.io.loadmat(str(path))
        scaler = MinMaxScaler()
        
        views_raw = [
            scaler.fit_transform(mat_file['X1'].astype(np.float32)),
            scaler.fit_transform(mat_file['X2'].astype(np.float32)),
            scaler.fit_transform(mat_file['X3'].astype(np.float32)),
            scaler.fit_transform(mat_file['X4'].astype(np.float32)),
            scaler.fit_transform(mat_file['X5'].astype(np.float32)),
        ]
        labels = mat_file['Y'].transpose()
        
        ntrain = int(labels.shape[0] * TRAIN_TEST_SPLIT)
        self._split_data(views_raw, labels, num_views, ntrain, train)

    def _split_data(self, views_raw, labels, num_views, split_idx, train):
        """Split data and select specified views."""
        # View selection mapping
        view_selection = {
            2: [0, 1],
            3: [0, 1, 4],
            4: [0, 1, 4, 3],
            5: [0, 1, 4, 3, 2],
        }
        selected_views = view_selection.get(num_views, [0, 1])
        
        if train:
            self.data_views = [views_raw[v][:split_idx] for v in selected_views]
            self.labels = labels[:split_idx]
        else:
            self.data_views = [views_raw[v][split_idx:] for v in selected_views]
            self.labels = labels[split_idx:]


class MultiViewDataLoader:
    """Efficient batch loader for multi-view datasets."""

    def __init__(self, dataset: BaseMultiViewDataset, batch_size: int = 32, 
                 shuffle: bool = False):
        """
        Initialize data loader.
        
        Args:
            dataset: Multi-view dataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle indices
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = len(dataset)
        self.shuffle = shuffle

    def __iter__(self):
        """Iterate over batches."""
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            # Collect views
            batch_views = [[] for _ in range(self.dataset.num_views)]
            batch_labels = []
            batch_sample_indices = []
            
            for idx in batch_indices:
                views, label, sample_idx = self.dataset[idx]
                for v, view in enumerate(views):
                    batch_views[v].append(view)
                batch_labels.append(label)
                batch_sample_indices.append(sample_idx)
            
            # Stack and convert to MLX arrays
            batch_views = [mx.stack(views) for views in batch_views]
            batch_labels = mx.stack(batch_labels)
            batch_indices = mx.stack(batch_sample_indices)
            
            yield batch_views, batch_labels, batch_indices

    def __len__(self) -> int:
        """Return number of batches."""
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def _split_uniform_indices(labels: np.ndarray, 
                           split_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split indices uniformly across classes.
    
    Args:
        labels: Array of labels
        split_ratio: Fraction for training set
        
    Returns:
        Tuple of (train_indices, test_indices)
    """
    unique_labels = np.unique(labels)
    train_idx, test_idx = [], []
    
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        split_point = int(len(label_indices) * split_ratio)
        train_idx.extend(label_indices[:split_point])
        test_idx.extend(label_indices[split_point:])
    
    return np.array(train_idx), np.array(test_idx)


# Dataset registry
DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    'BDGP': DatasetConfig('BDGP', [1750, 79], 2, 5),
    'MNIST-USPS': DatasetConfig('MNIST-USPS', [784, 784], 2, 10),
    'CCV': DatasetConfig('CCV', [5000, 5000, 4000], 3, 20),
    'Fashion': DatasetConfig('Fashion', [784, 784, 784], 3, 10),
    'Caltech-2V': DatasetConfig('Caltech-2V', [40, 254], 2, 7),
    'Caltech-3V': DatasetConfig('Caltech-3V', [40, 254, 928], 3, 7),
    'Caltech-4V': DatasetConfig('Caltech-4V', [40, 254, 928, 256], 4, 7),
    'Caltech-5V': DatasetConfig('Caltech-5V', [40, 254, 928, 256, 243], 5, 7),
}


def load_dataset(dataset_name: str, path: str, train: bool = True) -> Tuple[BaseMultiViewDataset, DatasetConfig]:
    """
    Load dataset by name.
    
    Args:
        dataset_name: Name of dataset
        path: Path to dataset directory
        train: Load training or test set
        
    Returns:
        Tuple of (dataset, config)
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = DATASET_REGISTRY[dataset_name]
    
    if dataset_name == 'BDGP':
        dataset = BDGP(path, train=train)
    elif dataset_name == 'MNIST-USPS':
        dataset = MNIST_USPS(path, train=train)
    elif dataset_name == 'CCV':
        dataset = CCV(path, train=train)
    elif dataset_name == 'Fashion':
        dataset = Fashion(path, train=train)
    elif dataset_name.startswith('Caltech'):
        num_views = int(dataset_name.split('-')[1][0])
        dataset = Caltech(path + 'Caltech-5V.mat', num_views=num_views, train=train)
    else:
        raise ValueError(f"Dataset loading not implemented: {dataset_name}")
    
    config.data_size = len(dataset)
    return dataset, config
