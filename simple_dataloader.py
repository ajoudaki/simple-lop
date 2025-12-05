"""
Simple DataLoader implementation using NumPy for mini-batch training.
"""
import numpy as np


class SimpleDataLoader:
    """
    A simple DataLoader that iterates over a dataset in mini-batches.
    
    Args:
        *arrays: Variable number of numpy arrays or torch tensors to iterate over.
                 All arrays must have the same length along the first dimension.
        batch_size: Size of each mini-batch.
        shuffle: Whether to shuffle the data at the start of each epoch.
        rng: NumPy random generator for reproducible shuffling. If None, uses default.
    """
    def __init__(self, *arrays, batch_size, shuffle=True, rng=None):
        if len(arrays) == 0:
            raise ValueError("At least one array must be provided")
        
        self.arrays = arrays
        self.n = len(arrays[0])
        
        # Validate all arrays have the same length
        for arr in arrays:
            if len(arr) != self.n:
                raise ValueError("All arrays must have the same length")
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = rng if rng is not None else np.random.default_rng()
        
    def __iter__(self):
        """Returns an iterator over mini-batches."""
        if self.shuffle:
            indices = self.rng.permutation(self.n)
        else:
            indices = np.arange(self.n)
        
        for start_idx in range(0, self.n, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n)
            batch_indices = indices[start_idx:end_idx]
            
            # Yield tuple of batched arrays
            yield tuple(arr[batch_indices] for arr in self.arrays)
    
    def __len__(self):
        """Returns the number of batches per epoch."""
        return (self.n + self.batch_size - 1) // self.batch_size
