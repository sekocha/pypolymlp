"""Utility functions for sequential regression."""

import os


def get_batch_slice(n_data, batch_size):
    """Calculate slice indices for a given batch size."""
    begin_batch = list(range(0, n_data, batch_size))
    if len(begin_batch) > 1:
        end_batch = list(begin_batch[1:]) + [n_data]
    else:
        end_batch = [n_data]
    return begin_batch, end_batch


def get_auto_batch_size(n_features: int, verbose: bool = False):
    """Return optimal batch size determined automatically."""
    mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    # Allocate of X.T @ X
    mem_bytes = mem_bytes * 0.9 - (n_features**2 * 8)
    # Allocate X of size (n_str * N3, n_features)
    N3, max_mem = 300, 1e11
    batch_size = round(min(mem_bytes, max_mem) / (N3 * n_features * 8))
    return batch_size
