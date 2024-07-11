"""Utility functions for sequential regression."""


def get_batch_slice(n_data, batch_size):
    """Calculate slice indices for a given batch size."""
    begin_batch = list(range(0, n_data, batch_size))
    if len(begin_batch) > 1:
        end_batch = list(begin_batch[1:]) + [n_data]
    else:
        end_batch = [n_data]
    return begin_batch, end_batch
