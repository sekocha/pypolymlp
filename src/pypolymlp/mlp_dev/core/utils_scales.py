"""Functions for applying scales."""

import numpy as np


def round_scales(
    scales: np.ndarray, include_force: bool = True, threshold: float = 1e-10
):
    """Set scales so that they are not used for zero features."""
    if include_force:
        zero_ids = np.abs(scales) < threshold
    else:
        # Threshold value can be improved.
        zero_ids = np.abs(scales) < threshold * threshold
    scales[zero_ids] = 1.0
    return scales, zero_ids


def compute_scales(
    scales: np.array,
    xe_sum: np.ndarray,
    xe_sq_sum: np.ndarray,
    n_data: int,
    include_force: bool = True,
):
    """Compute scales from xe_sum and xe_sq_sum."""
    if scales is None:
        variance = xe_sq_sum / n_data - np.square(xe_sum / n_data)
        variance[variance < 0.0] = 1.0
        scales = np.sqrt(variance)

    scales, zero_ids = round_scales(scales, include_force=include_force)
    return scales, zero_ids
