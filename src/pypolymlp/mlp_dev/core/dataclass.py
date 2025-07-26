"""Dataclass for polymlp model."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PolymlpDataMLP:
    """Dataclass for polymlp model.

    Parameters
    ----------
    coeffs: MLP coefficients, shape=(n_features).
    scales: Scales of x, shape=(n_features).
    """

    coeffs: Optional[np.ndarray] = None
    scales: Optional[np.ndarray] = None
    rmse_train: Optional[float] = None
    rmse_test: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    predictions_train: Optional[np.ndarray] = None
    predictions_test: Optional[np.ndarray] = None
    error_train: Optional[dict] = None
    error_test: Optional[dict] = None
