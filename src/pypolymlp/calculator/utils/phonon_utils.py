"""Utility functions for phonon calculations."""

import numpy as np


def is_imaginary(
    frequencies: np.ndarray,
    dos: np.ndarray,
    tol_frequency: float = -0.01,
):
    """Check if phonon DOS has imaginary frequencies."""
    frequencies = np.array(frequencies)
    dos = np.array(dos)
    is_imag = frequencies < tol_frequency
    return np.sum(dos[is_imag]) > 1e-6
