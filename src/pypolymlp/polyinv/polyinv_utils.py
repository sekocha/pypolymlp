"""Utility functions for enumerating polynomial invariants."""

import argparse
import itertools
import os

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad


def get_l_combs(maxl: int, order: int, lproj: int = 0):
    """Calculate all combinations of l values."""
    lcomb_all = itertools.combinations_with_replacement(range(0, maxl + 1), order)
    lcomb_all = list(lcomb_all)
    n_list = [num_basis(lcomb, lproj) for lcomb in lcomb_all]
    return (np.array(lcomb_all), np.array(n_list))


def num_basis(lcomb: list, lproj: int = 0):
    """Number of basis vectors calculated from characters."""
    num, _ = quad(
        lambda x: chi(x, lproj, 1) * chi_prod(x, lcomb) * np.sin(x / 2) ** 2,
        0,
        2 * np.pi,
    )
    return int(round(num / np.pi))


def chi(x: float, l: int, n: int):
    """Calculate characters."""
    return np.sin((2 * l + 1) * n * x / 2) / np.sin(x * n / 2)


def chi_prod(x: float, lcomb):
    """Calculate product of characters."""
    return np.prod([chi(x, l, 1) for l in lcomb])

 
def get_m_combs(lcomb: list | NDArray, lproj: int = 0):
    """Calculate all combinations of m values.

    m_array = [range(-l,l+1) for l in lcomb]
    m_all = [mcomb for mcomb in itertools.product(*m_array)
             if abs(sum(mcomb)) <= lproj]
    """
    if lproj > 0:
        return []

    m_array = []
    for l1 in lcomb[:-1]:
        m_array.append(range(-l1, l1 + 1))
        m_array.append(range(-l1, l1 + 1))
    m_all = list(itertools.product(*m_array))
    return m_all
