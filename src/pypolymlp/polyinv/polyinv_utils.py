"""Utility functions for enumerating polynomial invariants."""

import itertools

import numpy as np

# from numpy.typing import NDArray
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


# def get_m_combs(lcomb: list | NDArray, lproj: int = 0):
#    """Calculate all combinations of m values.
#
#    m_array = [range(-l,l+1) for l in lcomb]
#    m_all = [mcomb for mcomb in itertools.product(*m_array)
#             if abs(sum(mcomb)) <= lproj]
#    """
#    if lproj > 0:
#        return []
#
#    m_array = []
#    for l1 in lcomb[:-1]:
#        m_array.append(range(-l1, l1 + 1))
#        m_array.append(range(-l1, l1 + 1))
#    m_all = list(itertools.product(*m_array))
#    return m_all
#
#

# def get_m_combs(lcomb: list | NDArray, lproj: int = 0):
#     """Calculate all combinations of m values.
#
#     m_array = [range(-l,l+1) for l in lcomb]
#    m_all = [mcomb for mcomb in itertools.product(*m_array)
#             if abs(sum(mcomb)) <= lproj]
#    """
#    if lproj > 0:
#        raise RuntimeError("This function is available only for lproj=0.")
#
#    m_array = []
#    for l1 in lcomb[:-1]:
#        m_array.append(range(-l1, l1 + 1))
#        m_array.append(range(-l1, l1 + 1))
#
#    m_all = []
#    for mcomb in itertools.product(*m_array):
#        mcomb1 = check_m_nonzero(lcomb, mcomb)
#        if mcomb1 is not None:
#            m_all.append(mcomb1)
#    return m_all
#

# def check_m_nonzero(lcomb: list, mcomb: list):
#    """Check whether combination of m values shows nonzero elements in projector."""
#    mv1 = list(mcomb[::2])
#    mf = -sum(mv1)
#    if abs(mf) > lcomb[-1]:
#        return None
#
#    mv2 = list(mcomb[1::2])
#    mfp = -sum(mv2)
#    if abs(mfp) > lcomb[-1]:
#        return None
#
#    mv1.append(mf)
#    mv2.append(mfp)
#    index = lm_to_matrix_index(lcomb, mv1)
#    index_p = lm_to_matrix_index(lcomb, mv2)
#    if index <= index_p:
#        mv1.extend(mv2)
#        return np.array(mv1)
#    return None


def lm_to_matrix_index(l_list: list, m_list: list):
    """Convert (l, m) into sequential index."""
    l_plus_m_list = np.array(l_list) + np.array(m_list)
    l_list2 = 2 * np.array(l_list) + 1

    index = 0
    for i, lpm in enumerate(l_plus_m_list):
        index += lpm * np.prod(l_list2[i + 1 :])
    return index


def matrix_index_to_lm(index: int, l_list: list):
    """Convert sequential index into (l, m)."""
    n, i_list = index, []
    for l1 in reversed(l_list):
        i_list.append(n % (2 * l1 + 1))
        n = n // (2 * l1 + 1)
    i_list = reversed(i_list)
    return np.array([(l, i - l) for i, l in zip(i_list, l_list)])
