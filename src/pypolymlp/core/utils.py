"""Utility functions"""

import os

import numpy as np

from pypolymlp._version import __version__


def strtobool(val):
    """Convert a string to bool, True or False."""
    val = val.lower()
    if val in ("t", "true", "1"):
        return True
    elif val in ("f", "false", "0"):
        return False
    raise RuntimeError("Invalid string.")


def split_ids_train_test(n_data: int, train_ratio: float = 0.9):
    """Split dataset into training and test datasets."""
    n_train = round(n_data * train_ratio)
    n_test = n_data - n_train
    test_ids = np.round(np.linspace(0, n_data - 1, num=n_test)).astype(int)

    train_bools = np.ones(n_data, dtype=bool)
    train_bools[test_ids] = False
    train_ids = np.where(train_bools)[0]
    return train_ids, test_ids


def split_train_test(files: list, train_ratio: float = 0.9):
    """Split dataset into training and test datasets."""
    n_data = len(files)
    train_ids, test_ids = split_ids_train_test(n_data, train_ratio=train_ratio)
    return [files[i] for i in train_ids], [files[i] for i in test_ids]


def check_memory_size_in_regression(n_features: int):
    """Estimate memory size in regression."""
    mem_req = np.round(n_features**2 * 8e-9 * 2, 1)
    mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") * 1e-9
    if mem_req > mem_bytes:
        print("Minimum memory required for solver in GB:", mem_req, flush=True)
        raise RuntimeError("Larger size of memory required.")
    return mem_req


def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute root mean square errors."""
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def mass_table():
    """Get mass dictionary."""

    mass_table = {
        "H": 1.008,
        "He": 4.003,
        "Li": 6.941,
        "Be": 9.012,
        "B": 10.81,
        "C": 12.01,
        "N": 14.01,
        "O": 16.00,
        "F": 19.00,
        "Ne": 20.18,
        "Na": 22.99,
        "Mg": 24.31,
        "Al": 26.98,
        "Si": 28.09,
        "P": 30.97,
        "S": 32.07,
        "Cl": 35.45,
        "Ar": 39.95,
        "K": 39.10,
        "Ca": 40.08,
        "Sc": 44.96,
        "Ti": 47.88,
        "V": 50.94,
        "Cr": 52.00,
        "Mn": 54.94,
        "Fe": 55.85,
        "Co": 58.93,
        "Ni": 58.69,
        "Cu": 63.55,
        "Zn": 65.39,
        "Ga": 69.72,
        "Ge": 72.61,
        "As": 74.92,
        "Se": 78.96,
        "Br": 79.90,
        "Kr": 83.80,
        "Rb": 85.47,
        "Sr": 87.62,
        "Y": 88.91,
        "Zr": 91.22,
        "Nb": 92.91,
        "Mo": 95.94,
        "Tc": 99,
        "Ru": 101.1,
        "Rh": 102.9,
        "Pd": 106.4,
        "Ag": 107.9,
        "Cd": 112.4,
        "In": 114.8,
        "Sn": 118.7,
        "Sb": 121.8,
        "Te": 127.6,
        "I": 126.9,
        "Xe": 131.3,
        "Cs": 132.9,
        "Ba": 137.3,
        "La": 138.9,
        "Ce": 140.1,
        "Pr": 140.9,
        "Nd": 144.2,
        "Pm": 145,
        "Sm": 150.4,
        "Eu": 152.0,
        "Gd": 157.3,
        "Tb": 158.9,
        "Dy": 162.5,
        "Ho": 164.9,
        "Er": 167.3,
        "Tm": 168.9,
        "Yb": 173.0,
        "Lu": 175.0,
        "Hf": 178.5,
        "Ta": 180.9,
        "W": 183.8,
        "Re": 186.2,
        "Os": 190.2,
        "Ir": 192.2,
        "Pt": 195.1,
        "Au": 197.0,
        "Hg": 200.6,
        "Tl": 204.4,
        "Pb": 207.2,
        "Bi": 209.0,
        "Po": 210,
        "At": 210,
        "Rn": 222,
        "Fr": 223,
        "Ra": 226,
        "Ac": 227,
        "Th": 232.0,
        "Pa": 231.0,
        "U": 238.0,
        "Np": 237,
        "Pu": 239,
        "Am": 243,
        "Cm": 247,
        "Bk": 247,
        "Cf": 252,
        "Es": 252,
        "Fm": 257,
        "Md": 256,
        "No": 259,
        "Lr": 260,
    }

    return mass_table


def precision(x, alpha=0.0001):

    # std = np.std(x[:50], axis=0)
    # for col, val in enumerate(std):
    #    if abs(val) > 1e-15:
    #        x[:,col] /= val

    prod = x.T @ x
    for i in range(x.shape[1]):
        prod[i, i] += alpha

    var = np.linalg.inv(prod)
    # ave = np.average(x, axis=0)
    # dx = x - ave
    prec = np.mean([x1.T @ var @ x1 for x1 in x])
    return prec


def print_credit():
    """Print credit of pypolymlp."""
    print("Pypolymlp", "version", __version__, flush=True)
    print("  polynomial machine learning potential:", flush=True)
    print("  A. Seko, J. Appl. Phys. 133, 011101 (2023)", flush=True)
