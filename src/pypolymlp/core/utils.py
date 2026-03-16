"""Utility functions"""

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


def get_atomic_size_scales():
    """Return scale of atomic size.

    Atomic radius is normalized by that of element Ti.
    """
    atomic_radius = {
        "H": 0.53,
        "He": 0.31,
        "Li": 1.52,
        "Be": 1.12,
        "B": 0.85,
        "C": 0.70,
        "N": 0.65,
        "O": 0.60,
        "F": 0.50,
        "Ne": 0.38,
        "Na": 1.86,
        "Mg": 1.60,
        "Al": 1.43,
        "Si": 1.17,
        "P": 1.06,
        "S": 1.02,
        "Cl": 0.99,
        "Ar": 0.71,
        "K": 2.03,
        "Ca": 1.74,
        "Sc": 1.44,
        "Ti": 1.32,
        "V": 1.22,
        "Cr": 1.18,
        "Mn": 1.17,
        "Fe": 1.16,
        "Co": 1.11,
        "Ni": 1.10,
        "Cu": 1.28,
        "Zn": 1.33,
        "Ga": 1.26,
        "Ge": 1.22,
        "As": 1.20,
        "Se": 1.16,
        "Br": 1.14,
        "Kr": 1.03,
        "Rb": 2.16,
        "Sr": 1.91,
        "Y": 1.62,
        "Zr": 1.45,
        "Nb": 1.34,
        "Mo": 1.30,
        "Tc": 1.28,
        "Ru": 1.25,
        "Rh": 1.25,
        "Pd": 1.28,
        "Ag": 1.44,
        "Cd": 1.48,
        "In": 1.56,
        "Sn": 1.45,
        "Sb": 1.40,
        "Te": 1.36,
        "I": 1.33,
        "Xe": 1.31,
        "Cs": 2.35,
        "Ba": 1.98,
        "La": 1.87,
        "Ce": 1.82,
        "Pr": 1.82,
        "Nd": 1.82,
        "Pm": 1.82,
        "Sm": 1.80,
        "Eu": 1.85,
        "Gd": 1.80,
        "Tb": 1.79,
        "Dy": 1.78,
        "Ho": 1.78,
        "Er": 1.77,
        "Tm": 1.76,
        "Yb": 1.94,
        "Lu": 1.73,
        "Hf": 1.44,
        "Ta": 1.34,
        "W": 1.30,
        "Re": 1.28,
        "Os": 1.25,
        "Ir": 1.23,
        "Pt": 1.23,
        "Au": 1.44,
        "Hg": 1.49,
        "Tl": 1.56,
        "Pb": 1.46,
        "Bi": 1.48,
        "Po": 1.40,
        "At": 1.50,
        "Rn": 1.50,
        "Fr": 2.60,
        "Ra": 2.21,
    }
    for k, v in atomic_radius.items():
        atomic_radius[k] = v / 1.32
    return atomic_radius


def precision(x, alpha=0.0001):

    # std = np.std(x[:50], axis=0)
    # for col, val in enumerate(std):
    #    if abs(val) > 1e-15:
    #        x[:,col] /= val

    prod = x.T @ x
    for i in range(x.shape[1]):
        prod[i, i] += alpha

    var = np.linalg.inv(prod)
    prec = np.mean([x1.T @ var @ x1 for x1 in x])
    return prec


def print_credit():
    """Print credit of pypolymlp."""
    print("Pypolymlp", "version", __version__, flush=True)
    print("  Polynomial machine learning potential:", flush=True)
    print("  A. Seko, J. Appl. Phys. 133, 011101 (2023)", flush=True)
