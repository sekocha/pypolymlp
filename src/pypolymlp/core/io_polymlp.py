"""Class for saving and loading polymlp.yaml files"""

import glob
import io
from typing import Union

import numpy as np

from pypolymlp.core.data_format import PolymlpParams
from pypolymlp.core.io_polymlp_legacy import load_mlp_lammps
from pypolymlp.core.io_polymlp_yaml import load_mlp_yaml, save_mlp_yaml


def save_mlp(
    params: PolymlpParams,
    coeffs: np.ndarray,
    scales: np.ndarray,
    filename: bool = "polymlp.yaml",
):
    """Generate polymlp.yaml file for single polymlp model."""
    save_mlp_yaml(params, coeffs, scales, filename=filename)


def save_mlps(
    multiple_params: list[PolymlpParams],
    cumulative_n_features: list[int],
    coeffs: np.ndarray,
    scales: np.ndarray,
    prefix: str = "polymlp.yaml",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate polymlp.yaml files for hybrid polymlp model."""
    multiple_coeffs = []
    multiple_scales = []
    for i, params in enumerate(multiple_params):
        if i == 0:
            begin, end = 0, cumulative_n_features[0]
        else:
            begin = cumulative_n_features[i - 1]
            end = cumulative_n_features[i]

        filename = prefix + "." + str(i + 1)
        save_mlp(params, coeffs[begin:end], scales[begin:end], filename=filename)
        multiple_coeffs.append(coeffs[begin:end])
        multiple_scales.append(scales[begin:end])

    return multiple_coeffs, multiple_scales


def load_mlp(filename: Union[str, io.IOBase] = "polymlp.yaml"):
    """Load single polymlp file.

    Return
    ------
    params: Parameters in PolymlpParams.
    coeffs: polymlp model coefficients.

    Usage
    -----
    params, coeffs = load_mlp(filename='polymlp.yaml')
    """

    if not isinstance(filename, io.IOBase):
        filename = open(filename)

    line = filename.readline()
    legacy = True if "# ele" in line else False

    filename.seek(0)
    if legacy:
        params, coeffs = load_mlp_lammps(filename)
    else:
        params, coeffs = load_mlp_yaml(filename)

    return params, coeffs


def load_mlps(file_list_or_file):
    """Load polymlp files.

    Return
    ------
    params_array: List of parameters in PolymlpParams.
    coeffs_array: List of polymlp model coefficients.
    """

    if isinstance(file_list_or_file, list):
        if len(file_list_or_file) == 1:
            return load_mlp(file_list_or_file[0])

        params_array, coeffs_array = [], []
        for pot in file_list_or_file:
            params, coeffs = load_mlp(pot)
            params_array.append(params)
            coeffs_array.append(coeffs)
        return params_array, coeffs_array

    return load_mlp(file_list_or_file)


def find_mlps(path: str):
    """Find polymlp files in directory."""
    files = glob.glob(path + "/polymlp.yaml*")
    if len(files) > 0:
        return sorted(files)

    files = glob.glob(path + "/polymlp.lammps*")
    if len(files) > 0:
        return sorted(files)

    return None


def convert_to_yaml(
    txt: Union[str, io.IOBase] = "polymlp.lammps",
    yaml: str = "polymlp.yaml",
):
    """Convert text format to yaml format."""
    params, coeffs = load_mlp(txt)
    save_mlp(params, coeffs, np.ones(len(coeffs)), filename=yaml)
