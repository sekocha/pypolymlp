"""Class for saving and loading polymlp.yaml files"""

import glob
import io
from typing import Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpParamsSingle
from pypolymlp.core.io_polymlp_legacy import load_mlp_lammps
from pypolymlp.core.io_polymlp_yaml import load_mlp_yaml, save_mlp_yaml
from pypolymlp.core.params import PolymlpParams


def save_mlp(
    params: PolymlpParamsSingle,
    coeffs: np.ndarray,
    scales: np.ndarray,
    filename: bool = "polymlp.yaml",
):
    """Generate polymlp.yaml file for single polymlp model."""
    save_mlp_yaml(params, coeffs, scales, filename=filename)


def save_mlps(
    params: PolymlpParams,
    coeffs: np.ndarray,
    scales: np.ndarray,
    cumulative_n_features: Optional[list[int]] = None,
    filename: str = "polymlp.yaml",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate polymlp.yaml files for hybrid polymlp model."""
    if not params.is_hybrid:
        save_mlp(params.params, coeffs, scales, filename=filename)
        return [coeffs], [scales]

    multiple_coeffs, multiple_scales = [], []
    for i, params_single in enumerate(params):
        begin = 0 if i == 0 else cumulative_n_features[i - 1]
        end = cumulative_n_features[i]
        filename_rev = filename + "." + str(i + 1)
        save_mlp(
            params_single,
            coeffs[begin:end],
            scales[begin:end],
            filename=filename_rev,
        )
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

    legacy = is_legacy(filename)
    if legacy:
        params_single, coeffs = load_mlp_lammps(filename)
    else:
        params_single, coeffs = load_mlp_yaml(filename)
    return params_single, coeffs


def load_mlps(file_list_or_file):
    """Load polymlp files.

    Return
    ------
    params_array: PolymlpParams.
    coeffs_array: List of polymlp model coefficients.
    """
    if isinstance(file_list_or_file, str):
        params_single, coeffs = load_mlp(file_list_or_file)
        return PolymlpParams(params_single), coeffs

    if isinstance(file_list_or_file, (list, tuple, np.ndarray)):
        if len(file_list_or_file) == 1:
            params_single, coeffs = load_mlp(file_list_or_file[0])
            return PolymlpParams(params_single), coeffs

        params, coeffs_array = PolymlpParams(), []
        for pot in file_list_or_file:
            params_single, coeffs = load_mlp(pot)
            params.append(params_single)
            coeffs_array.append(coeffs)
        return params, coeffs_array

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
    if isinstance(txt, str):
        if not is_legacy(txt):
            return False
        params, coeffs = load_mlp(txt)
        save_mlp(params, coeffs, np.ones(len(coeffs)), filename=yaml)
        return True
    for i, file1 in enumerate(sorted(txt)):
        if not is_legacy(file1):
            return False
        params, coeffs = load_mlp(file1)
        if len(txt) > 1:
            filename = yaml + "." + str(i + 1)
        else:
            filename = yaml
        save_mlp(params, coeffs, np.ones(len(coeffs)), filename=filename)
    return True


def is_legacy(filename: Union[str, io.IOBase] = "polymlp.yaml"):
    """Check if MLP is a legacy one.

    Return
    ------
    legacy: If MLP is a legacy one, return True.
    """
    if not isinstance(filename, io.IOBase):
        filename = open(filename)

    line = filename.readline()
    legacy = True if "# ele" in line else False
    filename.seek(0)
    return legacy


def is_hybrid(filename: Union[str, io.IOBase, list] = "polymlp.yaml"):
    """Check if MLP is a hybrid one.

    Return
    ------
    legacy: If MLP is a hybrid-type one, return True.
    """
    if isinstance(filename, io.IOBase):
        return False
    if isinstance(filename, str):
        return False
    if isinstance(filename, (list, tuple, np.ndarray)):
        if len(filename) == 1:
            return False
        return True
    else:
        raise RuntimeError("filename must be strings or array-type.")
