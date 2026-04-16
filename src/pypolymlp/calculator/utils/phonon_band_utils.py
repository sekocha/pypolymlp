"""Utility functions for phonon band calculations."""

import os
from typing import Literal

import numpy as np
from phonopy import Phonopy
from phonopy.file_IO import parse_BORN
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

from pypolymlp.calculator.utils.phonon_utils import load_phonon
from pypolymlp.utils.phonopy_utils import structure_to_phonopy_cell


def _band_bcc():
    """Define band path and labels."""
    primitive = [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]]
    path = [
        [
            [0, 0, 0],
            [0.5, -0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0, 0, 0],
            [0, 0.0, 0.5],
        ]
    ]
    labels = ["G", "H", "P", "G", "N"]
    return (primitive, path, labels)


def _band_fcc():
    """Define band path and labels."""
    primitive = [[0.5, 0, 0.5], [0.5, 0.5, 0], [0, 0.5, 0.5]]
    path = [
        [[0, 0, 0], [0.5, 0.5, 0]],
        [
            [0.5, 0.5, 1],
            [0.375, 0.375, 0.75],
            [0, 0, 0],
            [0.5, 0.5, 0.5],
        ],
    ]
    labels = ["G", "X", "K", "G", "L"]
    return (primitive, path, labels)


def _band_hcp():
    """Define band path and labels."""
    return (None, None, None)


def _band_rocksalt():
    """Define band path and labels."""
    primitive = [[0.5, 0, 0.5], [0.5, 0.5, 0], [0, 0.5, 0.5]]
    path = [[[0.5, 0.5, 0], [0, 0, 0]], [[0, 0, 0], [0.5, 0.5, 0.5]]]
    labels = ["X", "G", "L"]
    return (primitive, path, labels)


def _band_perovskite():
    """Define band path and labels."""
    primitive = np.eye(3)
    path = [
        [[0, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0], [0, 0, 0]],
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0.5, 0.5]],
    ]
    labels = ["G", "X", "M", "G", "M", "R"]
    return (primitive, path, labels)


def _band_perovskite2():
    """Define band path and labels."""
    primitive = np.eye(3)
    path = [
        [[0, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0], [0, 0, 0]],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]],
    ]
    labels = ["G", "X", "M", "G", "R", "M"]
    return (primitive, path, labels)


def _get_band_attrs(
    structure_type: Literal[
        "bcc", "fcc", "hcp", "rocksalt", "perovskite", "perovskite2"
    ] = "fcc",
):
    """Get band path and labels."""
    if structure_type == "bcc":
        return _band_bcc()
    elif structure_type == "fcc":
        return _band_fcc()
    elif structure_type == "hcp":
        return _band_hcp()
    elif structure_type == "rocksalt":
        return _band_rocksalt()
    elif structure_type == "perovskite":
        return _band_perovskite()
    elif structure_type == "perovskite2":
        return _band_perovskite2()

    raise RuntimeError("Structure type not found.")


def calculate_phonon_bands(
    yamlfile: str = "polymlp_phonon.yaml",
    filefc2: str = "fc2.hdf5",
    fileborn: str = "BORN",
    structure_type: Literal[
        "bcc", "fcc", "hcp", "rocksalt", "perovskite", "perovskite2"
    ] = "fcc",
    npoints: int = 101,
):
    """Calculate band structure from force constants and structure.

    How to use
    ----------
    from pypolymlp.calculator.utils.phonon_band_utils import calculate_phonon_bands

    calculate_phonon_bands(
        yamlfile="polymlp_phonon.yaml",
        filefc2="fc2.hdf5",
        structure_type="perovskite",
    )
    """
    primitive, path, labels = _get_band_attrs(structure_type)
    unitcell, supercell, fc2 = load_phonon(
        yamlfile=yamlfile,
        filefc2=filefc2,
        return_matrix=False,
    )
    unitcell_ph = structure_to_phonopy_cell(unitcell)
    phonopy = Phonopy(
        unitcell_ph,
        supercell_matrix=supercell.supercell_matrix,
        primitive_matrix=primitive,
    )
    phonopy.force_constants = fc2

    if os.path.exists(fileborn):
        nac_params = parse_BORN(phonopy.primitive)
        phonopy = Phonopy(
            unitcell_ph,
            supercell_matrix=supercell.supercell_matrix,
            primitive_matrix=primitive,
        )
        phonopy.nac_params = nac_params
        phonopy.force_constants = fc2

    qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=npoints)
    phonopy.run_band_structure(qpoints, path_connections=connections, labels=labels)
    band_dict = phonopy.get_band_structure_dict()

    phonopy.write_yaml_band_structure(filename="phonon_band.yaml")
    with open("phonon_band.dat", "w") as f:
        for d1, freq in zip(band_dict["distances"], band_dict["frequencies"]):
            for d2, f2 in zip(d1, freq):
                print(d2, *f2, file=f)
