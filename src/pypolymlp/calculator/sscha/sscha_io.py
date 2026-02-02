"""Utility functions for input/output results of SSCHA."""

import os

import numpy as np

from pypolymlp.calculator.sscha.sscha_data import SSCHAData
from pypolymlp.calculator.sscha.sscha_params import SSCHAParams
from pypolymlp.core.units import EVtoKJmol
from pypolymlp.utils.yaml_utils import print_array1d, print_array2d, save_cell


def save_sscha_yaml(
    sscha_params: SSCHAParams,
    sscha_log: list[SSCHAData],
    filename="sscha_results.yaml",
):
    """Write SSCHA results to a file."""

    np.set_printoptions(legacy="1.21")
    properties = sscha_log[-1]

    f = open(filename, "w")
    print("parameters:", file=f)
    if isinstance(sscha_params.pot, list):
        pots = [os.path.abspath(p) for p in sscha_params.pot]
        print("  pot:     ", pots, file=f)
    else:
        print("  pot:     ", os.path.abspath(sscha_params.pot), file=f)

    print("  temperature:   ", properties.temperature, file=f)
    print("  n_steps:       ", sscha_params.n_samples_init, file=f)
    print("  n_steps_final: ", sscha_params.n_samples_final, file=f)
    print("  tolerance:     ", sscha_params.tol, file=f)
    print("  mixing:        ", sscha_params.mixing, file=f)
    print("  mesh_phonon:   ", list(sscha_params.mesh), file=f)
    print("", file=f)

    print("units:", file=f)
    print("  free_energy:            kJ/mol", file=f)
    print("  static_potential:       kJ/mol", file=f)
    print("  entropy:                J/K/mol", file=f)
    print("  harmonic_heat_capacity: J/K/mol", file=f)
    print("", file=f)

    print("properties:", file=f)
    print("  free_energy:           ", properties.free_energy, file=f)
    print("  harmonic_free_energy:  ", properties.harmonic_free_energy, file=f)
    print("  anharmonic_free_energy:", properties.anharmonic_free_energy, file=f)
    print("  static_potential:      ", properties.static_potential, file=f)
    print("  entropy:               ", properties.entropy, file=f)
    print("  harmonic_heat_capacity:", properties.harmonic_heat_capacity, file=f)
    print("", file=f)

    print("properties_eV:", file=f)
    val = properties.free_energy / EVtoKJmol
    print("  free_energy:           ", val, file=f)
    val = properties.harmonic_free_energy / EVtoKJmol
    print("  harmonic_free_energy:  ", val, file=f)
    val = properties.anharmonic_free_energy / EVtoKJmol
    print("  anharmonic_free_energy:", val, file=f)
    val = properties.static_potential / EVtoKJmol
    print("  static_potential:      ", val, file=f)
    print("", file=f)

    print("status:", file=f)
    print("  delta_fc:  ", properties.delta, file=f)
    print("  converge:  ", properties.converge, file=f)
    print("  imaginary: ", properties.imaginary, file=f)
    print("", file=f)

    save_cell(sscha_params.unitcell, tag="unitcell", file=f)
    print("supercell_matrix:", file=f)
    print(" -", list(sscha_params.supercell_matrix[0].astype(int)), file=f)
    print(" -", list(sscha_params.supercell_matrix[1].astype(int)), file=f)
    print(" -", list(sscha_params.supercell_matrix[2].astype(int)), file=f)
    print("", file=f)
    save_cell(sscha_params.supercell, tag="supercell", file=f)

    print_array2d(properties.average_forces.T, "average_forces", f, indent_l=0)
    print("", file=f)

    print("logs:", file=f)
    print_array1d([log.free_energy for log in sscha_log], "free_energy", f, indent_l=2)
    print("", file=f)

    array = [log.harmonic_potential for log in sscha_log]
    print_array1d(array, "harmonic_potential", f, indent_l=2)
    print("", file=f)

    array = [log.average_potential for log in sscha_log]
    print_array1d(array, "average_potential", f, indent_l=2)
    print("", file=f)

    array = [log.anharmonic_free_energy for log in sscha_log]
    print_array1d(array, "anharmonic_free_energy", f, indent_l=2)
    print("", file=f)

    f.close()
