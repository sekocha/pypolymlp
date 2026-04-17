"""Utility functions for input/output results of SSCHA."""

import io
import os

import numpy as np

from pypolymlp.calculator.sscha.sscha_data import SSCHAData
from pypolymlp.calculator.sscha.sscha_params import SSCHAParams
from pypolymlp.calculator.sscha.sscha_utils import symmetrize_properties
from pypolymlp.core.units import EVtoKJmol
from pypolymlp.utils.symfc_utils import compute_projector_cartesian
from pypolymlp.utils.tensor_utils import compute_spg_projector_O2
from pypolymlp.utils.yaml_utils import print_array1d, print_array2d, save_cell


def save_sscha_yaml(
    sscha_params: SSCHAParams,
    sscha_log: list[SSCHAData],
    filename: str = "sscha_results.yaml",
    symmetrize: bool = True,
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
    print("  force:                  eV/angstrom", file=f)
    print("  stress_tensor:          eV/unitcell", file=f)
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
    supercell_matrix = sscha_params.supercell_matrix.astype(int)
    print_array2d(supercell_matrix, "supercell_matrix", f, indent_l=0)
    print(file=f)
    save_cell(sscha_params.supercell, tag="supercell", file=f)

    _print_forces(properties.average_forces, "average_forces", file=f)
    _print_stress(properties.average_stress_tensor, "average_stress_tensor", file=f)

    total_f = properties.average_forces + properties.static_forces
    total_s = properties.average_stress_tensor + properties.static_stress_tensor
    _print_forces(total_f, "total_forces", file=f)
    _print_stress(total_s, "total_stress_tensor", file=f)

    if symmetrize:
        proj_f = compute_projector_cartesian(sscha_params.supercell)
        proj_s = compute_spg_projector_O2(sscha_params.unitcell)

        forces_sym, stress_sym = symmetrize_properties(
            properties.average_forces,
            properties.average_stress_tensor,
            proj_f,
            proj_s,
            sscha_params.n_unitcells,
        )
        _print_forces(forces_sym, "symmetrized_average_forces", file=f)
        _print_stress(stress_sym, "symmetrized_average_stress_tensor", file=f)

        forces_sym, stress_sym = symmetrize_properties(
            total_f,
            total_s,
            proj_f,
            proj_s,
            sscha_params.n_unitcells,
        )
        _print_forces(forces_sym, "symmetrized_total_forces", file=f)
        _print_stress(stress_sym, "symmetrized_total_stress_tensor", file=f)

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


def _print_forces(forces: np.ndarray, tag: str, file: io.IOBase):
    """Print forces."""
    print_array2d(forces.T, tag, file, indent_l=0)
    print(file=file)


def _print_stress(stress: np.ndarray, tag: str, file: io.IOBase):
    """Print stress tensor."""
    sigma = [
        [stress[0], stress[3], stress[5]],
        [stress[3], stress[1], stress[4]],
        [stress[5], stress[4], stress[2]],
    ]
    print_array2d(np.array(sigma), tag, file, indent_l=0)
    print(file=file)
