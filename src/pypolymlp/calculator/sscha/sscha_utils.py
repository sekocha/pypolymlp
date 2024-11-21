"""Utility functions for SSCHA."""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml
from phono3py.file_IO import read_fc2_from_hdf5

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.utils import kjmol_to_ev
from pypolymlp.utils.phonopy_utils import phonopy_supercell, structure_to_phonopy_cell


@dataclass
class PolymlpDataSSCHA:
    """Dataclass of sscha results.

    Parameters
    ----------
    temperature: Temperature (K).
    static_potential: Potential energy of equilibrium structure at 0 K.
    harmonic_potential: Harmonic potential energy for effective FC2.
    harmonic_free_energy: Harmonic free energy for effective FC2.
    average_potential: Averaged full potential energy.
    anharmonic_free_energy: Anharmonic free energy,
                            average_potential - harmonic_potential.
    free_energy: Free energy (harmonic_free_energy + anharmonic_free_energy).
    delta: Difference between old FC2 and updated FC2.
    converge: SSCHA calculations are converged or not.
    """

    temperature: float
    static_potential: float
    harmonic_potential: float
    harmonic_free_energy: float
    average_potential: float
    anharmonic_free_energy: float
    free_energy: Optional[float] = None
    delta: Optional[float] = None
    converge: Optional[bool] = None

    def __post_init__(self):
        self.free_energy = self.harmonic_free_energy + self.anharmonic_free_energy


class Restart:
    """Class for reading files to restart SSCHA."""

    def __init__(self, res_yaml: str, fc2hdf5: str = None, unit="kJ/mol"):
        """Init method."""

        self.yaml = res_yaml
        self._unit = unit

        self.load_sscha_yaml()
        if fc2hdf5 is not None:
            self._fc2 = read_fc2_from_hdf5(fc2hdf5)

    def load_sscha_yaml(self):

        yaml_data = yaml.safe_load(open(self.yaml))

        self._pot = yaml_data["parameters"]["pot"]
        self._temp = yaml_data["parameters"]["temperature"]

        self._free_energy = yaml_data["properties"]["free_energy"]
        self._static_potential = yaml_data["properties"]["static_potential"]

        self._delta_fc = yaml_data["status"]["delta_fc"]
        self._converge = yaml_data["status"]["converge"]
        self._logs = yaml_data["logs"]

        unitcell_yaml = yaml_data["unitcell"]
        unitcell_yaml["axis"] = np.array(unitcell_yaml["axis"]).T
        unitcell_yaml["positions"] = np.array(unitcell_yaml["positions"]).T
        self._unitcell = PolymlpStructure(**unitcell_yaml)
        self._supercell_matrix = np.array(yaml_data["supercell_matrix"])
        self._n_atom_unitcell = len(self._unitcell.elements)

    @property
    def polymlp(self):
        return self._pot

    @property
    def temperature(self):
        return self._temp

    @property
    def free_energy(self):
        if self._unit == "kJ/mol":
            return self._free_energy
        elif self._unit == "eV/atom":
            return kjmol_to_ev(self._free_energy) / self._n_atom_unitcell
        raise ValueError("Energy unit: kJ/mol or eV/atom")

    @property
    def static_potential(self):
        if self._unit == "kJ/mol":
            return self._static_potential
        elif self._unit == "eV/atom":
            return kjmol_to_ev(self._static_potential) / self._n_atom_unitcell
        raise ValueError("Energy unit: kJ/mol or eV/atom")

    @property
    def logs(self):
        return self._logs

    @property
    def force_constants(self):
        return self._fc2

    @property
    def unitcell(self):
        return self._unitcell

    @property
    def unitcell_phonopy(self):
        return structure_to_phonopy_cell(self._unitcell)

    @property
    def supercell_matrix(self):
        return self._supercell_matrix

    @property
    def n_unitcells(self):
        return int(round(np.linalg.det(self._supercell_matrix)))

    @property
    def supercell(self):
        cell = phonopy_supercell(
            self._unitcell,
            supercell_matrix=self._supercell_matrix,
            return_phonopy=False,
        )
        return cell

    @property
    def supercell_phonopy(self):
        cell = phonopy_supercell(
            self._unitcell,
            supercell_matrix=self._supercell_matrix,
            return_phonopy=True,
        )
        return cell

    @property
    def unitcell_volume(self):
        volume = np.linalg.det(self._unitcell["axis"])
        if self._unit == "kJ/mol":
            return volume
        elif self._unit == "eV/atom":
            return volume / self._n_atom_unitcell
        raise ValueError("Energy unit: kJ/mol or eV/atom")


def temperature_setting(args):

    if args.temp is not None:
        if np.isclose(args.temp, round(args.temp)):
            args.temp = int(args.temp)
        temp_array = [args.temp]
    else:
        if np.isclose(args.temp_min, round(args.temp_min)):
            args.temp_min = int(args.temp_min)
        if np.isclose(args.temp_max, round(args.temp_max)):
            args.temp_max = int(args.temp_max)
        if np.isclose(args.temp_step, round(args.temp_step)):
            args.temp_step = int(args.temp_step)
        temp_array = np.arange(args.temp_min, args.temp_max + 1, args.temp_step)
        if args.ascending_temp == False:
            temp_array = temp_array[::-1]
    args.temperatures = temp_array
    return args


def n_samples_setting(args, n_atom_supercell=None):

    if args.n_samples is None:
        n_samples_unit = round(6400 / n_atom_supercell)
        args.n_samples_init = 20 * n_samples_unit
        args.n_samples_final = 100 * n_samples_unit
    else:
        args.n_samples_init, args.n_samples_final = args.n_samples
    return args


def print_structure(cell: PolymlpStructure):
    """Print structure."""

    print(" # structure ")
    print("  - elements:     ", cell.elements)
    print("  - axis:         ", cell.axis.T[0])
    print("                  ", cell.axis.T[1])
    print("                  ", cell.axis.T[2])
    print("  - positions:    ", cell.positions.T[0])
    if cell.positions.shape[1] > 1:
        for pos in cell.positions.T[1:]:
            print("                  ", pos)


def print_parameters(supercell_matrix: np.ndarray, args):
    """Print parameters in SSCHA."""

    print(" # parameters")
    print("  - supercell:    ", supercell_matrix[0])
    print("                  ", supercell_matrix[1])
    print("                  ", supercell_matrix[2])
    print("  - temperatures: ", args.temperatures[0])
    if len(args.temperatures) > 1:
        for t in args.temperatures[1:]:
            print("                  ", t)

    if isinstance(args.pot, list):
        for p in args.pot:
            print("  - Polynomial ML potential:  ", os.path.abspath(p))
    else:
        print("  - Polynomial ML potential:  ", os.path.abspath(args.pot))

    print("  - FC tolerance:             ", args.tol)
    print("  - max iter:                 ", args.max_iter)
    print("  - num samples:              ", args.n_samples_init)
    print("  - num samples (last iter.): ", args.n_samples_final)
    print("  - q-mesh:                   ", args.mesh)


def print_array1d(array, tag, fstream, indent_l=0):
    prefix = "".join([" " for n in range(indent_l)])
    print(prefix + tag + ":", file=fstream)
    for i, d in enumerate(array):
        print(prefix + " -", d, file=fstream)


def print_array2d(array, tag, fstream, indent_l=0):
    prefix = "".join([" " for n in range(indent_l)])
    print(prefix + tag + ":", file=fstream)
    for i, d in enumerate(array):
        print(prefix + " -", list(d), file=fstream)


def save_cell(cell: PolymlpStructure, tag="unitcell", fstream=None, filename=None):
    """Write structure to a file."""

    np.set_printoptions(legacy="1.25")
    if fstream is None:
        fstream = open(filename, "w")

    print(tag + ":", file=fstream)
    print_array2d(cell.axis.T, "axis", fstream, indent_l=2)
    print_array2d(cell.positions.T, "positions", fstream, indent_l=2)
    print("  n_atoms:  ", list(cell.n_atoms), file=fstream)
    print("  types:    ", list(cell.types), file=fstream)
    print("  elements: ", list(cell.elements), file=fstream)

    if tag == "supercell":
        print("  n_unitcells: ", cell.n_unitcells, file=fstream)
        print_array2d(
            cell.supercell_matrix,
            "supercell_matrix",
            fstream,
            indent_l=2,
        )

    print("", file=fstream)


def save_sscha_yaml(
    unitcell: PolymlpStructure,
    supercell_matrix: np.ndarray,
    sscha_log: list[PolymlpDataSSCHA],
    args,
    filename="sscha_results.yaml",
):
    """Write SSCHA results to a file."""

    np.set_printoptions(legacy="1.25")
    properties = sscha_log[-1]

    f = open(filename, "w")
    print("parameters:", file=f)
    if isinstance(args.pot, list):
        for p in args.pot:
            print("  pot:     ", os.path.abspath(p), file=f)
    else:
        print("  pot:     ", os.path.abspath(args.pot), file=f)
    print("  temperature:   ", properties.temperature, file=f)
    print("  n_steps:       ", args.n_samples_init, file=f)
    print("  n_steps_final: ", args.n_samples_final, file=f)
    print("  tolerance:     ", args.tol, file=f)
    print("  mixing:        ", args.mixing, file=f)
    print("  mesh_phonon:   ", list(args.mesh), file=f)
    print("", file=f)

    print("properties:", file=f)
    print("  free_energy:     ", properties.free_energy, file=f)
    print("  static_potential:", properties.static_potential, file=f)

    print("status:", file=f)
    print("  delta_fc: ", properties.delta, file=f)
    print("  converge: ", properties.converge, file=f)
    print("", file=f)

    save_cell(unitcell, tag="unitcell", fstream=f)
    print("supercell_matrix:", file=f)
    print(" -", list(supercell_matrix[0].astype(int)), file=f)
    print(" -", list(supercell_matrix[1].astype(int)), file=f)
    print(" -", list(supercell_matrix[2].astype(int)), file=f)
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
