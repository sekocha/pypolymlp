"""Utility functions for SSCHA."""

import os
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import yaml
from phono3py.file_IO import read_fc2_from_hdf5

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_yaml import parse_structure_from_yaml
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
    entropy: Entropy for effective FC2.
    harmonic_heat_capacity: Harmonic heat capacity for effective FC2.
    free_energy: Free energy (harmonic_free_energy + anharmonic_free_energy).
    delta: Difference between old FC2 and updated FC2.
    converge: SSCHA calculations are converged or not.
    imaginary: Imaginary frequencies are found or not.
    """

    temperature: float
    static_potential: float
    harmonic_potential: float
    harmonic_free_energy: float
    average_potential: float
    anharmonic_free_energy: float
    entropy: Optional[float] = None
    harmonic_heat_capacity: Optional[float] = None
    free_energy: Optional[float] = None
    delta: Optional[float] = None
    converge: Optional[bool] = None
    imaginary: Optional[bool] = None

    def __post_init__(self):
        self.free_energy = self.harmonic_free_energy + self.anharmonic_free_energy


class Restart:
    """Class for reading files to restart SSCHA."""

    def __init__(
        self,
        res_yaml: str,
        fc2hdf5: Optional[str] = None,
        unit: Literal["kJ/mol", "eV/cell", "eV/atom"] = "kJ/mol",
    ):
        """Init method."""
        self._yaml = res_yaml
        self._unit = unit

        self._load_sscha_yaml()
        if fc2hdf5 is not None:
            self._fc2 = read_fc2_from_hdf5(fc2hdf5)

    def _load_sscha_yaml(self):
        """Load sscha_results.yaml file."""
        yaml_data = yaml.safe_load(open(self._yaml))

        self._pot = yaml_data["parameters"]["pot"]
        self._temp = yaml_data["parameters"]["temperature"]
        self._sscha_params = yaml_data["parameters"]

        properties = yaml_data["properties"]
        self._free_energy = properties["free_energy"]
        self._static_potential = properties["static_potential"]
        self._entropy = properties["entropy"]
        self._harmonic_heat_capacity = properties["harmonic_heat_capacity"]

        self._delta_fc = yaml_data["status"]["delta_fc"]
        self._converge = yaml_data["status"]["converge"]
        self._imaginary = yaml_data["status"]["imaginary"]
        self._logs = yaml_data["logs"]

        self._unitcell = parse_structure_from_yaml(yaml_data, tag="unitcell")
        self._supercell_matrix = np.array(yaml_data["supercell_matrix"])
        self._n_atom_unitcell = len(self._unitcell.elements)
        self._volume = np.linalg.det(self._unitcell.axis)

    @property
    def unit(self):
        """Return unit for free energy."""
        return self._unit

    @unit.setter
    def unit(self, unit_in: Literal["kJ/mol", "eV/cell", "eV/atom"]):
        """Set unit."""
        self._unit = unit_in

    @property
    def polymlp(self):
        """Return MLP file name."""
        return self._pot

    @property
    def temperature(self):
        """Return temperature."""
        return self._temp

    def _unit_conversion(self, val):
        """Convert unit for free energy and potentials."""
        if self._unit == "kJ/mol":
            return val
        elif self._unit == "eV/cell":
            return kjmol_to_ev(val)
        elif self._unit == "eV/atom":
            return kjmol_to_ev(val) / self._n_atom_unitcell
        raise RuntimeError("Unit must be kJ/mol, eV/cell, or eV/atom")

    def _unit_conversion_entropy(self, val):
        """Convert unit for entropy and heat capacity."""
        if self._unit == "kJ/mol":
            return val
        elif self._unit == "eV/cell":
            return kjmol_to_ev(val) / 1000
        elif self._unit == "eV/atom":
            return kjmol_to_ev(val) / 1000 / self._n_atom_unitcell
        raise RuntimeError("Unit must be kJ/mol, eV/cell, or eV/atom")

    @property
    def free_energy(self):
        """Return free energy."""
        return self._unit_conversion(self._free_energy)

    @property
    def static_potential(self):
        """Return static potential energy."""
        return self._unit_conversion(self._static_potential)

    @property
    def entropy(self):
        """Return entropy."""
        return self._unit_conversion_entropy(self._entropy)

    @property
    def harmonic_heat_capacity(self):
        """Return harmonic heat capacity."""
        return self._unit_conversion_entropy(self._harmonic_heat_capacity)

    @property
    def logs(self):
        """Return logs."""
        return self._logs

    @property
    def delta_fc(self):
        """Return FC difference between the last two iterations."""
        return self._delta_fc

    @property
    def converge(self):
        """Return convergence tag."""
        return self._converge

    @property
    def imaginary(self):
        """Return imaginary tag."""
        return self._imaginary

    @property
    def parameters(self):
        """Return simulation parameters."""
        return self._sscha_params

    @property
    def force_constants(self):
        """Return effective FC2."""
        return self._fc2

    @property
    def unitcell(self) -> PolymlpStructure:
        """Return unitcell."""
        return self._unitcell

    @property
    def unitcell_phonopy(self):
        """Return unitcell in PhonopyAtoms."""
        return structure_to_phonopy_cell(self._unitcell)

    @property
    def supercell_matrix(self) -> np.ndarray:
        """Return supercell matrix."""
        return self._supercell_matrix

    @property
    def n_unitcells(self) -> int:
        """Return number of unit cells."""
        return int(round(np.linalg.det(self._supercell_matrix)))

    @property
    def supercell(self) -> PolymlpStructure:
        """Return supercell."""
        cell = phonopy_supercell(
            self._unitcell,
            supercell_matrix=self._supercell_matrix,
            return_phonopy=False,
        )
        return cell

    @property
    def supercell_phonopy(self):
        """Return supercell in PhonopyAtoms."""
        cell = phonopy_supercell(
            self._unitcell,
            supercell_matrix=self._supercell_matrix,
            return_phonopy=True,
        )
        return cell

    @property
    def volume(self):
        return self._volume


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

    print(" # structure ", flush=True)
    print("  - elements:     ", cell.elements, flush=True)
    print("  - axis:         ", cell.axis.T[0], flush=True)
    print("                  ", cell.axis.T[1], flush=True)
    print("                  ", cell.axis.T[2], flush=True)
    print("  - positions:    ", cell.positions.T[0], flush=True)
    if cell.positions.shape[1] > 1:
        for pos in cell.positions.T[1:]:
            print("                  ", pos, flush=True)


def print_parameters(supercell_matrix: np.ndarray, args):
    """Print parameters in SSCHA."""

    print(" # parameters", flush=True)
    print("  - supercell:    ", supercell_matrix[0], flush=True)
    print("                  ", supercell_matrix[1], flush=True)
    print("                  ", supercell_matrix[2], flush=True)
    print("  - temperatures: ", args.temperatures[0], flush=True)
    if len(args.temperatures) > 1:
        for t in args.temperatures[1:]:
            print("                  ", t, flush=True)

    if isinstance(args.pot, list):
        for p in args.pot:
            print("  - Polynomial ML potential:  ", os.path.abspath(p), flush=True)
    else:
        print("  - Polynomial ML potential:  ", os.path.abspath(args.pot), flush=True)

    print("  - FC tolerance:             ", args.tol, flush=True)
    print("  - max iter:                 ", args.max_iter, flush=True)
    print("  - num samples:              ", args.n_samples_init, flush=True)
    print("  - num samples (last iter.): ", args.n_samples_final, flush=True)
    print("  - q-mesh:                   ", args.mesh, flush=True)


def print_array1d(array, tag, fstream, indent_l=0):
    prefix = "".join([" " for n in range(indent_l)])
    print(prefix + tag + ":", file=fstream, flush=True)
    for i, d in enumerate(array):
        print(prefix + " -", d, file=fstream, flush=True)


def print_array2d(array, tag, fstream, indent_l=0):
    prefix = "".join([" " for n in range(indent_l)])
    print(prefix + tag + ":", file=fstream, flush=True)
    for i, d in enumerate(array):
        print(prefix + " -", list(d), file=fstream, flush=True)


def save_cell(cell: PolymlpStructure, tag="unitcell", fstream=None, filename=None):
    """Write structure to a file."""

    np.set_printoptions(legacy="1.21")
    if fstream is None:
        fstream = open(filename, "w")

    print(tag + ":", file=fstream, flush=True)
    print_array2d(cell.axis.T, "axis", fstream, indent_l=2)
    print_array2d(cell.positions.T, "positions", fstream, indent_l=2)
    print("  n_atoms:  ", list(cell.n_atoms), file=fstream, flush=True)
    print("  types:    ", list(cell.types), file=fstream, flush=True)
    print("  elements: ", list(cell.elements), file=fstream, flush=True)

    if tag == "supercell":
        print("  n_unitcells: ", cell.n_unitcells, file=fstream, flush=True)
        print_array2d(
            cell.supercell_matrix,
            "supercell_matrix",
            fstream,
            indent_l=2,
        )

    print("", file=fstream, flush=True)


def save_sscha_yaml(
    unitcell: PolymlpStructure,
    supercell_matrix: np.ndarray,
    sscha_log: list[PolymlpDataSSCHA],
    args,
    filename="sscha_results.yaml",
):
    """Write SSCHA results to a file."""

    np.set_printoptions(legacy="1.21")
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

    print("units:", file=f)
    print("  free_energy:            kJ/mol", file=f)
    print("  static_potential:       kJ/mol", file=f)
    print("  entropy:                J/K/mol", file=f)
    print("  harmonic_heat_capacity: J/K/mol", file=f)
    print("", file=f)

    print("properties:", file=f)
    print("  free_energy:           ", properties.free_energy, file=f)
    print("  static_potential:      ", properties.static_potential, file=f)
    print("  entropy:               ", properties.entropy, file=f)
    print("  harmonic_heat_capacity:", properties.harmonic_heat_capacity, file=f)
    print("", file=f)

    print("status:", file=f)
    print("  delta_fc:  ", properties.delta, file=f)
    print("  converge:  ", properties.converge, file=f)
    print("  imaginary: ", properties.imaginary, file=f)
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


def is_imaginary(freq: np.ndarray, tol: float = -0.1):
    """Check branches with imaginary frequencies."""
    if np.min(freq) < tol:
        return True
    return False
