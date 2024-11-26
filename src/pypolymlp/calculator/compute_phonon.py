"""Class for computing phonon."""

import os
from typing import Optional

import numpy as np
from phonopy import Phonopy, PhonopyQHA

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure
from pypolymlp.utils.phonopy_utils import (
    phonopy_cell_to_structure,
    structure_to_phonopy_cell,
)
from pypolymlp.utils.structure_utils import isotropic_volume_change


class PolymlpPhonon:
    """Class for computing phonon."""

    def __init__(
        self,
        unitcell: PolymlpStructure,
        supercell_matrix: np.ndarray,
        pot: Optional[str] = None,
        params: Optional[PolymlpParams] = None,
        coeffs: Optional[np.ndarray] = None,
        properties: Optional[Properties] = None,
    ):
        """Init method.

        Parameters
        ----------
        unitcell: Unitcell in PolymlpStructure format
        supercell_matrix: Supercell matrix.
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties object.

        Any one of pot, (params, coeffs), and properties is needed.
        """

        if properties is not None:
            self.prop = properties
        else:
            self.prop = Properties(pot=pot, params=params, coeffs=coeffs)

        unitcell = structure_to_phonopy_cell(unitcell)
        self.ph = Phonopy(unitcell, supercell_matrix)
        self._with_pdos = False

    def produce_force_constants(self, distance: float = 0.001):
        """Produce force constants by evaluating forces for random structures."""
        self.ph.generate_displacements(distance=distance)
        supercells = self.ph.supercells_with_displacements
        structures = [phonopy_cell_to_structure(cell) for cell in supercells]

        # forces: (n_str, 3, n_atom) --> (n_str, n_atom, 3)
        _, forces, _ = self.prop.eval_multiple(structures)
        forces = np.array(forces).transpose((0, 2, 1))
        self.ph.forces = forces
        self.ph.produce_force_constants()
        return self

    def compute_properties(
        self,
        mesh: np.ndarray = (10, 10, 10),
        t_min: float = 0,
        t_max: float = 1000,
        t_step: float = 10,
        with_eigenvectors: bool = False,
        is_mesh_symmetry: bool = True,
        with_pdos: bool = False,
    ):
        """Compute phonon properties."""
        self.ph.run_mesh(
            mesh,
            with_eigenvectors=with_eigenvectors,
            is_mesh_symmetry=is_mesh_symmetry,
        )
        self.ph.run_total_dos()
        self.ph.run_thermal_properties(t_step=t_step, t_max=t_max, t_min=t_min)
        self.mesh_dict = self.ph.get_mesh_dict()
        if with_pdos:
            self._with_pdos = True
            self.ph.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)
            self.ph.run_projected_dos()
        return self

    def write_properties(self, path_output: str = "./"):
        """Save properties."""
        os.makedirs(path_output + "/polymlp_phonon", exist_ok=True)
        np.savetxt(
            path_output + "/polymlp_phonon/mesh-qpoints.txt",
            self.mesh_dict["qpoints"],
            fmt="%f",
        )
        self.ph.write_total_dos(filename=path_output + "/polymlp_phonon/total_dos.dat")
        self.ph.write_yaml_thermal_properties(
            filename=path_output + "/polymlp_phonon/thermal_properties.yaml"
        )

        if self._with_pdos:
            self.ph.write_projected_dos(
                filename=path_output + "/polymlp_phonon/proj_dos.dat"
            )

    def is_imaginary(self, threshold: float = -0.01) -> bool:
        """Check if imaginary phonon frequencies exist."""
        return np.min(self.mesh_dict["frequencies"]) < threshold

    @property
    def phonopy(self) -> Phonopy:
        """Return phonopy instance."""
        return self.ph


class PolymlpPhononQHA:
    """Class for computing QHA."""

    def __init__(
        self,
        unitcell: PolymlpStructure,
        supercell_matrix: np.ndarray,
        pot: Optional[str] = None,
        params: Optional[PolymlpParams] = None,
        coeffs: Optional[np.ndarray] = None,
        properties: Optional[Properties] = None,
    ):
        """Init method.

        Parameters
        ----------
        unitcell: unitcell in PolymlpStructure format
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties object.

        Any one of pot, (params, coeffs), and properties is needed.
        """

        if properties is not None:
            self.prop = properties
        else:
            self.prop = Properties(pot=pot, params=params, coeffs=coeffs)

        self._unitcell = unitcell
        self._supercell_matrix = supercell_matrix

    def run(
        self,
        distance: float = 0.001,
        mesh: np.ndarray = (10, 10, 10),
        t_min: float = 0,
        t_max: float = 1000,
        t_step: float = 10,
        eps_min: float = 0.8,
        eps_max: float = 1.2,
        eps_step: float = 0.02,
    ):
        """Run QHA."""
        eps_all = np.arange(eps_min, eps_max + 0.001, eps_step)
        unitcells = [
            isotropic_volume_change(self._unitcell, eps=eps) for eps in eps_all
        ]
        energies, _, _ = self.prop.eval_multiple(unitcells)
        volumes = np.array([st.volume for st in unitcells])

        free_energies, entropies, heat_capacities = [], [], []
        for unitcell in unitcells:
            ph = PolymlpPhonon(unitcell, self._supercell_matrix, properties=self.prop)
            ph.produce_force_constants(distance=distance)

            phonopy = ph.phonopy
            phonopy.run_mesh(mesh)
            phonopy.run_thermal_properties(t_step=t_step, t_max=t_max, t_min=t_min)

            tp_dict = phonopy.get_thermal_properties_dict()
            temperatures = tp_dict["temperatures"]
            free_energies.append(tp_dict["free_energy"])
            entropies.append(tp_dict["entropy"])
            heat_capacities.append(tp_dict["heat_capacity"])

        free_energies = np.array(free_energies).T
        entropies = np.array(entropies).T
        heat_capacities = np.array(heat_capacities).T
        self._qha = PhonopyQHA(
            volumes=volumes,
            electronic_energies=energies,
            temperatures=temperatures,
            free_energy=free_energies,
            entropy=entropies,
            cv=heat_capacities,
        )

    def write_qha(self, path_output: str = "./"):
        """Save results."""
        os.makedirs(path_output + "/polymlp_phonon_qha/", exist_ok=True)
        filename = path_output + "/polymlp_phonon_qha/helmholtz-volume.dat"
        self._qha.write_helmholtz_volume(filename=filename)
        filename = path_output + "/polymlp_phonon_qha/volume-temperature.dat"
        self._qha.write_volume_temperature(filename=filename)
        filename = path_output + "/polymlp_phonon_qha/thermal_expansion.dat"
        self._qha.write_thermal_expansion(filename=filename)
        filename = path_output + "/polymlp_phonon_qha/gibbs-temperature.dat"
        self._qha.write_gibbs_temperature(filename=filename)
        filename = path_output + "/polymlp_phonon_qha/bulk_modulus-temperature.dat"
        self._qha.write_bulk_modulus_temperature(filename=filename)
        filename = path_output + "/polymlp_phonon_qha/gruneisen-temperature.dat"
        self._qha.write_gruneisen_temperature(filename=filename)


if __name__ == "__main__":

    import argparse
    import signal

    from pypolymlp.core.interface_vasp import Poscar
    from pypolymlp.utils.yaml_utils import load_cells

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pot",
        nargs="*",
        type=str,
        default="polymlp.lammps",
        help="polymlp file",
    )
    parser.add_argument("--yaml", type=str, default=None, help="polymlp_str.yaml file")
    parser.add_argument("--poscar", type=str, default=None, help="poscar")
    parser.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        default=None,
        help="Supercell size (diagonal components)",
    )
    parser.add_argument(
        "--disp",
        type=float,
        default=0.01,
        help="random displacement (in Angstrom)",
    )

    parser.add_argument(
        "--ph_mesh",
        type=int,
        nargs=3,
        default=[10, 10, 10],
        help="k-mesh used for phonon calculation",
    )
    parser.add_argument("--ph_tmin", type=float, default=100, help="Temperature (min)")
    parser.add_argument("--ph_tmax", type=float, default=1000, help="Temperature (max)")
    parser.add_argument(
        "--ph_tstep", type=float, default=100, help="Temperature (step)"
    )
    parser.add_argument("--ph_pdos", action="store_true", help="Compute phonon PDOS")
    args = parser.parse_args()

    if args.yaml is not None:
        unitcell, supercell = load_cells(filename=args.yaml)
        supercell_matrix = supercell.supercell_matrix
    elif args.poscar is not None:
        unitcell = Poscar(args.poscar).structure
        supercell_matrix = np.diag(args.supercell)

    ph = PolymlpPhonon(unitcell, supercell_matrix, pot=args.pot)
    ph.produce_force_constants(displacements=args.disp)
    ph.compute_properties(
        mesh=args.ph_mesh,
        t_min=args.ph_tmin,
        t_max=args.ph_tmax,
        t_step=args.ph_tstep,
        pdos=args.ph_pdos,
    )

    qha = PolymlpPhononQHA(unitcell, supercell_matrix, pot=args.pot)
    qha.run()
    qha.write_qha()
