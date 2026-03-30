"""Utility functions for developing disordered model."""

import os
from typing import Optional

import numpy as np

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.api.pypolymlp_str import PypolymlpStructureGenerator
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.postproc.disorder_utils import (
    eval_substitutional_structures,
    set_full_occupancy,
)
from pypolymlp.utils.structure_utils import supercell_diagonal
from pypolymlp.utils.yaml_utils import save_data


class PolymlpDisorder:
    """Class for developing disordered polynomial MLP."""

    def __init__(
        self,
        occupancy: tuple,
        pot: str = "polymlp.yaml",
        lattice_poscar: str = "POSCAR",
        lattice_structure: Optional[PolymlpStructure] = None,
        supercell_size: Optional[tuple] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        occupancy: Element occupancy in list of list. The first index corresponds
                   to the sublattice index. For each sublattice, the occupancy is
                   given by a pair of element and probability. The sum of
                   probabilities must be equal to one for each sublattice.
        pot: Potential file name.
        lattice_poscar: POSCAR file name for lattice.
        lattice_structure: Structure for lattice.
        supercell_size: Diagonal elements of supercell matrix.

        Example
        -------
        occupancy = [[("Ag", 0.5), ("Au", 0.5)], [("Cu", 1.0)]]
        occupancy = [[(("Fe", 0), 0.5), (("Fe", 1), 0.5))]]
        """
        self._calc = PypolymlpCalc(pot=pot)
        self._params = self._calc.params
        self._occupancy = set_full_occupancy(self._params, occupancy)
        self._verbose = verbose

        self._lattice = None
        self._lattice_supercell = None

        self._displaced_lattices = None
        self._energies = None
        self._forces = None
        self._stresses = None

        self.load_lattice(
            filename=lattice_poscar,
            structure=lattice_structure,
            supercell_size=supercell_size,
        )

        if self._verbose:
            print("Generating MLP for disordered model.", flush=True)
            idx = 0
            elements = self._lattice.elements
            for i, (n, occ) in enumerate(zip(self._lattice.n_atoms, self._occupancy)):
                print("Sublattice", str(i + 1) + ":", flush=True)
                print("  Representation:", elements[idx], flush=True)
                print("  Occupancy:     ", occ, flush=True)
                idx += n

    def load_lattice(
        self,
        filename: str = "POSCAR",
        structure: Optional[PolymlpStructure] = None,
        supercell_size: Optional[tuple] = None,
    ):
        """Load lattice POSCAR file."""
        if structure is not None:
            self._lattice = structure
        else:
            self._lattice = Poscar(filename).structure

        if len(self._lattice.n_atoms) != len(self._occupancy):
            raise RuntimeError("Sizes of lattice and occupancy are inconsistent.")

        if supercell_size is not None:
            self._lattice_supercell = supercell_diagonal(
                self._lattice,
                size=supercell_size,
            )
        else:
            self._lattice_supercell = self._lattice
        return self

    def set_displaced_lattices(
        self,
        n_samples: int = 100,
        max_distance: float = 1.0,
        include_base_structure: bool = False,
    ):
        """Set structures with atomic displacements."""
        if self._lattice_supercell is None:
            raise RuntimeError("Supercell lattice not found.")

        stgen = PypolymlpStructureGenerator(base_structures=self._lattice_supercell)
        stgen.build_supercells_auto()
        stgen.run_standard_algorithm(n_samples=n_samples, max_distance=max_distance)

        if include_base_structure:
            self._displaced_lattices = [self._lattice_supercell]
            self._displaced_lattices.extend(stgen.sample_structures)
        else:
            self._displaced_lattices = stgen.sample_structures
        return self

    def eval_random_properties(
        self,
        n_samples: Optional[int] = 500,
        max_iter: int = 20,
        etol: float = 1e-4,
    ):
        """Evaluate average properties for substitutional structures."""
        if self._displaced_lattices is None:
            raise RuntimeError("Lattices with atomic displacements not found.")

        n_disps = len(self._displaced_lattices)
        self._energies, self._forces, self._stresses = [], [], []
        for i, lat in enumerate(self._displaced_lattices):
            if self._verbose:
                print("--- Displacement:", i + 1, "/", n_disps, "---", flush=True)

            e, f, s = self._eval_single(
                lat,
                n_samples=n_samples,
                max_iter=max_iter,
                etol=etol,
            )
            self._energies.append(e)
            self._forces.append(f)
            self._stresses.append(s)

        return self

    def _eval_single(
        self,
        lattice: PolymlpStructure,
        n_samples: Optional[int] = 500,
        max_iter: int = 20,
        etol: float = 1e-4,
    ):
        """Evaluate average properties over substitutional structures."""
        energies_all, forces_all, stresses_all = [], [], []
        diff = etol + 1.0
        ave_energy_prev = None
        iter1 = 1
        while diff > etol and iter1 <= max_iter:
            if self._verbose:
                print("Iteration", iter1, flush=True)

            energies, forces, stresses = eval_substitutional_structures(
                self._calc,
                lattice,
                self._occupancy,
                n_samples=n_samples,
            )
            energies_all.extend(energies)
            forces_all.extend(forces)
            stresses_all.extend(stresses)

            ave_energy = np.mean(energies_all)
            if self._verbose:
                print(" Number of samples:", len(energies_all), flush=True)
                print(" Average energy:   ", ave_energy, "[eV/cell]", flush=True)

            if iter1 > 1:
                diff = abs(ave_energy - ave_energy_prev) / sum(lattice.n_atoms)
                if self._verbose:
                    print(" Convergence score:", diff, "[eV/atom]", flush=True)

            ave_energy_prev = ave_energy
            iter1 += 1

        ave_forces = np.mean(forces_all, axis=0)
        ave_stress = np.mean(stresses_all, axis=0)
        return ave_energy, ave_forces, ave_stress

    def save_properties(self, path: str = "./polymlp_property"):
        """Save structure and properties."""
        zip_data = zip(
            self._displaced_lattices,
            self._energies,
            self._forces,
            self._stresses,
        )
        os.makedirs(path, exist_ok=True)
        for i, (st, e, f, s) in enumerate(zip_data):
            if len(self._energies) > 1:
                name = path + "/polymlp_disorder_" + str(i + 1).zfill(5) + ".yaml"
            else:
                name = path + "/polymlp_disorder.yaml"
            save_data(st, e, forces=f, stress=s, file=name)
        return self

    @property
    def structures(self):
        """Return structures for regression."""
        return self._displaced_lattices

    @property
    def properties(self):
        """Return properties for regression.

        Returns
        -------
        energy: Energy. shape=(n_str) in eV/supercell.
        forces: Forces. shape=(n_str, 3, n_atom) in eV/angstroms.
        stress: Stress tensor. shape=(n_str, 6), unit: eV/supercell
                in the order of xx, yy, zz, xy, yz, zx.
        """
        return (
            np.array(self._energies),
            np.array(self._forces),
            np.array(self._stresses),
        )
