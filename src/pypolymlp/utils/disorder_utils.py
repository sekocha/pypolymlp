"""Utility functions for developing disordered model."""

import copy
import random
from typing import Optional

import numpy as np

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.api.pypolymlp_str import PypolymlpStructureGenerator
from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.mlp_dev.disorder.disorder import _generate_disorder_params
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp
from pypolymlp.utils.structure_utils import sort_wrt_types, supercell_diagonal


def check_occupancy(params: PolymlpParams, occupancy: tuple):
    """Check occupancy format."""
    for occ in occupancy:
        if not np.isclose(sum([v for _, v in occ]), 1.0):
            raise RuntimeError("Sum of occupancy != 1.0")

        for ele, comp in occ:
            if ele not in params.elements:
                raise RuntimeError("Element", ele, "not found in polymlp.")


def initialize_lattice(lattice: PolymlpStructure, occupancy: tuple):
    """Initialize lattice using occupancy."""
    if len(lattice.n_atoms) != len(occupancy):
        raise RuntimeError("Sizes of lattice and occupancy are inconsistent.")

    lattice_types, lattice_elements = [], []
    i = 0
    for occ in occupancy:
        lattice_types.append(i)
        lattice_elements.append(occ[0][0])
        i += len(occ)

    n_atoms = lattice.n_atoms
    lattice.types = [t for n, t in zip(n_atoms, lattice_types) for _ in range(n)]
    lattice.elements = [e for n, e in zip(n_atoms, lattice_elements) for _ in range(n)]
    return lattice


class PolymlpDisorder:
    """Class for developing disordered polynomial MLP."""

    def __init__(
        self,
        occupancy: tuple,
        pot: str = "polymlp.yaml",
        lattice: str = "POSCAR",
        supercell_size: Optional[tuple] = None,
        verbose: bool = False,
    ):
        """Init method."""
        self._polymlp = Pypolymlp()
        self._calc = PypolymlpCalc(pot=pot)
        self._params = self._calc.params
        self._occupancy = occupancy
        self._verbose = verbose

        self._lattice = None
        self._lattice_supercell = None

        self._displaced_lattices = None
        self._energies = None
        self._forces = None
        self._stresses = None

        check_occupancy(self._params, self._occupancy)
        self.load_lattice(filename=lattice, supercell_size=supercell_size)

    def load_lattice(
        self,
        filename: str = "POSCAR",
        supercell_size: Optional[tuple] = None,
    ):
        """Load lattice POSCAR file."""
        self._lattice = Poscar(filename).structure
        self._lattice = initialize_lattice(self._lattice, self._occupancy)
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
    ):
        """Set structures with atomic displacements."""
        if self._lattice_supercell is None:
            raise RuntimeError("Supercell lattice not found.")

        stgen = PypolymlpStructureGenerator(base_structures=self._lattice_supercell)
        stgen.build_supercells_auto()
        stgen.run_standard_algorithm(n_samples=n_samples, max_distance=max_distance)
        self._displaced_lattices = stgen.sample_structures
        return self

    def _set_replacements(self):
        """Set atom indices to replace elements randomly."""
        if self._lattice_supercell is None:
            raise RuntimeError("Supercell lattice not found.")

        replace_ids = dict()
        atom_begin, itype = 0, 0
        for occ, n in zip(self._occupancy, self._lattice_supercell.n_atoms):
            atom_end = atom_begin + n
            if len(occ) > 1:
                itype += 1
                cand = range(atom_begin, atom_end)
                for ele, prob in occ[1:]:
                    n_replace = int(round(n * prob))
                    replace_ids[(ele, itype)] = random.sample(cand, n_replace)
                    cand = list(set(cand) - set(replace_ids[(ele, itype)]))
                    itype += 1
            else:
                itype += 1
            atom_begin = atom_end
        return replace_ids

    def _generate_substitutional_structures(
        self,
        lattice: PolymlpStructure,
        replaces: list[dict],
    ):
        """Generate substitutional structures using replacement indices."""
        structures, atom_orders = [], []
        for replace_dict in replaces:
            st = copy.deepcopy(lattice)
            for (ele, itype), replace_ids in replace_dict.items():
                st.elements[replace_ids] = ele
                st.types[replace_ids] = itype

            # TODO: sorted positions must be resorted to original order.
            st, ids = sort_wrt_types(st, return_ids=True)
            structures.append(st)
            atom_orders.append(ids)
        return structures, atom_orders

    def eval_random_properties(self, n_samples: int = 100):
        """Evaluate average properties for substitutional structures."""
        if self._displaced_lattices is None:
            raise RuntimeError("Lattices with atomic displacements not found.")

        replaces = [self._set_replacements() for i in range(n_samples)]

        self._energies = []
        self._forces = []
        self._stresses = []
        for i, lat in enumerate(self._displaced_lattices):
            if self._verbose:
                n_disps = len(self._displaced_lattices)
                print("Displacement:", i + 1, "/", n_disps, flush=True)

            subs, atom_orders = self._generate_substitutional_structures(lat, replaces)
            energies, forces_sorted_order, stresses = self._calc.eval(subs)

            forces = []
            for f, ids in zip(forces_sorted_order, atom_orders):
                f_reordered = np.zeros(f.shape)
                f_reordered[:, ids] = f
                forces.append(f_reordered)

            self._energies.append(np.mean(energies))
            self._forces.append(np.mean(forces, axis=0))
            self._stresses.append(np.mean(stresses, axis=0))
        return self

    def develop_mlp(self, filename: str = "polymlp.yaml.disorder"):
        """Develop polymlp."""
        params_disorder, _ = _generate_disorder_params(self._params, self._occupancy)
        params_disorder.set_alphas((-3, 5, 30))

        self._polymlp.set_params(params=params_disorder)
        self._polymlp.set_datasets_structures_autodiv(
            structures=self._displaced_lattices,
            energies=self._energies,
            forces=self._forces,
            stresses=None,
        )
        self._polymlp.fit(verbose=self._verbose)
        self._polymlp.save_mlp(filename=filename)
        self._polymlp.estimate_error(log_energy=True, verbose=self._verbose)
        return self
