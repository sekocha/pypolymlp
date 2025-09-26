"""Utility functions for developing disordered model."""

import copy
import random
from collections import defaultdict
from typing import Optional

import numpy as np

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.api.pypolymlp_str import PypolymlpStructureGenerator
from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp
from pypolymlp.utils.structure_utils import sort_wrt_types, supercell_diagonal


def _check_params(params: PolymlpParams):
    """Check constraints for parameters."""
    if not params.type_full:
        raise RuntimeError("type_full must be True")

    uniq_ids = set()
    for ids in params.model.pair_params_conditional.values():
        uniq_ids.add(tuple(ids))

    if len(uniq_ids) != 1:
        raise RuntimeError("All pair_params_conditional must be the same.")

    return True


def _generate_disorder_params(params: PolymlpParams, occupancy: tuple):
    """Check occupancy format."""
    _check_params(params)

    params_rand = copy.deepcopy(params)
    map_mass = dict(zip(params.elements, params.mass))

    elements_rand, mass_rand = [], []
    type_group = []
    for occ in occupancy:
        if not np.isclose(sum([v for _, v in occ]), 1.0):
            raise RuntimeError("Sum of occupancy != 1.0")

        mass, type_tmp = 0.0, []
        for ele, comp in occ:
            if ele not in params.elements:
                raise RuntimeError("Element", ele, "not found in polymlp.")

            mass += map_mass[ele] * comp
            type_tmp.append(params.elements.index(ele))

        elements_rand.append(occ[0][0])
        mass_rand.append(mass)
        type_group.append(type_tmp)

    params_rand.n_type = len(occupancy)
    params_rand.elements = elements_rand
    params_rand.element_order = elements_rand
    params_rand.mass = mass_rand

    # TODO: Modify type_pairs and type_indices
    #       Use type_group
    params_rand.type_full = True
    params_rand.type_indices = list(range(params_rand.n_type))

    occupancy_type = [
        [(params.elements.index(ele), comp) for ele, comp in occ] for occ in occupancy
    ]
    return params_rand, occupancy_type


def check_occupancy(params: PolymlpParams, occupancy: tuple):
    """Check occupancy format."""
    map_element_to_type = dict()
    itype = 0
    for occ in occupancy:
        if not np.isclose(sum([v for _, v in occ]), 1.0):
            raise RuntimeError("Sum of occupancy != 1.0")

        for ele, comp in occ:
            if ele not in params.elements:
                raise RuntimeError("Element", ele, "not found in polymlp.")
            if ele not in map_element_to_type:
                map_element_to_type[ele] = itype
                itype += 1
    return map_element_to_type


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
        """Init method.

        Parameters
        ----------
        occupancy: Element occupancy in list of list. The first index corresponds
                   to the sublattice index. For each sublattice, the occupancy is
                   given by a pair of element and probability. The sum of
                   probabilities must be equal to one for each sublattice.
        pot: Potential file name.
        lattice: POSCAR file name for lattice.
        supercell_size: Diagonal elements of supercell matrix.

        Example
        -------
        occupancy = [[("Ag", 0.5), ("Au", 0.5)], [("Cu", 1.0)]]
        """
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

        self._map_element_to_type = check_occupancy(self._params, self._occupancy)
        self.load_lattice(filename=lattice, supercell_size=supercell_size)

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
        supercell_size: Optional[tuple] = None,
    ):
        """Load lattice POSCAR file."""
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
    ):
        """Set structures with atomic displacements."""
        if self._lattice_supercell is None:
            raise RuntimeError("Supercell lattice not found.")

        stgen = PypolymlpStructureGenerator(base_structures=self._lattice_supercell)
        stgen.build_supercells_auto()
        stgen.run_standard_algorithm(n_samples=n_samples, max_distance=max_distance)
        self._displaced_lattices = [self._lattice_supercell] + stgen.sample_structures
        return self

    def _set_replacements(self):
        """Set atom indices to replace elements randomly."""
        if self._lattice_supercell is None:
            raise RuntimeError("Supercell lattice not found.")

        replace_ids = defaultdict(list)
        atom_begin = 0
        for occ, n in zip(self._occupancy, self._lattice_supercell.n_atoms):
            atom_end = atom_begin + n
            cand = range(atom_begin, atom_end)
            for ele, prob in occ:
                n_replace = int(round(n * prob))
                itype = self._map_element_to_type[ele]
                if len(cand) == n_replace:
                    replace_ids[(ele, itype)].extend(cand)
                else:
                    replace_ids[(ele, itype)].extend(random.sample(cand, n_replace))
                    cand = list(set(cand) - set(replace_ids[(ele, itype)]))
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
                print("Displacement:", i, "/", n_disps, flush=True)

            subs, atom_orders = self._generate_substitutional_structures(lat, replaces)
            energies, forces_sorted_order, stresses = self._calc.eval(subs)

            forces = []
            for f, ids in zip(forces_sorted_order, atom_orders):
                f_reordered = np.zeros(f.shape)
                f_reordered[:, ids] = f
                forces.append(f_reordered)
            np.set_printoptions(suppress=True)

            self._energies.append(np.mean(energies))
            self._forces.append(np.mean(forces, axis=0))
            self._stresses.append(np.mean(stresses, axis=0))
        return self

    def develop_mlp(
        self,
        reg_alpha_params: tuple = (-3, 5, 30),
        filename: str = "polymlp.yaml.disorder",
    ):
        """Develop polymlp."""
        params_disorder, _ = _generate_disorder_params(self._params, self._occupancy)
        params_disorder.set_alphas(reg_alpha_params)

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

    @property
    def polymlp(self):
        """Return Pypolymlp instance."""
        return self._polymlp

    @property
    def structures(self):
        """Return structures for regression."""
        return self._displaced_lattices

    @property
    def properties(self):
        """Return properties for regression."""
        return self._energies, self._forces, self._stresses
