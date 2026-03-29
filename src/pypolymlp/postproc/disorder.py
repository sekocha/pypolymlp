"""Utility functions for developing disordered model."""

import copy
from typing import Optional

import numpy as np

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.api.pypolymlp_str import PypolymlpStructureGenerator
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.core.params import PolymlpParams
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp
from pypolymlp.postproc.disorder_utils import (
    eval_substitutional_structures,
    set_element_map,
)
from pypolymlp.utils.structure_utils import supercell_diagonal
from pypolymlp.utils.yaml_utils import save_data


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

        self._map_element_to_type = set_element_map(self._params, self._occupancy)
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
                self._map_element_to_type,
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

    def save_properties(self, filename: str = "polymlp_prediction.yaml"):
        """Save structure and properties."""
        zip_data = zip(
            self._displaced_lattices,
            self._energies,
            self._forces,
            self._stresses,
        )
        for i, (st, e, f, s) in enumerate(zip_data):
            if len(self._energies) > 1:
                name = filename + "." + str(i + 1).zfill(5)
            else:
                name = filename
            save_data(st, e, forces=f, stress=s, file=name)
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
        """Return properties for regression.

        Returns
        -------
        energy: Energy. shape=(n_str) in eV/supercell.
        forces: Forces. shape=(n_str, 3, n_atom) in eV/angstroms.
        stress: Stress tensor. shape=(n_str, 6), unit: eV/supercell
                in the order of xx, yy, zz, xy, yz, zx.
        """
        return self._energies, self._forces, self._stresses
