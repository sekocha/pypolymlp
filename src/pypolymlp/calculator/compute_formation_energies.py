"""Class for computing formation energies."""

from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.utils.composition_utils import Composition
from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure


class PolymlpFormationEnergies:
    """Class for computing formation energies."""

    def __init__(
        self,
        pot: Optional[str] = None,
        params: Optional[PolymlpParams] = None,
        coeffs: Optional[np.ndarray] = None,
        properties: Optional[Properties] = None,
        elements: Optional[tuple] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties object.

        Any one of pot, (params, coeffs), and properties is needed.
        """
        if all(x is None for x in (properties, pot, params, coeffs)):
            self._prop = None
        elif properties is not None:
            self._prop = properties
        else:
            self._prop = Properties(pot=pot, params=params, coeffs=coeffs)

        if self._prop is not None:
            self._elements = self._prop.params.elements
        else:
            if elements is None:
                raise RuntimeError("Elements required if polymlp not provided.")
            self._elements = elements

        self._verbose = verbose
        self._n_elements = len(self._elements)
        self._comp = None

        self._data = None
        self._data_convex = None
        self._structure_names = None
        self._structure_names_convex = None

    def define_end_members(
        self,
        structures: Optional[list[PolymlpStructure]] = None,
        energies: Optional[np.ndarray] = None,
    ):
        """Define end members.

        For example, if Ag4 and Au8 are endmembers that are used to set end members
        and define the composition, the numbers of atoms in end member structures
        must be given as [[4, 0], [0, 8]].

        If structures are not given, energies are regarded as those per atom.
        """
        # TODO: Implement initialization using input of number of atoms.
        self._comp = _initialize_composition(
            end_structures=structures,
            end_energies=energies,
            elements=self._elements,
            properties=self._prop,
        )
        return self

    def compute(
        self,
        structures: list[PolymlpStructure],
        energies: Optional[np.ndarray] = None,
        structure_names: Optional[np.ndarray] = None,
    ):
        """Compute formation energies.

        Return
        ------
        form: Formation energies.
              delta E = E - (x1 * E_end1 + x2 * E_end2 + ...).
              For example, when Ag4 and Au8 are defined as end members,
              the formation energies per [Ag4](1-x)[Au8]x are calculated as
              delta E = E - [(1-x) * E(Ag4) + x * E(Au8)].
        """
        if self._comp is None:
            raise RuntimeError("End members not defined.")

        if energies is not None:
            if len(energies) != len(structures):
                raise RuntimeError("Sizes of structures and energies inconsistent.")
        else:
            energies, _, _ = self._prop.eval_multiple(structures)

        n_atoms_array = _get_n_atoms(structures, elements=self._elements)
        form = self._comp.compute_formation_energies(energies, n_atoms_array)
        compositions = self._comp.compositions

        if structure_names is None:
            self._structure_names = list(range(form.shape[0]))
        # self._structure_names = structure_names
        self._data = np.hstack([compositions, form.reshape((-1, 1))])
        return self._data

    def _sort_compositions(self, data: np.ndarray):
        """Sort data with respect to composition."""
        order = [-1] + list(range(0, self._n_elements))
        keys = [data[:, i] for i in order]
        return np.lexsort(keys)

    def _append_end_members(self):
        """Append end members to calculate convex hull if needed."""
        data = list(self._data)
        for i in range(self._n_elements):
            target = np.zeros(self._n_elements + 1)
            target[i] = 1.0
            mask = np.all(np.isclose(self._data, target), axis=1)
            if np.count_nonzero(mask) == 0:
                data.append(target)
                if self._structure_names is not None:
                    self._structure_names.append("End-" + str(i + 1))
        self._data = np.array(data)
        return self._data

    def _slice(self, data: np.ndarray, structure_names: list, indices: np.ndarray):
        """Slice data and structure names."""
        data_sliced = data[indices]
        if structure_names is not None:
            structure_names_sliced = [structure_names[i] for i in indices]
        else:
            structure_names_sliced = None
        return data_sliced, structure_names_sliced

    def convex_hull(self, tol: float = 1e-8):
        """Calculate convex hull."""
        self._append_end_members()
        negative_indices = np.where(self._data[:, -1] < tol)[0]
        data, names = self._slice(self._data, self._structure_names, negative_indices)

        if data.shape[0] > self._n_elements:
            ch = ConvexHull(data[:, 1:])
            v_convex = np.unique(ch.simplices)
            data, names = self._slice(data, names, v_convex)

        sort_keys = self._sort_compositions(data)
        res = self._slice(data, names, sort_keys)
        self._data_convex, self._structure_names_convex = res
        return self._data_convex

    @property
    def has_end_members(self):
        """Return whether end members are already defined."""
        return self._comp is not None

    @property
    def structure_names_convex(self):
        """Return names of structures on convex hull."""
        return self._structure_names_convex


def _get_n_atoms(structures: list[PolymlpStructure], elements: tuple):
    """Get number of atoms with reordering elements."""
    n_atoms_array = []
    for st in structures:
        elems = np.array(st.elements)
        uniq, counts = np.unique(elems, return_counts=True)
        count_dict = dict(zip(uniq, counts))
        n_atoms = []
        for ele in elements:
            try:
                n = count_dict[ele]
            except:
                n = 0
            n_atoms.append(n)
        n_atoms_array.append(n_atoms)
    return np.array(n_atoms_array)


def _initialize_composition(
    end_structures: Optional[list[PolymlpStructure]] = None,
    end_energies: Optional[np.ndarray] = None,
    elements: Optional[list] = None,
    properties: Optional[Properties] = None,
):
    """Initialize composition class."""
    # TODO: Not available for Pseudo-k-ary systems,
    #       but available for these systems by using Composition class
    if end_structures is None and end_energies is None:
        raise RuntimeError("Definition of end members not provided.")

    if end_structures is None:
        n_type = len(end_energies)
        chemical_comps_end_members = np.eye(n_type)
        composition = Composition(chemical_comps_end_members)
        composition.energies_end_members = end_energies
    else:
        if elements is None:
            raise RuntimeError("Elements required when using end_structures.")

        n_type = len(elements)
        if len(end_structures) != n_type:
            raise RuntimeError("Incosistent size of structures with n_type.")

        chemical_comps_end_members = _get_n_atoms(end_structures, elements=elements)
        composition = Composition(chemical_comps_end_members)

        if end_energies is None:
            if properties is None:
                raise RuntimeError("Properties class not found.")
            energies, _, _ = properties.eval_multiple(end_structures)
            composition.energies_end_members = energies
        else:
            if len(end_energies) != n_type:
                raise RuntimeError("Incosistent size of energies with n_type.")
            composition.energies_end_members = end_energies

    return composition


def find_endmembers(structures: list, energies: np.array, element_strings: tuple):
    """Find end members with lowest energies.

    Return
    ------
    endmembers: End members. Their energies are represented by unit of per atom.
    """
    endmembers = []
    for ele in element_strings:
        min_e, min_st = 1e10, None
        for st, ene in zip(structures, energies):
            if len(st.n_atoms) > 1 or st.elements[0] != ele:
                continue
            ene_per_atom = ene / len(st.elements)
            if ene_per_atom < min_e:
                min_e = ene_per_atom
                min_st = st
        if min_st is None:
            raise RuntimeError("End members not found.")
        endmembers.append((min_st, min_e))
    return endmembers
