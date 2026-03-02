"""Class for computing formation energies."""

from typing import Optional

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.utils.composition_utils import Composition
from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure


def compute_formation_energies(
    structures: list[PolymlpStructure],
    end_structures: Optional[list[PolymlpStructure]] = None,
    energies: Optional[np.ndarray] = None,
    end_energies: Optional[np.ndarray] = None,
    elements: Optional[list] = None,
    convex_hull: bool = True,
):
    """Compute formation energies and their convex hull."""
    _initialize_composition(end_structures, end_energies)

    # self._end_structures = structures
    # self._end_energies, _, _ = self._prop.eval_multiple(structures)

    # chemical_comps_end_members = [
    #     [np.count_nonzero(st.elements == ele) for ele in self._elements]
    #     for st in structures
    # ]
    # chemical_comps_end_members = np.array(chemical_comps_end_members)
    # self._comp = Composition(chemical_comps_end_members)
    # self._comp.energies_end_members = self._end_energies


def _initialize_composition(
    end_structures: Optional[list[PolymlpStructure]] = None,
    end_energies: Optional[np.ndarray] = None,
    elements: Optional[list] = None,
    properties: Optional[Properties] = None,
):
    """Initialize composition class."""
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

        chemical_comps_end_members = [
            [np.count_nonzero(np.array(st.elements) == ele) for ele in elements]
            for st in end_structures
        ]
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


class PolymlpFormationEnergies:
    """Class for computing formation energies."""

    def __init__(
        self,
        pot: Optional[str] = None,
        params: Optional[PolymlpParams] = None,
        coeffs: Optional[np.ndarray] = None,
        properties: Optional[Properties] = None,
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

        if properties is not None:
            self._prop = properties
        else:
            self._prop = Properties(pot=pot, params=params, coeffs=coeffs)

        self._elements = self._prop.params.elements
        self._n_elements = len(self._elements)

        self._comp = None
        self._end_structures = None
        self._end_energies = None

    def _get_n_atoms(self, structures: list[PolymlpStructure]):
        """Get number of atoms with reordering elements."""
        n_atoms_array = []
        for st in structures:
            elems = np.array(st.elements)
            uniq, counts = np.unique(elems, return_counts=True)
            count_dict = dict(zip(uniq, counts))
            n_atoms = []
            for ele in self._elements:
                try:
                    n = count_dict[ele]
                except:
                    n = 0
                n_atoms.append(n)
            n_atoms_array.append(n_atoms)
        return np.array(n_atoms_array)

    def compute(
        self,
        structures: list[PolymlpStructure],
        energies: Optional[np.ndarray] = None,
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
        if self._end_energies is None:
            raise RuntimeError("Energies for end members not found.")

        if energies is not None:
            if len(energies) != len(structures):
                raise RuntimeError("Sizes of structures and energies inconsistent.")
        else:
            energies, _, _ = self._prop.eval_multiple(structures)
        n_atoms_array = self._get_n_atoms(structures)
        form = self._comp.compute_formation_energies(energies, n_atoms_array)
        return form

    @property
    def end_structures(self):
        """Return structures of end members."""
        return self._end_structures

    @end_structures.setter
    def end_structures(self, structures: list[PolymlpStructure]):
        """Set structures of end members.

        For example, if Ag4 and Au8 are endmembers that are used
        to define the composition, the numbers of atoms for end member structures
        must be given as [[4, 0], [0, 8]].
        """
        if len(structures) != self._n_elements:
            # TODO: Not available for Pseudo-k-ary systems,
            #       but available for these systems by using Composition class
            raise RuntimeError("Length of end member structures must be n_elements.")

        self._end_structures = structures
        self._end_energies, _, _ = self._prop.eval_multiple(structures)

        chemical_comps_end_members = [
            [np.count_nonzero(st.elements == ele) for ele in self._elements]
            for st in structures
        ]
        chemical_comps_end_members = np.array(chemical_comps_end_members)
        self._comp = Composition(chemical_comps_end_members)
        self._comp.energies_end_members = self._end_energies
