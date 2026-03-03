"""Class for computing formation energies."""

from typing import Optional

import numpy as np

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
        data = np.hstack([compositions, form.reshape((-1, 1))])
        return data


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
        print(chemical_comps_end_members)
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
