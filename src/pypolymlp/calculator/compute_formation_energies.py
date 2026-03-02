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
        """Set structures of end members."""
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
