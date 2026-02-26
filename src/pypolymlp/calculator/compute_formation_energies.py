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

    def compute(self, structures: list[PolymlpStructure]):
        """Compute formation energies."""
        energies, _, _ = self._prop.eval_multiple(structures)
        n_atoms_array = self._get_n_atoms(structures)

        # TODO: Set end members
        chemical_comps_end_members = np.eye(len(self._elements))
        comp = Composition(chemical_comps_end_members)
        form = comp.compute_formation_energies(energies, n_atoms_array)
        print(form)
