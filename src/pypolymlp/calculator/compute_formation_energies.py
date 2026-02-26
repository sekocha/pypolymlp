"""Class for computing formation energies."""

from typing import Optional

import numpy as np

from pypolymlp.calculator.properties import Properties

# from pypolymlp.calculator.utils.composition_utils import Composition
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

    def compute(self, structures: list[PolymlpStructure]):
        """Compute formation energies."""
