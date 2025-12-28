"""API Class for systematically calculating properties."""

from typing import Optional, Union

import numpy as np

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.calculator.auto.structures_element import get_structure_list_element
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpParams


class PypolymlpAutoCalc:
    """API Class for systematically calculating properties."""

    def __init__(
        self,
        pot: Union[str, list[str]] = None,
        params: Union[PolymlpParams, list[PolymlpParams]] = None,
        coeffs: Union[np.ndarray, list[np.ndarray]] = None,
        properties: Optional[Properties] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties instance.

        Any one of pot, (params, coeffs), and properties is needed.
        """
        self._calc = PypolymlpCalc(
            pot=pot,
            params=params,
            coeffs=coeffs,
            properties=properties,
            verbose=verbose,
        )
        self._prop = self._calc._prop
        self._verbose = verbose

        self._structures = None

        np.set_printoptions(legacy="1.21")

    def load_structures(self, n_types: int = 1):
        """Load a list of initial structures.

        Parameters
        ----------
        n_types: Number of atom species.
        """
        if n_types not in {1, 2}:
            raise RuntimeError("Structure list not found for systems beyond ternary.")

        element_strings = ["Ti"]
        if len(element_strings) == 1:
            self._prototypes = get_structure_list_element(element_strings)
        elif len(element_strings) == 2:
            pass

        return self
