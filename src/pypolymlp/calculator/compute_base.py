"""Base class for computing properties."""

from typing import Optional, Union

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpParams


class PolymlpComputeBase:
    """Base class for computing properties."""

    def __init__(
        self,
        pot: Optional[str, list[str]] = None,
        params: Optional[PolymlpParams] = None,
        coeffs: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
        properties: Optional[Properties] = None,
        return_none: bool = False,
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
            if return_none:
                self._prop = None
            else:
                raise RuntimeError("Polymlp not provided.")
        elif properties is not None:
            self._prop = properties
        else:
            if params is not None:
                if coeffs is None:
                    raise RuntimeError("Coefficients not provided.")
                if len(params) != len(coeffs):
                    raise RuntimeError("Length of params and coeffs not consistent.")
            self._prop = Properties(pot=pot, params=params, coeffs=coeffs)

        self._verbose = verbose

    @property
    def prop(self):
        """Return instance of Properties."""
        return self._prop
