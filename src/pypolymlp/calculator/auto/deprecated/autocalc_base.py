"""Base Class for systematic calculations."""

import os
from typing import Optional, Union

import numpy as np

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpParams


class AutoCalcBase:
    """Base Class for systematic calculations."""

    def __init__(
        self,
        pot: Union[str, list[str]] = None,
        params: Union[PolymlpParams, list[PolymlpParams]] = None,
        coeffs: Union[np.ndarray, list[np.ndarray]] = None,
        properties: Optional[Properties] = None,
        path_output: str = ".",
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
        self._pot = pot
        self._prop = self._calc._prop
        self._verbose = verbose

        self._element_strings = self._prop.params.elements
        self._n_types = len(self._element_strings)
        if self._n_types not in {1, 2}:
            raise RuntimeError("Structure list not found for systems beyond ternary.")

        os.makedirs(path_output, exist_ok=True)
        self._path_output = path_output
        self._path_header = self._path_output + "/" + "polymlp_"

        np.set_printoptions(legacy="1.21")

    @property
    def calc_api(self):
        """Return PypolymlpCalc API instance."""
        return self._calc

    @property
    def properties(self):
        """Return Properties instance."""
        return self._prop

    @property
    def pot(self):
        """Return polymlp name."""
        return self._pot

    @property
    def element_strings(self):
        """Return strings of elements."""
        return self._element_strings

    @property
    def n_types(self):
        """Return number of atom types."""
        return self._n_types

    @property
    def path_output(self):
        """Return directory path for generating files."""
        return self._path_output

    @property
    def path_header(self):
        """Return string header of files and directories in generating files."""
        return self._path_header
