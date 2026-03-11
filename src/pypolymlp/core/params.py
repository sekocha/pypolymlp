"""Class for input parameters including hybrid polymlps."""

# import numpy as np

# from pypolymlp.core.data_format import PolymlpParams


class PolymlpParams:
    """Class for input parameters including hybrid polymlps."""

    def __init__(self):
        """Init method."""
        self._params = []

    def __iter__(self):
        """Iter method."""
        return iter(self._params)

    def append(self):
        """Append parameters."""
