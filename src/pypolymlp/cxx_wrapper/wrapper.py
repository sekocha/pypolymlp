"""API wrapper to test API functions in C++ library."""

from pypolymlp.cxx.lib import libmlpcpp


class PolymlpCPPAPI:
    """API wrapper class."""

    def __init__(self):
        """Init method."""
        self._api = libmlpcpp.PolymlpAPI()

    def parse_polymlp_file(self, pot: str, elements: list, mass: list):
        """Parse polymlp file."""
        self._api.parse_polymlp_file(pot, elements, mass)

    def convert_unit(
        self,
        energy_conv: float = 1.0,
        length_conv: float = 1.0,
        inv_length_conv: float = 1.0,
    ):
        """Convert units."""
        self._api.convert_unit(energy_conv, length_conv, inv_length_conv)

    @property
    def feature_params(self):
        """Return parameters for features."""
        return self._api.get_fp()
