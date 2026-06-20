"""API wrapper to test API functions in C++ library."""

from numpy.typing import NDArray

from pypolymlp.cxx.lib import libmlpcpp


class PolymlpCPPAPI:
    """API wrapper class."""

    def __init__(self):
        """Init method."""
        self._api = libmlpcpp.PolymlpAPI()

    def parse_polymlp_file(self, pot: str, elements: list, mass: list):
        """Parse polymlp file."""
        self._api.parse_polymlp_file(pot, elements, mass)

    def set_features(self, fp: libmlpcpp.FeatureParams):
        """Set feature class."""
        self._api.set_features(fp)

    def set_model_parameters(self, fp: libmlpcpp.FeatureParams):
        """Set model parameters."""
        self._api.set_model_parameters(fp)

    def set_potential_model(self, fp: libmlpcpp.FeatureParams, pot: NDArray):
        """Set potential model."""
        self._api.set_potential_model(fp, pot)

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

    @property
    def n_variables(self):
        """Return parameters for features."""
        return self._api.get_n_variables()
