"""API wrapper to test API functions in C++ library."""

import numpy as np
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

    def compute_anlmtp_conjugate(
        self,
        anlmtp_r: NDArray,
        anlmtp_i: NDArray,
        type1: int,
    ):
        """Compute complex conjugate of anlmtp."""
        anlmtp = self._api.compute_anlmtp_conjugate(anlmtp_r, anlmtp_i, type1)
        return np.array(anlmtp)

    def compute_features_real(self, antp: NDArray, type1: int):
        """Compute features from antp."""
        features = self._api.compute_features_real(antp, type1)
        return np.array(features)

    def compute_features(self, anlmtp: NDArray, type1: int):
        """Compute features from anlmtp."""
        features = self._api.compute_features(anlmtp, type1)
        return np.array(features)

    def compute_features_deriv(
        self,
        anlmtp: NDArray,
        anlmtp_dfx: NDArray,
        anlmtp_dfy: NDArray,
        anlmtp_dfz: NDArray,
        anlmtp_ds: NDArray,
        type1: int,
    ):
        """Compute features from anlmtp."""
        dfx, dfy, dfz, ds = self._api.compute_features_deriv(
            anlmtp, anlmtp_dfx, anlmtp_dfy, anlmtp_dfz, anlmtp_ds, type1
        )
        return np.array(dfx), np.array(dfy), np.array(dfz), np.array(ds)

    def compute_sum_of_prod_antp(self, antp: NDArray, type1: int):
        """Compute products from antp."""
        prod_e, prod_f = self._api.compute_sum_of_prod_antp(antp, type1)
        return np.array(prod_e), np.array(prod_f)

    def compute_sum_of_prod_anlmtp(self, anlmtp: NDArray, type1: int):
        """Compute products from anlmtp."""
        prod_e, prod_f = self._api.compute_sum_of_prod_anlmtp(anlmtp, type1)
        return np.array(prod_e), np.array(prod_f)
