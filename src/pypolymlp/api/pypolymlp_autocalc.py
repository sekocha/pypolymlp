"""API Class for systematically calculating properties."""

from typing import Optional, Union

import numpy as np

from pypolymlp.calculator.auto.autocalc_distribution import AutoCalcDistribution
from pypolymlp.calculator.auto.autocalc_prototypes import AutoCalcPrototypes
from pypolymlp.calculator.properties import Properties, initialize_polymlp_calculator
from pypolymlp.core.params import PolymlpParams


class PypolymlpAutoCalc:
    """API Class for systematically calculating properties."""

    def __init__(
        self,
        pot: Optional[Union[str, list[str]]] = None,
        params: Optional[PolymlpParams] = None,
        coeffs: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
        properties: Optional[Properties] = None,
        path_output: str = ".",
        functional: str = "PBE",
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
        self._prop = initialize_polymlp_calculator(
            pot=pot,
            params=params,
            coeffs=coeffs,
            properties=properties,
        )
        self._auto_prot = AutoCalcPrototypes(
            self._prop,
            path_output=path_output,
            verbose=verbose,
        )
        self._verbose = verbose
        self._prop = self._auto_prot._prop
        self._n_types = self._auto_prot._n_types

        self._auto_dist = AutoCalcDistribution(
            self._prop,
            path_output=path_output,
            functional=functional,
            verbose=verbose,
        )

    def load_prototypes(self):
        """Load prototype structures."""
        self._auto_prot.load_structures()
        return self

    def calc_prototypes(self):
        """Calculate properties of prototype structures."""
        self._auto_prot.run()
        return self

    def run_prototypes(self):
        """Run calculations for prototype structures."""
        self.load_prototypes()
        self.calc_prototypes()
        return self

    def set_prototypes_from_DFT(self, vaspruns: list, icsd_ids: list):
        """Set DFT properties for prototypes."""
        self._auto_prot.set_dft_properties(vaspruns, icsd_ids)
        return self

    def save_prototypes(self):
        """Save properties of prototypes."""
        self._auto_prot.save_properties()
        return self

    def plot_prototypes(self, system: str, pot_id: str):
        """Plot properties of prototypes."""
        self._auto_prot.plot_properties(system, pot_id)
        return self

    def calc_comparison_with_dft(
        self,
        vaspruns: list,
        icsd_ids: Optional[list] = None,
        filename_suffix: Optional[str] = None,
    ):
        """Calculate properties for DFT structures."""
        self._auto_dist.compare_with_dft(
            vaspruns=vaspruns,
            icsd_ids=icsd_ids,
            filename_suffix=filename_suffix,
        )
        return self

    def calc_formation_energies(
        self,
        vaspruns: Optional[list] = None,
        icsd_ids: Optional[list] = None,
        geometry_optimization: bool = False,
    ):
        """run formation energy calcultions."""
        self._auto_dist.calc_formation_energies(
            vaspruns=vaspruns,
            icsd_ids=icsd_ids,
            geometry_optimization=geometry_optimization,
        )
        return self

    def calc_energy_distribution(
        self,
        vaspruns_train: list,
        vaspruns_test: list,
    ):
        """Calculate properties for structures in training and test datasets."""
        self._auto_dist.calc_energy_distribution(vaspruns_train, vaspruns_test)
        return self

    def plot_comparison_with_dft(
        self,
        system: str,
        pot_id: str,
        filename_suffix: Optional[str] = None,
    ):
        """Plot comparison of mlp predictions with dft."""
        self._auto_dist.plot_comparison_with_dft(
            system,
            pot_id,
            filename_suffix=filename_suffix,
        )
        return self

    def plot_binary_formation_energies(self, system: str, pot_id: str):
        """Plot formation energies."""
        self._auto_dist.plot_binary_formation_energies(system, pot_id)
        return self

    def plot_energy_distribution(self, system: str, pot_id: str):
        """Plot comparison of mlp predictions with dft."""
        self._auto_dist.plot_energy_distribution(system, pot_id)
        return self

    @property
    def prototypes(self):
        return self._auto_prot.prototypes

    @prototypes.setter
    def prototypes(self, values: list):
        self._auto_prot.prototypes = values
