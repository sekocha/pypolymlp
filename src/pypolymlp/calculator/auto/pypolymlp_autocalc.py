"""API Class for systematically calculating properties."""

from typing import Optional, Union

import numpy as np

from pypolymlp.calculator.auto.autocalc_distribution import AutoCalcDistribution
from pypolymlp.calculator.auto.autocalc_prototypes import AutoCalcPrototypes
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
        self._auto_prot = AutoCalcPrototypes(
            pot=pot,
            params=params,
            coeffs=coeffs,
            properties=properties,
            path_output=path_output,
            verbose=verbose,
        )
        self._prop = self._auto_prot._prop

        self._auto_dist = AutoCalcDistribution(
            properties=self._prop,
            path_output=path_output,
            verbose=verbose,
        )
        self._verbose = verbose

    def run_prototypes(self):
        """Run calculations for prototype structures."""
        self._auto_prot.load_structures()
        self._auto_prot.run()
        return self

    def set_prototypes_from_DFT(self, vaspruns: list, icsd_ids: list):
        """Set DFT properties for prototypes."""
        self._auto_prot.set_dft_properties(vaspruns, icsd_ids)
        return self

    def save_prototypes(self):
        """Save properties of prototypes."""
        self._auto_prot.save_properties()
        return self

    def calc_energy_distribution(
        self,
        vaspruns_train: list,
        vaspruns_test: list,
        functional: str = "PBE",
    ):
        """Calculate properties for structures in training and test datasets."""
        self._auto_dist.calc_energy_distribution(
            vaspruns_train,
            vaspruns_test,
            functional=functional,
        )
        return self

    def calc_comparison_with_dft(
        self,
        vaspruns: list,
        icsd_ids: Optional[list] = None,
        functional: str = "PBE",
        filename: Optional[str] = None,
    ):
        """Calculate properties for DFT structures."""
        self._auto_dist.compare_with_dft(
            vaspruns=vaspruns,
            icsd_ids=icsd_ids,
            functional=functional,
            filename=filename,
        )
        return self

    def plot_energy_distribution(self, system: str, pot_id: str):
        """Plot comparison of mlp predictions with dft."""
        self._auto_dist.plot_energy_distribution(system, pot_id)
        return self

    def plot_comparison_with_dft(self, system: str, pot_id: str):
        """Plot comparison of mlp predictions with dft."""
        self._auto_dist.plot_comparison_with_dft(system, pot_id)
        return self

    #     def run_formation_energy(
    #         self,
    #         vaspruns: Optional[list] = None,
    #         names: Optional[str] = None,
    #         geometry_optimization: bool = False,
    #     ):
    #         """Plot comparison of mlp predictions with dft."""
    #         if self._n_types < 2:
    #             raise RuntimeError(
    #                 "Formation energy calculations not available for elemental system."
    #             )
    #         if self._formation is None:
    #             raise RuntimeError("PolymlpFormationEnergies class not defined.")
    #
    #         if vaspruns is not None:
    #             structures = parse_structures_from_vaspruns(vaspruns)
    #
    #         if geometry_optimization:
    #             structures_success = []
    #             energies_success = []
    #             for st in structures:
    #                 self._calc.structures = st
    #                 self._calc.init_geometry_optimization(
    #                     relax_cell=True, relax_volume=True
    #                 )
    #                 try:
    #                     energy, _, success = self._calc.run_geometry_optimization(
    #                         method="CG")
    #                     structures_success.append(self._calc.converged_structure)
    #                     energies_success.append(energy)
    #                 except:
    #                     pass
    #                     #energy, success = None, False
    #
    #             delta_e = self._formation.compute(
    #                 structures=structures_success, energies=energies_success
    #             )
    #             self._formation.convex_hull()
    #             print(delta_e)
    #         else:
    #             delta_e = self._formation.compute(structures)
    #             self._formation.convex_hull()
    #             print(delta_e)
    #             return self
    #
    @property
    def prototypes(self):
        return self._auto_prot.prototypes
