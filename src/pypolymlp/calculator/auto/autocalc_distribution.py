"""API Class for evaluating accuracy of distributions."""

from typing import Optional, Union

import numpy as np

from pypolymlp.calculator.auto.autocalc_base import AutoCalcBase
from pypolymlp.calculator.auto.autocalc_utils import find_endmembers
from pypolymlp.calculator.auto.figures_properties import (
    plot_energy_distribution,
    plot_prototype_prediction,
)
from pypolymlp.calculator.auto.structures_binary import get_structure_type_binary
from pypolymlp.calculator.auto.structures_element import get_structure_type_element
from pypolymlp.calculator.compute_formation_energies import PolymlpFormationEnergies
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpParams
from pypolymlp.core.interface_vasp import (
    parse_properties_from_vaspruns,
    parse_structures_from_vaspruns,
)
from pypolymlp.utils.atomic_energies.atomic_energies import get_atomic_energies


class AutoCalcDistribution(AutoCalcBase):
    """API Class for evaluating accuracy of distributions."""

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
        super().__init__(
            pot=pot,
            params=params,
            coeffs=coeffs,
            properties=properties,
            path_output=path_output,
            verbose=verbose,
        )
        self._comparison = None
        self._distribution_train = None
        self._distribution_test = None

        self._formation = None
        if self._n_types > 1:
            self._formation_mlp = PolymlpFormationEnergies(properties=self._prop)
            self._formation_dft = PolymlpFormationEnergies(
                elements=self._element_strings
            )

    def set_end_members_mlp(self, prototypes: list):
        """Set end_energies from polymlp."""
        if self._n_types == 1:
            raise RuntimeError("End members not required for elemental systems.")

        ends = find_endmembers(prototypes, self._element_strings)
        end_energies = np.array([prot.energy for prot in ends])
        self._formation_mlp.define_end_members(energies=end_energies)
        return self

    def set_end_members_dft(self, vaspruns: list):
        """Set endmembers from DFT calculations."""
        if self._n_types == 1:
            raise RuntimeError("End members not required for elemental systems.")
        pass

    def calc_energy_distribution(
        self,
        vaspruns_train: list,
        vaspruns_test: list,
        functional: str = "PBE",
    ):
        """Calculate properties for structures in training and test datasets."""
        if self._verbose:
            print("Compute energies for training and test data.")

        energies_dft, energies_mlp = self._eval_energies(vaspruns_train, functional)
        self._distribution_train = np.stack([energies_dft, energies_mlp]).T
        energies_dft, energies_mlp = self._eval_energies(vaspruns_test, functional)
        self._distribution_test = np.stack([energies_dft, energies_mlp]).T

        size = self._distribution_train.shape[0] + self._distribution_test.shape[0]
        with open(self._path_header + "size.dat", "w") as f:
            print(size, file=f)

        return self

    def compare_with_dft(
        self,
        vaspruns: list,
        icsd_ids: Optional[list] = None,
        functional: str = "PBE",
        filename: Optional[str] = None,
    ):
        """Calculate properties for DFT structures."""
        if self._verbose:
            print("Compute energies for structures.")

        energies_dft, energies_mlp = self._eval_energies(vaspruns, functional)

        names = self._set_structure_names(vaspruns, icsd_ids=icsd_ids)
        data = np.stack([energies_dft, energies_mlp, names]).T
        self._comparison = data[data[:, 0].argsort()]

        if filename is None:
            filename = self._path_header + "comparison.dat"
        header = "DFT (eV/atom), MLP (eV/atom), ID"
        np.savetxt(filename, self._comparison, fmt="%s", header=header)
        return self._comparison

    def _eval_energies(
        self,
        vaspruns: list,
        functional: str = "PBE",
        decimals: int = 6,
    ):
        """Evaluate MLP energies and DFT cohesive energies"""
        structures, (energies_dft, _, _) = parse_properties_from_vaspruns(vaspruns)
        atomic_energies = self._calc_atomic_energies(structures, functional=functional)
        energies_dft -= atomic_energies

        self._calc.structures = structures
        energies_mlp, _, _ = self._calc.eval()

        n_atom = np.array([np.sum(st.n_atoms) for st in structures])
        energies_dft /= n_atom
        energies_mlp /= n_atom
        return (np.round(energies_dft, decimals), np.round(energies_mlp, decimals))

    def _calc_atomic_energies(self, structures: list, functional: str = "PBE"):
        """Calculate atomic energies for structures."""
        atom_e = get_atomic_energies(
            elements=self._element_strings,
            functional=functional,
            return_dict=True,
        )
        atomic_energies = np.array(
            [np.sum([atom_e[ele] for ele in st.elements]) for st in structures]
        )
        return atomic_energies

    def _set_structure_names(self, vaspruns: list, icsd_ids: Optional[list] = None):
        """Set structure names."""
        if icsd_ids is not None:
            if self._n_types == 1:
                structure_type = get_structure_type_element()
            elif self._n_types == 2:
                structure_type = get_structure_type_binary()
            names = [
                "ICSD-" + str(i) + "-[" + structure_type[i] + "]" for i in icsd_ids
            ]
        else:
            names = vaspruns
        return np.array(names)

    def plot_energy_distribution(self, system: str, pot_id: str):
        """Plot comparison of mlp predictions with dft."""
        if self._distribution_train is None or self._distribution_test is None:
            raise RuntimeError("Distribution data not found.")

        plot_energy_distribution(
            self._distribution_train,
            self._distribution_test,
            system,
            pot_id,
            path_output=self._path_output,
        )
        return self

    def plot_comparison_with_dft(self, system: str, pot_id: str):
        """Plot comparison of mlp predictions with dft."""
        if self._comparison is None:
            raise RuntimeError("Comparison data not found.")

        plot_prototype_prediction(
            self._comparison,
            system,
            pot_id,
            path_output=self._path_output,
        )
        return self

    def run_formation_energy(
        self,
        vaspruns: Optional[list] = None,
        names: Optional[str] = None,
        geometry_optimization: bool = False,
    ):
        """Plot comparison of mlp predictions with dft."""
        if self._n_types < 2:
            raise RuntimeError(
                "Formation energy calculations not available for elemental system."
            )
        if self._formation is None:
            raise RuntimeError("PolymlpFormationEnergies class not defined.")

        if vaspruns is not None:
            structures = parse_structures_from_vaspruns(vaspruns)

        if geometry_optimization:
            structures_success = []
            energies_success = []
            for st in structures:
                self._calc.structures = st
                self._calc.init_geometry_optimization(
                    relax_cell=True, relax_volume=True
                )
                try:
                    energy, _, success = self._calc.run_geometry_optimization(
                        method="CG"
                    )
                    structures_success.append(self._calc.converged_structure)
                    energies_success.append(energy)
                except:
                    pass
                    # energy, success = None, False

            delta_e = self._formation.compute(
                structures=structures_success, energies=energies_success
            )
            self._formation.convex_hull()
            print(delta_e)
        else:
            delta_e = self._formation.compute(structures)
            self._formation.convex_hull()
            print(delta_e)
            return self

    @property
    def prototypes(self):
        return self._prototypes

    @prototypes.setter
    def prototypes(self, value: list):
        self._prototypes = value
