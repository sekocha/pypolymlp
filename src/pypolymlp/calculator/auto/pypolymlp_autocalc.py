"""API Class for systematically calculating properties."""

import os
from typing import Optional, Union

import numpy as np

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.calculator.auto.autocalc_utils import Prototype
from pypolymlp.calculator.auto.figures_properties import (
    plot_energy_distribution,
    plot_prototype_prediction,
)
from pypolymlp.calculator.auto.structures_binary import (
    get_structure_list_binary,
    get_structure_type_binary,
)
from pypolymlp.calculator.auto.structures_element import (
    get_structure_list_element,
    get_structure_type_element,
)
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpParams
from pypolymlp.core.interface_vasp import parse_properties_from_vaspruns
from pypolymlp.utils.atomic_energies.atomic_energies import get_atomic_energies
from pypolymlp.utils.spglib_utils import SymCell


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

        self._path_output = path_output
        os.makedirs(path_output, exist_ok=True)
        self._path_header = self._path_output + "/" + "polymlp_"
        self._prototypes = None

        self._comparison = None
        self._distribution_train = None
        self._distribution_test = None

        np.set_printoptions(legacy="1.21")

    def load_structures(self):
        """Load a list of initial structures from database."""
        if self._n_types == 1:
            self._prototypes = get_structure_list_element(self._element_strings)
        elif self._n_types == 2:
            self._prototypes = get_structure_list_binary(self._element_strings)
        return self._prototypes

    def run(self):
        """Calculate properties systematically for prototype structures."""
        if self._prototypes is None:
            raise RuntimeError("Prototype structures not found.")

        if self._verbose:
            self._print_targets()

        for prot in self._prototypes:
            path = self._path_header + prot.name + "/"
            os.makedirs(path, exist_ok=True)
            if self._verbose:
                print("---- Structure", prot.name, "----", flush=True)
            poscar = path + "POSCAR_eq"

            energy, success = self._run_geometry_optimization(prot)
            if not success:
                if self._verbose:
                    print("Warning: Geometry optimization failed.", flush=True)
                continue
            prot.structure_eq = self._calc.converged_structure
            self._calc.save_poscars(filename=poscar)

            self._run_eos(prot)
            self._run_elastic(prot, poscar)
            self._run_phonon(prot)
            # self._run_qha(prot)
        return self

    def save_properties(self):
        """Save properties."""
        for prot in self._prototypes:
            path = self._path_header + prot.name + "/"
            prot.save_properties(filename=path + "polymlp_predictions.yaml")
        return self

    def _print_targets(self):
        """Print target structures and polymlp."""
        print("##### Systematic calculations #####", flush=True)
        if self._pot is not None:
            if isinstance(self._pot, str):
                print("Polymlp:", os.path.abspath(self._pot), flush=True)
            else:
                print("Polymlp:", flush=True)
                for p in self._pot:
                    print("- ", os.path.abspath(p), flush=True)

        print("Target structures:", flush=True)
        for prot in self._prototypes:
            print("-", prot.name, flush=True)
        return self

    def _run_geometry_optimization(self, prototype: Prototype):
        """Run geometry optimization for single prototype."""
        self._calc.structures = prototype.structure
        self._calc.init_geometry_optimization(relax_cell=True, relax_volume=True)
        try:
            energy, _, success = self._calc.run_geometry_optimization(method="CG")
        except:
            energy, success = None, False
        return energy, success

    def _run_eos(self, prototype: Prototype):
        """Run EOS calculation and fit for single prototype."""
        if self._verbose:
            print("Run EOS calculations.", flush=True)
        self._calc.run_eos(eos_fit=True)
        e0, v0, b0 = self._calc.eos_fit_data
        eos_mlp, eos_fit = self._calc.eos_curve_data
        prototype.set_eos_data(e0, v0, b0, eos_mlp, eos_fit)
        return prototype

    def _run_elastic(self, prototype: Prototype, poscar: str):
        """Run elastic constant calculations."""
        if self._verbose:
            print("Run elastic constant calculations.", flush=True)
        prototype.elastic_constants = self._calc.run_elastic_constants(poscar=poscar)
        return prototype

    def _run_phonon(self, prototype: Prototype):
        """Run phonon calculations for single prototype."""
        if self._verbose:
            print("Run phonon calculations.", flush=True)
        supercell_matrix = np.diag(prototype.phonon_supercell)
        self._calc.init_phonon(supercell_matrix=supercell_matrix)
        self._calc.run_phonon(distance=0.01)
        self._calc.write_phonon(path=self._path_header + prototype.name)
        prototype.phonon_dos = self._calc.phonon_dos
        return prototype

    def _run_qha(self, prototype: Prototype):
        """Run QHA calculations for single prototype."""
        if self._calc.is_imaginary:
            return prototype

        if self._verbose:
            print("Run QHA calculations.", flush=True)
        supercell_matrix = np.diag(prototype.phonon_supercell)
        self._calc.run_qha(distance=0.01, supercell_matrix=supercell_matrix)
        self._calc.write_qha(path=self._path_header + prototype.name)
        prototype.set_qha_data(
            self._calc.temperatures,
            self._calc.thermal_expansion,
            self._calc.bulk_modulus_temperature,
        )
        return prototype

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
        self._comparison = data[energies_dft.argsort()]

        if filename is None:
            filename = self._path_header + "comparison.dat"
        header = "DFT (eV/atom), MLP (eV/atom), ID"
        np.savetxt(filename, self._comparison, fmt="%s", header=header)

        if icsd_ids is not None:
            self._set_dft_structures(self._calc.structures, icsd_ids)
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

    def _set_dft_structures(self, structures: list, icsd_ids: list):
        """Set DFT structures to prototypes."""
        structure_dict = dict(zip(icsd_ids, structures))
        for prot in self._prototypes:
            try:
                sym = SymCell(st=structure_dict[prot.icsd_id], symprec=1e-3)
                prot.structure_dft = sym.refine_cell()
            except:
                pass
        return self

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

    @property
    def prototypes(self):
        return self._prototypes
