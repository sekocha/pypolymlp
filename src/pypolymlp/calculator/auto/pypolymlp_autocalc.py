"""API Class for systematically calculating properties."""

import os
from typing import Optional, Union

import numpy as np

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.calculator.auto.dataclass import Prototype
from pypolymlp.calculator.auto.structures_binary import (  # get_structure_type_element,
    get_structure_list_binary,
)
from pypolymlp.calculator.auto.structures_element import (
    get_structure_list_element,
    get_structure_type_element,
)
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpParams
from pypolymlp.core.interface_vasp import parse_properties_from_vaspruns
from pypolymlp.utils.atomic_energies.atomic_energies import get_atomic_energies


class PypolymlpAutoCalc:
    """API Class for systematically calculating properties."""

    def __init__(
        self,
        pot: Union[str, list[str]] = None,
        params: Union[PolymlpParams, list[PolymlpParams]] = None,
        coeffs: Union[np.ndarray, list[np.ndarray]] = None,
        properties: Optional[Properties] = None,
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

        self._prototypes = None

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
            path = "polymlp_" + prot.name + "/"
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
            prot.save_properties(filename=path + "polymlp_predictions.yaml")

    def _print_targets(self):
        """Print target structures and polymlp."""
        print("##### Systematic calculations #####", flush=True)
        if self._pot is not None:
            print("Polymlp:", os.path.abspath(self._pot), flush=True)
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
        self._calc.run_eos(eos_fit=True)
        e0, v0, b0 = self._calc.eos_fit_data
        eos_mlp, eos_fit = self._calc.eos_curve_data
        prototype.energy = e0 / prototype.n_atom
        prototype.volume = v0 / prototype.n_atom
        prototype.bulk_modulus = b0
        prototype.eos_mlp = eos_mlp / prototype.n_atom
        prototype.eos_fit = eos_fit / prototype.n_atom
        return prototype

    def _run_elastic(self, prototype: Prototype, poscar: str):
        """Run elastic constant calculations."""
        prototype.elastic_constants = self._calc.run_elastic_constants(poscar=poscar)
        return prototype

    def _run_phonon(self, prototype: Prototype):
        """Run phonon calculations for single prototype."""
        supercell_matrix = np.diag(prototype.phonon_supercell)
        self._calc.init_phonon(supercell_matrix=supercell_matrix)
        self._calc.run_phonon(
            distance=0.01,
            mesh=(10, 10, 10),
            t_min=0,
            t_max=1000,
            t_step=10,
            with_eigenvectors=False,
            is_mesh_symmetry=True,
            with_pdos=False,
        )
        self._calc.write_phonon(path="polymlp_" + prototype.name)
        return prototype

    def compare_with_dft(
        self,
        vaspruns: list,
        icsd_ids: Optional[list] = None,
        functional: str = "PBE",
        filename: str = "polymlp_comparison.dat",
    ):
        """Calculate properties for DFT structures."""
        if self._verbose:
            print("Compute energies for structures.")

        structures, (energies_dft, _, _) = parse_properties_from_vaspruns(vaspruns)
        atomic_energies = self._calc_atomic_energies(structures, functional=functional)
        energies_dft -= atomic_energies

        self._calc.structures = structures
        energies_mlp, _, _ = self._calc.eval()

        n_atom = np.array([np.sum(st.n_atoms) for st in structures])
        energies_dft /= n_atom
        energies_mlp /= n_atom
        sorted_indices = energies_dft.argsort()

        names = self._set_structure_names(vaspruns, icsd_ids=icsd_ids)
        data = np.stack([np.round(energies_dft, 6), np.round(energies_mlp, 6), names]).T
        header = "DFT (eV/atom), MLP (eV/atom), ID"
        np.savetxt(filename, data[sorted_indices], fmt="%s", header=header)

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
                pass
            names = [
                "ICSD-" + str(i) + "-[" + structure_type[i] + "]" for i in icsd_ids
            ]
        else:
            names = vaspruns
        return np.array(names)

    @property
    def prototypes(self):
        return self._prototypes
