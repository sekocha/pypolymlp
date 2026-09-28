"""API Class for systematically calculating properties."""

import os

import numpy as np

from pypolymlp.calculator.auto.autocalc_utils import AutoCalcBase, Prototype
from pypolymlp.calculator.auto.figures_properties import (
    plot_eos,
    plot_eos_separate,
    plot_phonon,
    plot_qha,
)
from pypolymlp.calculator.auto.structures_binary import get_structure_list_binary
from pypolymlp.calculator.auto.structures_element import get_structure_list_element
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.interface_vasp import parse_properties_from_vaspruns
from pypolymlp.utils.spglib_utils import SymCell


class AutoCalcPrototypes(AutoCalcBase):
    """API Class for systematically calculating properties."""

    def __init__(
        self,
        properties: Properties,
        path_output: str = ".",
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        properties: Properties instance.
        """
        super().__init__(properties, path_output=path_output, verbose=verbose)
        self._prototypes = None

    def load_structures(self):
        """Load a list of initial structures from database."""
        if self._n_types == 1:
            self._prototypes = get_structure_list_element(self._element_strings)
        elif self._n_types == 2:
            self._prototypes = get_structure_list_binary(self._element_strings)

        if self._n_types > 1:
            for ele in self._element_strings:
                for prot in get_structure_list_element([ele]):
                    prot.name = prot.name + "_" + ele
                    self._prototypes.append(prot)
        return self._prototypes

    def set_dft_properties(self, vaspruns: list, icsd_ids: list):
        """Set DFT properties for prototypes."""
        if self._prototypes is None:
            raise RuntimeError("Prototypes not found.")

        if len(vaspruns) != len(icsd_ids):
            raise RuntimeError("Inconsistent sizes of vaspruns and icsd IDs.")

        structures, (energies_dft, _, _) = parse_properties_from_vaspruns(vaspruns)
        structure_dict = dict(zip(icsd_ids, structures))
        for prot in self._prototypes:
            try:
                sym = SymCell(st=structure_dict[prot.icsd_id], symprec=1e-3)
                prot.structure_dft = sym.refine_cell()
            except:
                pass
        return self

    def run(self, run_qha: bool = True):
        """Calculate properties systematically for prototype structures."""
        if self._prototypes is None:
            raise RuntimeError("Prototype structures not found.")

        if self._verbose:
            self._print_targets()

        for prot in self._prototypes:
            if self._verbose:
                print("---- Structure", prot.name, "----", flush=True)

            path = self._path_header + prot.name + "/"
            os.makedirs(path, exist_ok=True)
            poscar = path + "POSCAR_eq"

            _, success = self._run_geometry_optimization(prot, poscar)
            if not success:
                continue
            self._run_eos(prot)
            self._run_elastic(prot, poscar)
            self._run_phonon(prot)
            if run_qha:
                self._run_qha(prot)

        return self

    def save_properties(self):
        """Save properties."""
        for prot in self._prototypes:
            path = self._path_header + prot.name + "/"
            prot.save_properties(filename=path + "polymlp_predictions.yaml")
        return self

    def plot_properties(self, system: str, pot_id: str):
        """Plot properties."""
        path = self._path_output
        if self._n_types == 1:
            plot_eos(self._prototypes, system, pot_id, path_output=path)

        plot_eos_separate(self._prototypes, system, pot_id, path_output=path)
        plot_phonon(self._prototypes, system, pot_id, path_output=path)
        plot_qha(
            self._prototypes,
            system,
            pot_id,
            target="thermal_expansion",
            path_output=path,
        )
        plot_qha(
            self._prototypes,
            system,
            pot_id,
            target="bulk_modulus",
            path_output=path,
        )
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

    def _run_geometry_optimization(self, prototype: Prototype, poscar: str):
        """Run geometry optimization for single prototype."""
        self._calc.structures = prototype.structure
        self._calc.init_geometry_optimization(relax_cell=True, relax_volume=True)
        try:
            energy, _, success = self._calc.run_geometry_optimization(method="CG")
        except:
            energy, success = None, False

        if success:
            prototype.structure_eq = self._calc.converged_structure
            n_atom = len(prototype.structure_eq.elements)
            prototype.energy = energy / n_atom
            prototype.volume = prototype.structure_eq.volume / n_atom
            self._calc.save_poscars(filename=poscar)
        else:
            if self._verbose:
                print("Warning: Geometry optimization failed.", flush=True)
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
        self._calc.write_phonon(
            path=self._path_header + prototype.name, write_fc2=False
        )
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

    @property
    def prototypes(self):
        return self._prototypes

    @prototypes.setter
    def prototypes(self, value: list):
        self._prototypes = value
