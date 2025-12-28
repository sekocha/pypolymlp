"""API Class for systematically calculating properties."""

from typing import Optional, Union

import numpy as np

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.calculator.auto.structures_element import get_structure_list_element
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpParams
from pypolymlp.utils.structure_utils import get_lattice_constants


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
            pass
        return self._prototypes

    def run(self):
        """Calculate properties systematically for prototype structures."""
        if self._prototypes is None:
            raise ("Prototype structures not found.")

        for prot in self._prototypes:
            if self._verbose:
                print("---- Structure", prot.name, "----", flush=True)

            self._calc.structures = prot.structure
            self._calc.init_geometry_optimization(
                with_sym=True,
                relax_cell=True,
                relax_volume=True,
                relax_positions=True,
            )
            energy, _, success = self._calc.run_geometry_optimization()
            if not success:
                if self._verbose:
                    print("Geometry optimization failed.", flush=True)
                continue

            lattice_consts = get_lattice_constants(self._calc.converged_structure)
            poscar_name = "POSCAR-eq-" + prot.name
            self._calc.save_poscars(filename=poscar_name)

            self._calc.run_eos(eos_fit=True)
            self._calc.write_eos(filename="polymlp_eos_" + prot.name + ".yaml")

            elas = self._calc.run_elastic_constants(poscar=poscar_name)

            print(energy / prot.n_atom)
            print(lattice_consts)
            print(elas)
