"""API class for thermodynamics calculations."""

import copy
from typing import Optional

import numpy as np

from pypolymlp.calculator.thermodynamics.io_utils import load_thermodynamics_yaml
from pypolymlp.calculator.thermodynamics.thermodynamics import (
    Thermodynamics,
    load_yamls,
)
from pypolymlp.calculator.thermodynamics.thermodynamics_utils import sum_matrix_data
from pypolymlp.calculator.thermodynamics.transition import (
    compute_phase_boundary,
    find_transition,
)


class PypolymlpThermodynamics:
    """API class for thermodynamics calculations."""

    def __init__(
        self,
        yamls_sscha: list[str],
        yamls_electron: Optional[list[str]] = None,
        yamls_ti: Optional[list[str]] = None,
        yamls_electron_phonon: Optional[list[str]] = None,
        verbose: bool = False,
    ):
        """Init method."""
        if yamls_electron is None and yamls_electron_phonon is not None:
            raise RuntimeError("Specify path of electron.yaml files")

        (self._sscha, self._electron, self._ti, self._ti_ref, self._electron_ph) = (
            load_yamls(
                yamls_sscha=yamls_sscha,
                yamls_electron=yamls_electron,
                yamls_ti=yamls_ti,
                yamls_electron_phonon=yamls_electron_phonon,
                verbose=verbose,
            )
        )
        self._verbose = verbose
        self._sscha_el = None
        self._total = None

        if self._verbose:
            np.set_printoptions(legacy="1.21")

    def _get_sum_properties(
        self,
        thermodynamics1: Thermodynamics,
        thermodynamics2: Thermodynamics,
    ):
        """Calculate sums of properties."""
        f1 = thermodynamics1.get_data(attr="free_energy")
        s1 = thermodynamics1.get_data(attr="entropy")
        f2 = thermodynamics2.get_data(attr="free_energy")
        s2 = thermodynamics2.get_data(attr="entropy")
        f_sum = sum_matrix_data(f1, f2)
        s_sum = sum_matrix_data(s1, s2)
        new = copy.deepcopy(thermodynamics1)
        new.replace_free_energies(f_sum)
        new.replace_entropies(s_sum)
        return new

    def _run_standard(self, thermo: Thermodynamics, assign_fit_values: bool = False):
        """Use a standard fitting procedure."""
        thermo.fit_free_energy_volume()
        thermo.fit_entropy_volume(max_order=6, assign_fit_values=assign_fit_values)
        thermo.eval_entropy_equilibrium()

        thermo.fit_entropy_temperature(max_order=4)
        try:
            thermo.fit_cv_volume(max_order=4)
            thermo.eval_cp_equilibrium()
        except:
            if self._verbose:
                print("ERROR: Volume-Cv fit failed.", flush=True)
            return None
        return thermo

    def run(self):
        """Fit results and evalulate equilibrium properties."""
        if self._verbose:
            print("# ----- SSCHA contribution ----- #", flush=True)
        self._sscha = self._run_standard(self._sscha, assign_fit_values=True)

        if self._electron is not None:
            if self._verbose:
                print("# ----- Electronic contribution ----- #", flush=True)
            self._sscha_el = self._get_sum_properties(self._sscha, self._electron)
            self._sscha_el = self._run_standard(self._sscha_el, assign_fit_values=True)

        if self._ti is not None:
            if self._verbose:
                print("# --- Thermodynamic integration contribution --- #", flush=True)
            self._total = self._get_sum_properties(self._ti, self._ti_ref)
            if self._electron is not None:
                self._total = self._get_sum_properties(self._electron, self._total)

            self._total = self._run_standard(self._total, assign_fit_values=True)

            if self._electron_ph is not None:
                if self._verbose:
                    print(
                        "# --- Include adiabatic ele-ph contribution --- #", flush=True
                    )
                self._total_ele_ph = self._get_sum_properties(self._ti, self._ti_ref)
                self._total_ele_ph = self._get_sum_properties(
                    self._total_ele_ph,
                    self._electron,
                )
                self._total_ele_ph = self._get_sum_properties(
                    self._total_ele_ph,
                    self._electron_ph,
                )
                self._total_ele_ph = self._run_standard(
                    self._total_ele_ph,
                    assign_fit_values=True,
                )

        return self

    def save_sscha(self, filename: str = "polymlp_thermodynamics_sscha.yaml"):
        """Save fitted SSCHA properties."""
        if self._sscha is not None:
            self._sscha.save_thermodynamics_yaml(filename=filename)
        # sp = filename.split(".yaml")
        # filedata = "".join(sp[:-1]) + "_grid.yaml"
        # self._sscha.save_data(filename=filedata)
        return self

    def save_sscha_ele(self, filename: str = "polymlp_thermodynamics_sscha_ele.yaml"):
        """Save fitted SSCHA + electronic properties."""
        if self._sscha_el is not None:
            self._sscha_el.save_thermodynamics_yaml(filename=filename)
        # sp = filename.split(".yaml")
        # filedata = "".join(sp[:-1]) + "_grid.yaml"
        # self._sscha_el.save_data(filename=filedata)
        return self

    def save_total(self, filename: str = "polymlp_thermodynamics_total.yaml"):
        """Save fitted SSCHA + electronic + TI properties."""
        if self._total is not None:
            self._total.save_thermodynamics_yaml(filename=filename)
        # sp = filename.split(".yaml")
        # filedata = "".join(sp[:-1]) + "_grid.yaml"
        # self._total.save_data(filename=filedata)
        return self

    def save_total_ele_ph(
        self, filename: str = "polymlp_thermodynamics_total_ele_ph.yaml"
    ):
        """Save fitted SSCHA + electronic + TI + ele-ph properties."""
        if self._total_ele_ph is not None:
            self._total_ele_ph.save_thermodynamics_yaml(filename=filename)
        return self


class PypolymlpTransition:
    """API class for finding phase boundary."""

    def __init__(self, yaml1: str, yaml2: str, verbose: bool = False):
        """Init method.

        Parameters
        ----------
        yaml1: polymlp_thermodynamics.yaml for the first structure.
        yaml2: polymlp_thermodynamics.yaml for the second structure.
        """
        self._verbose = verbose
        self._prop1 = load_thermodynamics_yaml(yaml1)
        self._prop2 = load_thermodynamics_yaml(yaml2)

    def find_phase_transition(self):
        """Find phase transition and its temperature."""
        tc = find_transition(self._prop1.get_T_F(), self._prop2.get_T_F())
        return tc

    def compute_phase_boundary(self):
        """Compute phase boundary between two structures."""
        boundary = compute_phase_boundary(
            self._prop1.gibbs,
            self._prop1.temperatures,
            self._prop2.gibbs,
            self._prop2.temperatures,
        )
        return boundary
