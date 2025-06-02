"""API class for thermodynamics calculations."""

import copy
from typing import Optional

import numpy as np

from pypolymlp.calculator.thermodynamics.fit_utils import fit_cv_temperature
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
        verbose: bool = False,
    ):
        """Init method."""
        self._sscha, self._electron, self._ti = load_yamls(
            yamls_sscha, yamls_electron, yamls_ti, verbose
        )
        self._verbose = verbose
        self._sscha_el = None
        self._total = None

    def _replace_thermodynamics(self, free_energies: np.ndarray, entropies: np.ndarray):
        """Initialize Thermodynamics instance."""
        new = copy.deepcopy(self._sscha)
        new.replace_free_energies(free_energies)
        new.replace_entropies(entropies)
        return new

    def _sum_properties(
        self, thermodynamics: Thermodynamics, f_total: np.ndarray, s_total: np.ndarray
    ):
        """Calculate sums of properties."""
        f_contrib = thermodynamics.get_data(attr="free_energy")
        s_contrib = thermodynamics.get_data(attr="entropy")
        f_sum = sum_matrix_data(f_total, f_contrib)
        s_sum = sum_matrix_data(s_total, s_contrib)
        return f_sum, s_sum

    def _fit_eval_electron(self, f_total: np.ndarray, s_total: np.ndarray):
        """Include electronic contribution."""
        f_total, s_total = self._sum_properties(self._electron, f_total, s_total)
        self._sscha_el = self._replace_thermodynamics(f_total, s_total)
        self._sscha_el.fit_eval_sscha()
        return f_total, s_total

    def _fit_ti(self, f_total: np.ndarray, s_total: np.ndarray):
        """Fit TI contribution."""
        self._ti.fit_free_energy_temperature(max_order=6)
        self._ti.fit_cv_volume(max_order=4)
        f_total_ti, s_total_ti = self._sum_properties(self._ti, f_total, s_total)
        return f_total_ti, s_total_ti

    def run(self):
        """Fit results and evalulate equilibrium properties."""
        self._sscha.fit_eval_sscha()

        if self._electron is None and self._ti is None:
            return self

        f_total = self._sscha.get_data(attr="free_energy")
        s_total = self._sscha.get_data(attr="entropy")
        if self._electron is not None:
            if self._verbose:
                print("### Electronic contribution ###", flush=True)
            f_total, s_total = self._fit_eval_electron(f_total, s_total)

        if self._ti is not None:
            if self._verbose:
                print("### TI contribution ###", flush=True)
            f_total_ti, s_total_ti = self._fit_ti(f_total, s_total)
            # self._ti.fit_free_energy_temperature(max_order=6)
            # self._ti.fit_cv_volume(max_order=4)
            # f_total_ti, s_total_ti = self._sum_properties(self._ti, f_total, s_total)

            if self._verbose:
                print("### Total properties ###", flush=True)
            self._total = self._replace_thermodynamics(f_total_ti, s_total_ti)
            self._total.fit_eos()
            self._total.fit_eval_entropy(max_order=6)

            self._total.replace_entropies(s_total, reset_fit=False)
            self._total.fit_eval_cp(max_order=4, from_entropy=True)
            self._total.replace_entropies(s_total_ti, reset_fit=False)

            temperatures = self._total.temperatures
            eos_fits = self._total.fitted_models.eos_fits
            cv_fits_ti = self._ti.fitted_models.cv_fits
            cp_add = [cv_fit.eval(eos.v0) for eos, cv_fit in zip(eos_fits, cv_fits_ti)]
            # for t, c in zip(temperatures, cp_add):
            #     print(t, c)
            cp_add = fit_cv_temperature(temperatures, cp_add, verbose=self._verbose)
            self._total.add_cp(cp_add)

        return self

    def save_sscha(self, filename: str = "polymlp_thermodynamics_sscha.yaml"):
        """Save fitted SSCHA properties."""
        self._sscha.save_thermodynamics_yaml(filename=filename)
        return self

    def save_sscha_ele(self, filename: str = "polymlp_thermodynamics_sscha_ele.yaml"):
        """Save fitted SSCHA + electronic properties."""
        self._sscha_el.save_thermodynamics_yaml(filename=filename)
        return self

    def save_total(self, filename: str = "polymlp_thermodynamics_total.yaml"):
        """Save fitted SSCHA properties."""
        self._total.save_thermodynamics_yaml(filename=filename)
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
