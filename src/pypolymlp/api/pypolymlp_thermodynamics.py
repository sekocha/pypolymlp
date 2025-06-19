"""API class for thermodynamics calculations."""

import copy
from typing import Optional

import numpy as np

from pypolymlp.calculator.thermodynamics.io_utils import load_thermodynamics_yaml
from pypolymlp.calculator.thermodynamics.thermodynamics import (
    Thermodynamics,
    fit_cv_temperature,
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

    def _run_electron(self, f_total: np.ndarray, s_total: np.ndarray):
        """Include electronic contribution."""
        f_total, s_total = self._sum_properties(self._electron, f_total, s_total)
        self._sscha_el = self._replace_thermodynamics(f_total, s_total)
        self._sscha_el.run_standard()
        return f_total, s_total

    def _fit_ti(self, f_total: np.ndarray, s_total: np.ndarray):
        """Fit TI contribution."""
        self._ti.fit_free_energy_temperature(max_order=6)
        self._ti.fit_cv_volume(max_order=4)
        f_total_ti, s_total_ti = self._sum_properties(self._ti, f_total, s_total)
        return f_total_ti, s_total_ti

    def run_use_harmonic(self):
        """Fit results and evalulate equilibrium properties."""
        self._sscha.run_standard()

        if self._electron is None and self._ti is None:
            return self

        f_total = self._sscha.get_data(attr="free_energy")
        s_total = self._sscha.get_data(attr="entropy")
        if self._electron is not None:
            if self._verbose:
                print("### Electronic contribution ###", flush=True)
            f_total, s_total = self._run_electron(f_total, s_total)

        if self._ti is not None:
            if self._verbose:
                print("### TI contribution ###", flush=True)

            f_ti = self._ti.get_data(attr="free_energy")
            s_ti = self._ti.get_data(attr="entropy")
            if self._electron is not None:
                f_ele_ti, s_ele_ti = self._sum_properties(self._electron, f_ti, s_ti)
            else:
                f_ele_ti, s_ele_ti = f_ti, s_ti

            ele_ti = self._replace_thermodynamics(f_ele_ti, s_ele_ti)
            ele_ti.fit_entropy_temperature(max_order=6, reference=False)
            ele_ti.fit_cv_volume(max_order=4)
            print(ele_ti.get_data(attr="heat_capacity"))
            print(ele_ti.eval_heat_capacities(ele_ti.volumes))

            cv_fc2 = self._sscha.get_data(attr="harmonic_heat_capacity")

            if self._verbose:
                print("### Total contribution ###", flush=True)

            f_total, s_total = self._sum_properties(self._ti, f_total, s_total)
            self._total = self._replace_thermodynamics(f_total, s_total)
            self._total.replace_heat_capacities(cv_fc2)
            self._total.fit_free_energy_volume()
            self._total.fit_eval_entropy(max_order=6)
            self._total.fit_cv_volume(max_order=4)
            self._total.eval_cp_equilibrium()

            # temperatures = self._total.temperatures
            fv_fits = self._total.fitted_models.fv_fits
            cv_fits_ti = ele_ti.fitted_models.cv_fits
            cp_add = [cv.eval(fv.v0) for fv, cv in zip(fv_fits, cv_fits_ti)]
            print(cp_add)
            # cp_add = fit_cv_temperature(temperatures, cp_add, verbose=self._verbose)
            self._total.add_cp(cp_add)

        return self

    def run(self):
        """Fit results and evalulate equilibrium properties."""
        self._sscha.run_standard()

        if self._electron is None and self._ti is None:
            return self

        f_total = self._sscha.get_data(attr="free_energy")
        s_total = self._sscha.get_data(attr="entropy")
        if self._electron is not None:
            if self._verbose:
                print("### Electronic contribution ###", flush=True)
            f_total, s_total = self._run_electron(f_total, s_total)

        if self._ti is not None:
            if self._verbose:
                print("### TI contribution ###", flush=True)
            f_total, s_total = self._sum_properties(self._ti, f_total, s_total)
            self._total = self._replace_thermodynamics(f_total, s_total)
            self._total.run_standard()

        return self

    def run_use_md_energy(self):
        """Fit results and evalulate equilibrium properties."""
        self._sscha.run_standard()

        if self._electron is None and self._ti is None:
            return self

        f_total = self._sscha.get_data(attr="free_energy")
        s_total = self._sscha.get_data(attr="entropy")
        if self._electron is not None:
            if self._verbose:
                print("### Electronic contribution ###", flush=True)
            f_total, s_total = self._run_electron(f_total, s_total)

        if self._ti is not None:
            if self._verbose:
                print("### Total properties ###", flush=True)
            self._ti.fit_energy_temperature(max_order=6)
            self._ti.fit_cv_volume(max_order=4)

            f_total_ti, s_total_ti = self._sum_properties(self._ti, f_total, s_total)
            self._total = self._replace_thermodynamics(f_total_ti, s_total_ti)
            self._total.fit_free_energy_volume()
            self._total.fit_eval_entropy(max_order=6)

            if self._ti.is_heat_capacity:
                self._total.replace_entropies(s_total, reset_fit=False)
                self._total.fit_eval_heat_capacity(max_order=4, from_entropy=True)
                self._total.replace_entropies(s_total_ti, reset_fit=False)

                temperatures = self._total.temperatures
                fv_fits = self._total.fitted_models.fv_fits
                cv_fits_ti = self._ti.fitted_models.cv_fits
                cp_add = [cv.eval(fv.v0) for fv, cv in zip(fv_fits, cv_fits_ti)]
                cp_add = fit_cv_temperature(temperatures, cp_add, verbose=self._verbose)
                self._total.add_cp(cp_add)
            else:
                self._total.clear_heat_capacities()

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
