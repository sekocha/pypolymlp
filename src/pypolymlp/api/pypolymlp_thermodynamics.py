"""API class for thermodynamics calculations."""

import copy
from typing import Optional

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
        self._sscha, self._electron, self._ti, self._ti_correction = load_yamls(
            yamls_sscha=yamls_sscha,
            yamls_electron=yamls_electron,
            yamls_ti=yamls_ti,
            verbose=verbose,
        )
        self._verbose = verbose
        self._sscha_el = None
        self._total = None

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

    def run(self):
        """Fit results and evalulate equilibrium properties."""
        if self._verbose:
            print("# ----- SSCHA contribution ----- #", flush=True)
        self._sscha.fit_free_energy_volume()
        self._sscha.fit_entropy_volume(max_order=6)
        self._sscha.eval_entropy_equilibrium()

        self._sscha.fit_entropy_temperature(max_order=5)
        self._sscha.fit_cv_volume(max_order=6)
        self._sscha.eval_cp_equilibrium()

        if self._electron is None and self._ti is None:
            return self

        if self._electron is not None:
            if self._verbose:
                print("# ----- Electronic contribution ----- #", flush=True)
            self._sscha_el = self._get_sum_properties(self._sscha, self._electron)

            self._sscha_el.fit_free_energy_volume()
            self._sscha_el.fit_entropy_volume(max_order=6)
            self._sscha_el.eval_entropy_equilibrium()

            self._sscha_el.fit_entropy_temperature(max_order=5)
            self._sscha_el.fit_cv_volume(max_order=6)
            self._sscha_el.eval_cp_equilibrium()

        if self._ti is not None:
            if self._verbose:
                print(
                    "# ----- Thermodynamic integration contribution ----- #", flush=True
                )
            self._ti = self._get_sum_properties(self._ti, self._ti_correction)
            if self._electron is not None:
                self._total = self._get_sum_properties(self._sscha_el, self._ti)
            else:
                self._total = self._get_sum_properties(self._sscha, self._ti)

            self._total.fit_free_energy_volume()
            self._total.fit_entropy_volume(max_order=6)
            self._total.eval_entropy_equilibrium()

            self._total.fit_entropy_temperature(max_order=4)
            self._total.fit_cv_volume(max_order=4)
            self._total.eval_cp_equilibrium()

        return self

    def save_sscha(self, filename: str = "polymlp_thermodynamics_sscha.yaml"):
        """Save fitted SSCHA properties."""
        self._sscha.save_thermodynamics_yaml(filename=filename)
        sp = filename.split(".yaml")
        filedata = "".join(sp[:-1]) + "_grid.yaml"
        self._sscha.save_data(filename=filedata)
        return self

    def save_sscha_ele(self, filename: str = "polymlp_thermodynamics_sscha_ele.yaml"):
        """Save fitted SSCHA + electronic properties."""
        self._sscha_el.save_thermodynamics_yaml(filename=filename)
        sp = filename.split(".yaml")
        filedata = "".join(sp[:-1]) + "_grid.yaml"
        self._sscha_el.save_data(filename=filedata)
        return self

    def save_total(self, filename: str = "polymlp_thermodynamics_total.yaml"):
        """Save fitted SSCHA properties."""
        self._total.save_thermodynamics_yaml(filename=filename)
        sp = filename.split(".yaml")
        filedata = "".join(sp[:-1]) + "_grid.yaml"
        self._total.save_data(filename=filedata)
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


#    def run_use_harmonic(self):
#        """Fit results and evalulate equilibrium properties."""
#        self._sscha.run_standard()
#
#        if self._electron is None and self._ti is None:
#            return self
#
#        f_total = self._sscha.get_data(attr="free_energy")
#        s_total = self._sscha.get_data(attr="entropy")
#        if self._electron is not None:
#            if self._verbose:
#                print("### Electronic contribution ###", flush=True)
#            f_total, s_total = self._run_electron(f_total, s_total)
#
#        if self._ti is not None:
#            if self._verbose:
#                print("### TI contribution ###", flush=True)
#
#            f_ti = self._ti.get_data(attr="free_energy")
#            s_ti = self._ti.get_data(attr="entropy")
#            if self._electron is not None:
#                f_ele_ti, s_ele_ti = self._sum_properties(self._electron, f_ti, s_ti)
#            else:
#                f_ele_ti, s_ele_ti = f_ti, s_ti
#
#            ele_ti = self._replace_thermodynamics(f_ele_ti, s_ele_ti)
#            ele_ti.fit_entropy_temperature(max_order=6, reference=False)
#            ele_ti.fit_cv_volume(max_order=4)
#            print(ele_ti.get_data(attr="heat_capacity"))
#            print(ele_ti.eval_heat_capacities(ele_ti.volumes))
#
#            cv_fc2 = self._sscha.get_data(attr="harmonic_heat_capacity")
#
#            if self._verbose:
#                print("### Total contribution ###", flush=True)
#
#            f_total, s_total = self._sum_properties(self._ti, f_total, s_total)
#            self._total = self._replace_thermodynamics(f_total, s_total)
#            self._total.replace_heat_capacities(cv_fc2)
#            self._total.fit_free_energy_volume()
#            self._total.fit_eval_entropy(max_order=6)
#            self._total.fit_cv_volume(max_order=4)
#            self._total.eval_cp_equilibrium()
#
#            # temperatures = self._total.temperatures
#            fv_fits = self._total.fitted_models.fv_fits
#            cv_fits_ti = ele_ti.fitted_models.cv_fits
#            cp_add = [cv.eval(fv.v0) for fv, cv in zip(fv_fits, cv_fits_ti)]
#            print(cp_add)
#            # cp_add = fit_cv_temperature(temperatures, cp_add, verbose=self._verbose)
#            self._total.add_cp(cp_add)
#
#        return self
#
#    def run_complete(self):
#        """Fit results and evalulate equilibrium properties."""
#        self._sscha.run_standard()
#
#        if self._electron is None and self._ti is None:
#            return self
#
#        f_total = self._sscha.get_data(attr="free_energy")
#        s_total = self._sscha.get_data(attr="entropy")
#        if self._electron is not None:
#            if self._verbose:
#                print("### Electronic contribution ###", flush=True)
#            f_total, s_total = self._run_electron(f_total, s_total)
#            # f_total, s_total = self._sum_properties(self._electron, f_total, s_total)
#            # self._sscha_el = self._replace_thermodynamics(f_total, s_total)
#            # self._sscha_el.run_standard()
#
#        if self._ti is not None:
#            if self._verbose:
#                print("### TI contribution ###", flush=True)
#            f_total, s_total = self._sum_properties(self._ti, f_total, s_total)
#            self._total = self._replace_thermodynamics(f_total, s_total)
#            self._total.run_standard()
#
#        return self
#
#

#    def run_use_md_energy(self):
#        """Fit results and evalulate equilibrium properties."""
#        self._sscha.run_standard()
#
#        if self._electron is None and self._ti is None:
#            return self
#
#        f_total = self._sscha.get_data(attr="free_energy")
#        s_total = self._sscha.get_data(attr="entropy")
#        if self._electron is not None:
#            if self._verbose:
#                print("### Electronic contribution ###", flush=True)
#            f_total, s_total = self._run_electron(f_total, s_total)
#
#        if self._ti is not None:
#            if self._verbose:
#                print("### Total properties ###", flush=True)
#            self._ti.fit_energy_temperature(max_order=6)
#            self._ti.fit_cv_volume(max_order=4)
#
#            f_total_ti, s_total_ti = self._sum_properties(self._ti, f_total, s_total)
#            self._total = self._replace_thermodynamics(f_total_ti, s_total_ti)
#            self._total.fit_free_energy_volume()
#            self._total.fit_eval_entropy(max_order=6)
#
#            if self._ti.is_heat_capacity:
#                self._total.replace_entropies(s_total, reset_fit=False)
#                self._total.fit_eval_heat_capacity(max_order=4, from_entropy=True)
#                self._total.replace_entropies(s_total_ti, reset_fit=False)
#
#                temperatures = self._total.temperatures
#                fv_fits = self._total.fitted_models.fv_fits
#                cv_fits_ti = self._ti.fitted_models.cv_fits
#                cp_add = [cv.eval(fv.v0) for fv, cv in zip(fv_fits, cv_fits_ti)]
#                cp_add = fit_cv_temperature(temperatures, cp_add, verbose=self._verbose)
#                self._total.add_cp(cp_add)
#            else:
#                self._total.clear_heat_capacities()
#
#        return self
#
