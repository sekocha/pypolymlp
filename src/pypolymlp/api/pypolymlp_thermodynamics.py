"""API class for thermodynamics calculations."""

import copy
from typing import Optional

from pypolymlp.calculator.thermodynamics.fit_utils import fit_cv_temperature
from pypolymlp.calculator.thermodynamics.thermodynamics import load_yamls
from pypolymlp.calculator.thermodynamics.thermodynamics_utils import sum_matrix_data


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
        self._total = None

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
            f_ele = self._electron.get_data(attr="free_energy")
            s_ele = self._electron.get_data(attr="entropy")
            f_total = sum_matrix_data(f_total, f_ele)
            s_total = sum_matrix_data(s_total, s_ele)

        if self._ti is not None:
            if self._verbose:
                print("### TI contribution ###", flush=True)
            self._ti.fit_free_energy_temperature(max_order=6)
            self._ti.fit_cv_volume(max_order=4)
            f_ti = self._ti.get_data(attr="free_energy")
            s_ti = self._ti.get_data(attr="entropy")
            f_total_ti = sum_matrix_data(f_total, f_ti)
            s_total_ti = sum_matrix_data(s_total, s_ti)
        else:
            f_total_ti = f_total
            s_total_ti = s_total

        if self._electron is not None or self._ti is not None:
            if self._verbose:
                print("### Total properties ###", flush=True)
            self._total = copy.deepcopy(self._sscha)
            self._total.replace_free_energies(f_total_ti)
            self._total.replace_entropies(s_total_ti)
            self._total.fit_eos()
            self._total.fit_eval_entropy(max_order=6)

            self._total.replace_entropies(s_total, reset_fit=False)
            self._total.fit_eval_cp(max_order=4, from_entropy=True)
            self._total.replace_entropies(s_total_ti, reset_fit=False)

            if self._ti is not None:
                temperatures = self._total.temperatures
                eos_fits = self._total.fitted_models.eos_fits
                cv_fits_ti = self._ti.fitted_models.cv_fits
                # cv_ti = self._ti.get_data(attr="heat_capacity")
                cp_add = [
                    cv_fit.eval(eos.v0) for eos, cv_fit in zip(eos_fits, cv_fits_ti)
                ]
                cp_add = fit_cv_temperature(temperatures, cp_add, verbose=self._verbose)
                self._total.add_cp(cp_add)

        return self

    def save_sscha(self, filename: str = "polymlp_thermodynamics_sscha.yaml"):
        """Save fitted SSCHA properties."""
        self._sscha.save_thermodynamics_yaml(filename=filename)
        return self

    def save_total(self, filename: str = "polymlp_thermodynamics_total.yaml"):
        """Save fitted SSCHA properties."""
        self._total.save_thermodynamics_yaml(filename=filename)
        return self


#     def find_phase_transition(self, yaml1: str, yaml2: str):
#         """Find phase transition and its temperature.
#
#         Parameters
#         ----------
#         yaml1: sscha_properties.yaml for the first structure.
#         yaml2: sscha_properties.yaml for the second structure.
#         """
#         tc_linear, tc_quartic = find_transition(yaml1, yaml2)
#         return tc_linear, tc_quartic
#
#     def compute_phase_boundary(self, yaml1: str, yaml2: str):
#         """Compute phase boundary between two structures.
#
#         Parameters
#         ----------
#         yaml1: sscha_properties.yaml for the first structure.
#         yaml2: sscha_properties.yaml for the second structure.
#
#         Return
#         ------
#         boundary: [pressures, temperatures].
#         """
#         boundary = compute_phase_boundary(yaml1, yaml2)
#         return boundary
