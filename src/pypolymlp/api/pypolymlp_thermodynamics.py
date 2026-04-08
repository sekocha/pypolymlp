"""API class for thermodynamics calculations."""

from typing import Optional

import numpy as np

from pypolymlp.calculator.thermodynamics.api_thermodynamics import Thermodynamics
from pypolymlp.calculator.thermodynamics.thermodynamics_grid import sum_grids
from pypolymlp.calculator.thermodynamics.thermodynamics_io import (
    load_thermodynamics_yaml,
)
from pypolymlp.calculator.thermodynamics.thermodynamics_parser import load_yamls
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
        extrapolation_ti: bool = False,
        # extrapolation_ti: bool = True,
        verbose: bool = False,
    ):
        """Init method."""
        if yamls_electron is None and yamls_electron_phonon is not None:
            raise RuntimeError("Specify path of electron.yaml files")

        grid_sscha, grid_electron, grid_ti = load_yamls(
            yamls_sscha=yamls_sscha,
            yamls_electron=yamls_electron,
            yamls_ti=yamls_ti,
            # yamls_electron_phonon=yamls_electron_phonon,
            # extrapolation_ti=extrapolation_ti,
        )
        self._sscha = Thermodynamics(grid_sscha, verbose=verbose)

        self._sscha_el = None
        if grid_electron is not None:
            grid = sum_grids([grid_sscha, grid_electron])
            self._sscha_el = Thermodynamics(grid, verbose=verbose)

        self._sscha_el_ti = None
        if grid_ti is not None:
            grid = sum_grids([grid_sscha, grid_electron, grid_ti])
            self._sscha_el_ti = Thermodynamics(grid, verbose=verbose)

        self._verbose = verbose
        if self._verbose:
            np.set_printoptions(legacy="1.21")

    def _run_standard(self, thermo: Thermodynamics):
        """Use a standard fitting procedure."""
        thermo.fit_free_energy_volume(max_order=4)
        thermo.fit_entropy_volume(max_order=4)
        thermo.eval_entropy_equilibrium()
        thermo.eval_cp_numerical()

        # thermo.fit_entropy_temperature(max_order=4)
        # thermo.fit_cv_volume(max_order=4)
        # thermo.eval_cp_equilibrium()
        return thermo

    def run(self):
        """Fit results and evalulate equilibrium properties."""
        if self._verbose:
            print("# ------- SSCHA ------- #", flush=True)
        self._sscha = self._run_standard(self._sscha)

        if self._sscha_el is not None:
            if self._verbose:
                print("# ----- SSCHA + Electron ----- #", flush=True)
            self._sscha_el = self._run_standard(self._sscha_el)

        if self._sscha_el_ti is not None:
            if self._verbose:
                print("# --- SSCHA + TI + Electron --- #", flush=True)
            self._sscha_el_ti = self._run_standard(self._sscha_el_ti)

        # if self._electron_ph is not None:
        #     if self._verbose:
        #         print(
        #             "# --- Include adiabatic ele-ph contribution --- #", flush=True
        #         )
        #     self._total_ele_ph = self._run_standard(
        #         self._total_ele_ph,
        #         assign_fit_values=True,
        #     )
        #
        return self

    def save_sscha(self, filename: str = "polymlp_thermodynamics_sscha.yaml"):
        """Save fitted SSCHA properties."""
        if self._sscha is not None:
            self._sscha.save_thermodynamics_yaml(filename=filename)
        return self

    def save_sscha_ele(self, filename: str = "polymlp_thermodynamics_sscha_ele.yaml"):
        """Save fitted SSCHA + electronic properties."""
        if self._sscha_el is not None:
            self._sscha_el.save_thermodynamics_yaml(filename=filename)
        return self

    def save_total(self, filename: str = "polymlp_thermodynamics_total.yaml"):
        """Save fitted SSCHA + electronic + TI properties."""
        if self._total is not None:
            self._total.save_thermodynamics_yaml(filename=filename)
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

        if self._verbose:
            self._print_logs()

    def _print_logs(self):
        """Print logs."""
        temp1 = self._prop1.temperatures
        temp2 = self._prop2.temperatures
        min_temp = max(min(temp1), min(temp2))
        max_temp = min(max(temp1), max(temp2))
        print("Common temperature range (K):", min_temp, "--", max_temp, flush=True)
        p1 = max([np.min(g[:, 0]) for g in self._prop1.gibbs])
        p2 = max([np.min(g[:, 0]) for g in self._prop2.gibbs])
        min_p = np.round(max(p1, p2), 1)
        p1 = min([np.max(g[:, 0]) for g in self._prop1.gibbs])
        p2 = min([np.max(g[:, 0]) for g in self._prop2.gibbs])
        max_p = np.round(min(p1, p2), 1)
        print("Common pressure range (GPa): ", min_p, "--", max_p, flush=True)
        return self

    def find_phase_transition(self):
        """Find phase transition and its temperature."""
        tc = find_transition(self._prop1.get_T_F(), self._prop2.get_T_F())
        return tc

    def compute_phase_boundary(
        self,
        pressure_interval: float = 0.25,
        fit_gibbs_max_order: int = 4,
    ):
        """Compute phase boundary between two structures.

        Upper bound of pressure is automatically determined.

        Parameters
        ----------
        pressure_interval: Pressure interval (GPa).
        fit_gibbs_max_order: Maximum order of pressure-G polynomial fitting.
        """
        boundary = compute_phase_boundary(
            self._prop1.gibbs,
            self._prop1.temperatures,
            self._prop2.gibbs,
            self._prop2.temperatures,
            pressure_interval=pressure_interval,
            fit_gibbs_max_order=fit_gibbs_max_order,
        )
        return boundary
