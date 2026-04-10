"""API class for thermodynamics calculations."""

from typing import Optional

import numpy as np

from pypolymlp.calculator.thermodynamics.api_thermodynamics import (
    Thermodynamics,
    ThermodynamicsData,
    calculate_reference_grid,
    compute_grid_sum,
    load_thermodynamics_yaml,
    load_yamls,
    set_reference_paths,
)
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
        ref_fc2: Optional[list] = None,
        verbose: bool = False,
    ):
        """Init method."""
        if yamls_electron is None and yamls_electron_phonon is not None:
            raise RuntimeError("Specify path of electron.yaml files")

        self._verbose = verbose
        if self._verbose:
            np.set_printoptions(legacy="1.21")

        grids = self._load_grid_data(
            yamls_sscha=yamls_sscha,
            yamls_electron=yamls_electron,
            yamls_ti=yamls_ti,
            yamls_electron_phonon=yamls_electron_phonon,
            ref_fc2=ref_fc2,
        )
        self._thermo = self._calculate_grid_sums(grids)

    def _load_grid_data(
        self,
        yamls_sscha: list[str],
        yamls_electron: Optional[list[str]] = None,
        yamls_ti: Optional[list[str]] = None,
        yamls_electron_phonon: Optional[list[str]] = None,
        ref_fc2: Optional[list] = None,
    ):
        """Load yaml files and set grid data."""
        grid_sscha, grid_el, grid_ti, grid_ti_ext, grid_el_ph = load_yamls(
            yamls_sscha=yamls_sscha,
            yamls_electron=yamls_electron,
            yamls_ti=yamls_ti,
            yamls_electron_phonon=yamls_electron_phonon,
        )
        if grid_ti is not None:
            grid_ti = set_reference_paths(grid_ti, ref_fc2)
            grid_ti.copy_static_data(grid_sscha)
            grid_ti_ext = set_reference_paths(grid_ti_ext, ref_fc2)
            grid_ti_ext.copy_static_data(grid_sscha)
            grid_ref = calculate_reference_grid(grid_ti)
        else:
            grid_ref = None

        return (grid_sscha, grid_el, grid_ti, grid_ti_ext, grid_el_ph, grid_ref)

    def _get_thermodynamics(self, grid_list: list):
        """Get thermodynamics instance from list of grids."""
        grid = compute_grid_sum(grid_list)
        return Thermodynamics(grid, verbose=self._verbose)

    def _calculate_grid_sums(self, grids: tuple):
        """Calculate sums of properties on grid and set Thermodynamics instance."""
        grid_sscha, grid_el, grid_ti, grid_ti_ext, grid_el_ph, grid_ref = grids

        sscha = Thermodynamics(grid_sscha, verbose=self._verbose)
        self._thermo = ThermodynamicsData(sscha)

        if grid_el is not None:
            glist = [grid_sscha, grid_el]
            self._thermo.sscha_el = self._get_thermodynamics(glist)
            if grid_el_ph is not None:
                glist = [grid_sscha, grid_el, grid_el_ph]
                self._thermo.sscha_el_ph = self._get_thermodynamics(glist)

        if grid_ti is not None:
            glist = [grid_ref, grid_ti]
            self._thermo.ti = self._get_thermodynamics(glist)
            glist = [grid_ref, grid_ti_ext]
            self._thermo.ti_ext = self._get_thermodynamics(glist)
            if grid_el is not None:
                glist = [grid_ref, grid_ti, grid_el]
                self._thermo.ti_el = self._get_thermodynamics(glist)
                glist = [grid_ref, grid_ti_ext, grid_el]
                self._thermo.ti_ext_el = self._get_thermodynamics(glist)
                if grid_el_ph is not None:
                    glist = [grid_ref, grid_ti, grid_el, grid_el_ph]
                    self._thermo.ti_el_ph = self._get_thermodynamics(glist)
                    glist = [grid_ref, grid_ti_ext, grid_el, grid_el_ph]
                    self._thermo.ti_ext_el_ph = self._get_thermodynamics(glist)
        return self._thermo

    def run(self):
        """Fit results and evalulate equilibrium properties."""
        self._thermo.run(verbose=self._verbose)
        return self

    def save(self, path: str = "polymlp_thermodynamics"):
        """Save fitted and equilibrium properties."""
        self._thermo.save(path=path)
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
