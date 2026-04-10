"""Classes for data on volume-temperature grid."""

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pypolymlp.calculator.sscha.sscha_restart import Restart
from pypolymlp.calculator.utils.eos_utils import EOS
from pypolymlp.calculator.utils.fit_utils import Polyfit
from pypolymlp.core.units import EVtoJmol


@dataclass
class GridPointData:
    """Dataclass for properties on a volume-temperature grid point."""

    volume: float
    temperature: float
    restart: Optional[Restart] = None

    free_energy: Optional[float] = None
    entropy: Optional[float] = None
    heat_capacity: Optional[float] = None

    energy: Optional[float] = None
    static_potential: Optional[float] = None

    path_yaml: Optional[float] = None
    path_fc2: Optional[float] = None

    def _add_single(self, gp_data, attr: str):
        """Add value to single attribute."""
        if gp_data is None:
            return None
        if getattr(self, attr) is not None and getattr(gp_data, attr) is not None:
            return getattr(self, attr) + getattr(gp_data, attr)
        return None

    def add(self, gp_data):
        """Add data."""
        attrs = [
            "free_energy",
            "entropy",
            "heat_capacity",
            "energy",
            "static_potential",
        ]
        for attr in attrs:
            setattr(self, attr, self._add_single(gp_data, attr))

        self.restart = self.restart or gp_data.restart
        self.path_fc2 = self.path_fc2 or gp_data.path_fc2
        return self

    def exist_attr(self, attr: str = "free_energy"):
        """Check whether attribute exists in GridPointData."""
        if getattr(self, attr) is not None:
            return True
        return False

    @property
    def is_empty(self):
        """Return whether the grid point is empty or not."""
        return self.volume is None

    @property
    def unitcell(self):
        """Return unitcell."""
        if self.restart is None:
            return None
        return self.restart.unitcell

    @property
    def supercell_matrix(self):
        """Return supercell matrix."""
        if self.restart is None:
            return None
        return self.restart.supercell_matrix


class GridVT:
    """Dataclass for 2D array of GridPointData."""

    def __init__(
        self,
        volumes: np.ndarray,
        temperatures: np.ndarray,
        data: np.ndarray[GridPointData],
        verbose: bool = False,
    ):
        """Init method."""
        self._volumes = volumes
        self._temperatures = temperatures
        self._data = data
        self._verbose = verbose

    def __iter__(self):
        """Iter method."""
        for i, d1 in enumerate(self._data):
            for j, d2 in enumerate(d1):
                yield (i, j, d2)

    def __getitem__(self, key: tuple):
        """Get data item."""
        i, j = key
        return self._data[i][j]

    def __setitem__(self, key: tuple, value: float):
        """Set data item."""
        i, j = key
        self._data[i][j] = value

    @property
    def volumes(self):
        """Return volumes."""
        return self._volumes

    @property
    def temperatures(self):
        """Return temperatures."""
        return self._temperatures

    @property
    def data(self):
        """Return 2D data."""
        return self._data

    @property
    def shape(self):
        """Return shape of 2D data."""
        return self._data.shape

    def copy_static_data(self, grid):
        """Copy static properties and structure."""
        if grid.shape != self.shape:
            raise RuntimeError("Shapes mismatch.")

        for i, j, d in self:
            if d.is_empty:
                continue
            d.restart = grid[i, j].restart
            d.static_potential = d.restart.static_potential
        return self

    def get_properties(self, attr: str = "free_energy"):
        """Return property data."""
        arr = np.full(self._data.shape, None, dtype=object)
        for i, j, d in self:
            if d.exist_attr(attr):
                arr[i, j] = getattr(d, attr)
        return arr

    def get_volumes_properties(self, attr: str = "free_energy"):
        """Return volume-property data."""
        volumes = [
            [d2.volume for d2 in d1 if d2.exist_attr(attr)] for d1 in self._data.T
        ]
        properties = [
            [getattr(d2, attr) for d2 in d1 if d2.exist_attr(attr)]
            for d1 in self._data.T
        ]
        return zip(self._temperatures, volumes, properties)

    def get_temperatures_properties(self, attr: str = "free_energy"):
        """Return temperature-property data."""
        temperatures = [
            [d2.temperature for d2 in d1 if d2.exist_attr(attr)] for d1 in self._data
        ]
        properties = [
            [getattr(d2, attr) for d2 in d1 if d2.exist_attr(attr)] for d1 in self._data
        ]
        indices = [
            [i for i, d2 in enumerate(d1) if d2.exist_attr(attr)] for d1 in self._data
        ]
        return zip(self._volumes, temperatures, properties, indices)

    def fit_free_energy_volume(self):
        """Fit volume-free energy data to Vinet EOS."""
        fv_fits = []
        for _, volumes, properties in self.get_volumes_properties("free_energy"):
            try:
                eos = EOS(volumes, properties)
            except:
                eos = None
            fv_fits.append(eos)
        return fv_fits

    def fit_entropy_volume(self, max_order: int = 6):
        """Fit volume-entropy data using polynomial."""
        sv_fits = []
        for temp, volumes, properties in self.get_volumes_properties("entropy"):
            polyfit = Polyfit(volumes, properties)
            polyfit.fit(max_order=max_order, intercept=True, add_sqrt=False)
            sv_fits.append(polyfit)
            if self._verbose:
                print("- temperature:", temp, flush=True)
                header = "  model_rmse: "
                print(header, polyfit.best_model, f"{polyfit.error: .3e}", flush=True)
        return sv_fits

    def fit_cv_volume(self, max_order: int = 4):
        """Fit volume-Cv data using polynomial."""
        cv_fits = []
        for temp, volumes, properties in self.get_volumes_properties("heat_capacity"):
            polyfit = Polyfit(volumes, properties)
            polyfit.fit(max_order=max_order, intercept=True, add_sqrt=False)
            cv_fits.append(polyfit)
            if self._verbose:
                print("- temperature:", temp, flush=True)
                header = "  model_rmse: "
                print(header, polyfit.best_model, f"{polyfit.error: .4f}", flush=True)
                self.print_predictions(polyfit, volumes, properties, use_exp=False)
        return cv_fits

    def fit_entropy_temperature(self, max_order: int = 4):
        """Fit temperature-entropy data using polynomial.

        Deprecated.
        """
        st_fits = []
        data = self.get_temperatures_properties("entropy")
        for ivol, (vol, temps, properties, indices) in enumerate(data):
            polyfit = Polyfit(temps, properties)
            if np.isclose(temps[0], 0.0):
                polyfit.fit(
                    max_order=max_order,
                    intercept=False,
                    add_sqrt=True,
                    weight_begin=False,
                    weight_end=True,
                )
            else:
                polyfit.fit(
                    max_order=max_order,
                    intercept=True,
                    add_sqrt=True,
                    weight_begin=True,
                    weight_end=True,
                )
            st_fits.append(polyfit)
            if self._verbose:
                print("- volume:", vol, flush=True)
                header = "  model_rmse: "
                print(header, polyfit.best_model, f"{polyfit.error: .3e}", flush=True)
                self.print_predictions(polyfit, temps, properties)

            cv = temps * polyfit.eval_derivative(temps) * EVtoJmol
            for itemp, val in zip(indices, cv):
                self._data[ivol, itemp].heat_capacity = val

        return st_fits

    def print_predictions(
        self,
        polyfit: Polyfit,
        x: np.ndarray,
        y: np.ndarray,
        decimals_x: int = 3,
        use_exp: bool = True,
    ):
        """Print prediction and observation values."""
        print("  predictions:", flush=True)
        pred = polyfit.eval(x)
        for x1, f1, f2 in zip(x, y, pred):
            x1print = np.round(x1, decimals_x)
            if use_exp:
                print("  -", x1print, f"{f1: .3e}", f"{f2: .3e}", flush=True)
            else:
                print("  -", x1print, f"{f1: .3f}", f"{f2: .3f}", flush=True)
        return self


def sum_grids(grid_list: list):
    """Calculate sum of grid data."""
    if len(grid_list) == 1:
        return grid_list[0]

    for grid in grid_list[1:]:
        if not np.allclose(grid.volumes, grid_list[0].volumes):
            raise RuntimeError("Volumes not consistent.")
        if not np.allclose(grid.temperatures, grid_list[0].temperatures):
            raise RuntimeError("Temperatures not consistent.")

    grid_sum = copy.deepcopy(grid_list[0])
    for grid in grid_list[1:]:
        for i, j, g in grid:
            grid_sum[i, j].add(g)

    return grid_sum
