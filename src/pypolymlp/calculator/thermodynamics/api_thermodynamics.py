"""API Class for calculating properties from datasets on grid points."""

import numpy as np

from pypolymlp.calculator.thermodynamics.thermodynamics_io import (
    save_thermodynamics_yaml,
)
from pypolymlp.calculator.thermodynamics.thermodynamics_parser import GridVT
from pypolymlp.calculator.thermodynamics.thermodynamics_utils import FittedModels

# from pypolymlp.calculator.thermodynamics.init import (
#     calculate_harmonic_free_energies,
#     calculate_reference,
# )


class Thermodynamics:
    """API Class for calculating properties from datasets on grid points."""

    def __init__(self, grid: GridVT, verbose: bool = False):
        """Init method.

        Input data units
        ----------------
        free_energy: eV/atom
        entropy: eV/K/atom
        heat_capacity: J/K/mol (/Avogadro's number of atoms)
        """
        self._grid = grid
        self._grid._verbose = verbose
        self._verbose = verbose

        self._volumes = self._grid.volumes
        self._temperatures = self._grid.temperatures
        self._models = FittedModels(self._volumes, self._temperatures)

        self._eq_free_energies = None
        self._eq_entropies = None
        self._eq_cp = None

        # self._is_heat_capacity = self._check_heat_capacity()
        # self._scan_data()

    def fit_free_energy_volume(self):
        """Fit volume-free energy data to Vinet EOS."""
        if self._verbose:
            print("Volume-FreeEnergy fitting.", flush=True)

        self._models.fv_fits = self._grid.fit_free_energy_volume()
        return self

    def fit_entropy_volume(self, max_order: int = 6):
        """Fit volume-entropy data using polynomial."""
        if self._verbose:
            print("Volume-Entropy fitting.", flush=True)

        self._models.sv_fits = self._grid.fit_entropy_volume(max_order=max_order)
        return self

    def fit_entropy_temperature(self, max_order: int = 4):
        """Fit temperature-entropy data using polynomial."""
        if self._verbose:
            print("Temperature-Entropy fitting.", flush=True)

        self._models.st_fits = self._grid.fit_entropy_temperature(max_order=max_order)
        return self

    def fit_cv_volume(self, max_order: int = 4):
        """Fit volume-Cv data using polynomial."""
        if self._verbose:
            print("Volume-Cv fitting.", flush=True)

        self._models.cv_fits = self._grid.fit_cv_volume(max_order=max_order)
        return self

    def eval_entropy_equilibrium(self):
        """Evaluate entropies at equilibrium volumes."""
        self._eq_entropies = np.array(
            [self._models.eval_eq_entropy(i) for i, _ in enumerate(self._temperatures)]
        )
        return self._eq_entropies

    def eval_cp_equilibrium(self):
        """Evaluate Cp from S and Cv functions."""
        self._eq_cp = np.array(
            [self._models.eval_eq_cp(i) for i, _ in enumerate(self._temperatures)]
        )
        return self._eq_cp

    def save_thermodynamics_yaml(self, filename: str = "polymlp_thermodynamics.yaml"):
        """Save fitted thermodynamics properties."""
        save_thermodynamics_yaml(
            self._volumes,
            self._temperatures,
            self._models,
            self._eq_entropies,
            self._eq_cp,
            self._grid.get_properties("free_energy"),
            filename=filename,
        )
        return self

    def eval_free_energies(self, volumes: np.ndarray):
        """Return free energy.

        Return
        ------
        Helmholtz free energies.

        Rows and columns correspond to volumes and temperatures, respectively.
        """
        return self._models.eval_helmholtz_free_energies(volumes)

    def eval_entropies(self, volumes: np.ndarray):
        """Return entropies.

        Return
        ------
        Entropies.

        Rows and columns correspond to volumes and temperatures, respectively.
        """
        return self._models.eval_entropies(volumes)

    def eval_heat_capacities(self, volumes: np.ndarray):
        """Return heat capacities.

        Return
        ------
        Heat capacities.

        Rows and columns correspond to volumes and temperatures, respectively.
        """
        if not self._is_heat_capacity:
            return None
        return self._models.eval_heat_capacities(volumes)

    def eval_gibbs_free_energies(self, volumes: np.ndarray):
        """Return Gibbs free energy.

        Return
        ------
        Gibbs free energies.
            Array of (temperature index, pressure in GPa, Gibbs free energy).
        """
        return self._models.eval_gibbs_free_energies(volumes)


#     def calculate_reference(self):
#         """Calculate reference properties."""
#         if self._verbose:
#             print("Calculate reference properties.", flush=True)
#
#         for g in self._grid:
#             g = calculate_reference(g, self._temperatures)
#         return self
#
#     def copy_reference(self, grid: np.ndarray):
#         """Copy reference properties."""
#         for g1, g2 in zip(grid, self._grid):
#             for p, q in zip(g1, g2):
#                 if q is not None:
#                     q.copy_reference(p)
#         return self
#
#     def calculate_harmonic_free_energies(self):
#         """Calculate harmonic free energies."""
#         if self._verbose:
#             print("Calculate harmonic free energies.", flush=True)
#
#         for g, vol in zip(self._grid, self._volumes):
#             if self._verbose:
#                 print("- Volume:", np.round(vol, 3), flush=True)
#             g = calculate_harmonic_free_energies(g)
#         return self
#

#     def _check_distribution(self, volumes: np.ndarray, props: np.ndarray):
#         """Check property distribution before fitting."""
#         idx = np.where(props > 0.0)[0]
#         volumes, props = volumes[idx], props[idx]
#
#         cbegin = int(len(volumes) * 0.45)
#         cend = 2 * cbegin
#
#         ave, std = np.mean(props[cbegin:cend]), np.std(props[cbegin:cend])
#         if std > ave * 0.5:
#             return None, None
#
#         cond = (props < ave + 2.0 * std) & (props > ave - 2.0 * std)
#         target = cond[cbegin:cend]
#
#         if np.count_nonzero(target) < len(target) * 0.8:
#             cond = (props < ave + 3.0 * std) & (props > ave - 3.0 * std)
#             target = cond[cbegin:cend]
#             if np.count_nonzero(target) < len(target) * 0.8:
#                 cond = (props < ave + 4.0 * std) & (props > ave - 4.0 * std)
#                 if self._verbose:
#                     print("Volume-Cv data is largely scattering.")
#                     print(" Average, Std:", ave, std, flush=True)
#                     print(" Cv:", flush=True)
#                     print(props, flush=True)
#                     print(" Selected Cv:", flush=True)
#                     print(props[cond], flush=True)
#
#         volumes, props = volumes[cond], props[cond]
#         return volumes, props
#
#     def fit_free_energy_temperature(self, max_order: int = 6, intercept: bool = False):
#         """Fit temperature-free-energy data using polynomial."""
#         if self._verbose:
#             print("Temperature-FreeEnergy fitting.", flush=True)
#
#         ft_fits = []
#         for ivol, data in enumerate(self._grid):
#             points = np.array([d for d in data if _exist_attr(d, "free_energy")])
#             if len(points) > max_order + 1:
#                 temperatures = np.array([p.temperature for p in points])
#                 free_energies = np.array([p.free_energy for p in points])
#                 ref = np.array([p.reference_free_energy for p in points])
#
#                 polyfit = Polyfit(temperatures, free_energies - ref)
#                 if np.isclose(temperatures[0], 0.0):
#                     polyfit.fit(
#                         max_order=max_order,
#                         intercept=intercept,
#                         add_sqrt=False,
#                         weight_begin=False,
#                         weight_end=True,
#                     )
#                 else:
#                     polyfit.fit(
#                         max_order=max_order,
#                         intercept=intercept,
#                         add_sqrt=False,
#                         weight_begin=True,
#                         weight_end=True,
#                     )
#                 ft_fits.append(polyfit)
#                 if self._verbose:
#                     print("- volume:", np.round(self._volumes[ivol], 3), flush=True)
#                     print(
#                         "  model_rmse:  ", polyfit.best_model, polyfit.error, flush=True
#                     )
#
#                 # self._print_predictions(temperatures, free_energies - ref, polyfit)
#                 # entropy calculations
#                 entropies = -polyfit.eval_derivative(temperatures)
#                 for p, val in zip(points, entropies):
#                     p.entropy = p.reference_entropy + val
#             else:
#                 ft_fits.append(None)
#         self._models.ft_fits = ft_fits
#         return self
#

#     def fit_energy_temperature(self, max_order: int = 4):
#         """Fit temperature-enenergy data using polynomial."""
#         if self._data_type != "ti":
#             raise RuntimeError("fit_energy_temperature is available for TI.")
#
#         if self._verbose:
#             print("Temperature-Energy fitting.", flush=True)
#
#         et_fits = []
#         for ivol, data in enumerate(self._grid):
#             points = np.array([d for d in data if _exist_attr(d, "energy")])
#             if len(points) > max_order + 1:
#                 temperatures = np.array([p.temperature for p in points])
#                 energies = np.array([p.energy for p in points])
#                 polyfit = Polyfit(temperatures, energies)
#                 if np.isclose(temperatures[0], 0.0):
#                     polyfit.fit(max_order=max_order, intercept=False, add_sqrt=False)
#                 else:
#                     polyfit.fit(max_order=max_order, intercept=True, add_sqrt=False)
#
#                 et_fits.append(polyfit)
#                 if self._verbose:
#                     print("- volume:", np.round(self._volumes[ivol], 3), flush=True)
#                     print(
#                         "  model_rmse:  ", polyfit.best_model, polyfit.error, flush=True
#                     )
#
#                 self._print_predictions(temperatures, energies, polyfit)
#                 cv = polyfit.eval_derivative(temperatures)
#                 for p, cv in zip(points, cv):
#                     p.heat_capacity = cv
#             else:
#                 et_fits.append(None)
#         self._models.et_fits = et_fits
#         self._is_heat_capacity = True
#         return self
#

#     def eval_cp_equilibrium(self):
#         """Evaluate Cp from S and Cv functions."""
#         if not self._is_heat_capacity:
#             return None
#         self._eq_cp = np.array(
#             [self._models.eval_eq_cp(i) for i, _ in enumerate(self._temperatures)]
#         )
#         return self._eq_cp
#
#     def get_data(self, attr: str = "free_energy"):
#         """Retrun data of given attribute."""
#         props = []
#         for data1 in self._grid:
#             array = []
#             for d in data1:
#                 p = None if d is None else getattr(d, attr)
#                 array.append(p)
#             props.append(array)
#
#         return np.array(props)
#
#     def reshape(self, ix_v: np.ndarray, ix_t: np.ndarray):
#         """Reshape using ixgrid."""
#         self._volumes = self._volumes[ix_v]
#         self._temperatures = self._temperatures[ix_t]
#         self._grid = self._grid[np.ix_(ix_v, ix_t)]
#         self._models.reshape(ix_v, ix_t)
#         return self
#
#     def replace_free_energies(self, free_energies: np.ndarray, reset_fit: bool = True):
#         """Replace free energies."""
#         if reset_fit:
#             self._models.fv_fits = None
#             self._models.ft_fits = None
#         self._replace(free_energies, attr="free_energy")
#         return self
#
#     def replace_entropies(self, entropies: np.ndarray, reset_fit: bool = True):
#         """Replace entropies."""
#         if reset_fit:
#             self._models.sv_fits = None
#             self._models.st_fits = None
#             self._eq_entropies = None
#         self._replace(entropies, attr="entropy")
#         return self
#
#     def replace_heat_capacities(
#         self, heat_capacities: np.ndarray, reset_fit: bool = True
#     ):
#         """Replace heat capacities."""
#         if reset_fit:
#             self._models.cv_fits = None
#             self._eq_cp = None
#         self._replace(heat_capacities, attr="heat_capacity")
#         return self
#
#     def _replace(self, properties: np.ndarray, attr: str = "free_energy"):
#         """Replace properties."""
#         if properties.shape != self._grid.shape:
#             raise RuntimeError("Different grid points in two objects.")
#
#         for i, g1 in enumerate(self._grid):
#             for j, g2 in enumerate(g1):
#                 if g2 is None:
#                     self._grid[i, j] = GridPointData(
#                         volume=self._volumes[i],
#                         temperature=self._temperatures[j],
#                         data_type=self._data_type,
#                     )
#                 setattr(self._grid[i, j], attr, properties[i, j])
#         return self
#
#     def eval_free_energies(self, volumes: np.ndarray):
#         """Return free energy.
#
#         Return
#         ------
#         Helmholtz free energies.
#
#         Rows and columns correspond to volumes and temperatures, respectively.
#         """
#         return self._models.eval_helmholtz_free_energies(volumes)
#
#     def eval_entropies(self, volumes: np.ndarray):
#         """Return entropies.
#
#         Return
#         ------
#         Entropies.
#
#         Rows and columns correspond to volumes and temperatures, respectively.
#         """
#         return self._models.eval_entropies(volumes)
#
#     def eval_heat_capacities(self, volumes: np.ndarray):
#         """Return heat capacities.
#
#         Return
#         ------
#         Heat capacities.
#
#         Rows and columns correspond to volumes and temperatures, respectively.
#         """
#         if not self._is_heat_capacity:
#             return None
#         return self._models.eval_heat_capacities(volumes)
#
#     def eval_gibbs_free_energies(self, volumes: np.ndarray):
#         """Return Gibbs free energy.
#
#         Return
#         ------
#         Gibbs free energies.
#             Array of (temperature index, pressure in GPa, Gibbs free energy).
#         """
#         return self._models.eval_gibbs_free_energies(volumes)
#
#     @property
#     def grid(self):
#         """Return grid points.
#
#         Rows and columns correspond to volumes and temperatures, respectively.
#         """
#         return self._grid
#
#     @grid.setter
#     def grid(self, _grid: np.ndarray):
#         """Set grid points."""
#         self._grid = _grid
#
#     @property
#     def volumes(self):
#         """Return volumes."""
#         return self._volumes
#
#     @volumes.setter
#     def volumes(self, _volumes: np.ndarray):
#         """Set volumes."""
#         self._volumes = _volumes
#
#     @property
#     def temperatures(self):
#         """Return temperatures."""
#         return self._temperatures
#
#     @temperatures.setter
#     def temperatures(self, _temperatures: np.ndarray):
#         """Set temperatures."""
#         self._temperatures = _temperatures
#
#     @property
#     def fitted_models(self):
#         """Return fitted models."""
#         return self._models
#
#     @property
#     def is_heat_capacity(self):
#         """Return whether heat capacity exists or not."""
#         return self._is_heat_capacity
#
#     def save_data(self, filename: str = "polymlp_thermodynamics_grid.yaml"):
#         """Save grid data to file."""
#         with open(filename, "w") as f:
#             print("grid_data:", file=f)
#             for grid, temp in zip(self._grid.T, self._temperatures):
#                 print("- temperature:", temp, file=f)
#                 for g2 in grid:
#                     if g2 is not None and g2.entropy is not None:
#                         print("  - volume:       ", g2.volume, file=f)
#                         print("    free_energy:  ", g2.free_energy, file=f)
#                         print("    entropy:      ", g2.entropy * EVtoJmol, file=f)
#                         print("    heat_capacity:", g2.heat_capacity, file=f)
#                         print(file=f)
#
