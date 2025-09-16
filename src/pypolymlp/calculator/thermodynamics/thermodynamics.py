"""Functions for calculating thermodynamic properties."""

import copy
from typing import Literal, Optional

import numpy as np

from pypolymlp.calculator.thermodynamics.fit_utils import Polyfit
from pypolymlp.calculator.thermodynamics.initialization import (
    calculate_harmonic_free_energies,
    calculate_reference,
    get_common_grid,
    load_electron_yamls,
    load_sscha_yamls,
    load_ti_yamls,
)
from pypolymlp.calculator.thermodynamics.io_utils import save_thermodynamics_yaml
from pypolymlp.calculator.thermodynamics.thermodynamics_utils import (
    FittedModels,
    GridPointData,
    sum_matrix_data,
)
from pypolymlp.calculator.utils.eos_utils import EOS
from pypolymlp.core.units import EVtoJmol


def _exist_attr(d: GridPointData, attr: str = "free_energy"):
    """Check whether attribute exists in GridPointData."""
    if d is not None and getattr(d, attr) is not None:
        return True
    return False


class Thermodynamics:
    """Class for calculating properties from datasets on grid points."""

    def __init__(
        self,
        data: list[GridPointData],
        data_type: Optional[Literal["sscha", "ti", "electron", "electron_ph"]] = None,
        verbose: bool = False,
    ):
        """Init method.

        Units of input data
        -------------------
        free_energy: eV/atom
        entropy: eV/K/atom
        heat_capacity: J/K/mol (/Avogadro's number of atoms)
        """
        self._data = data
        self._data_type = data_type
        self._verbose = verbose

        if self._verbose:
            print("Dataset type:", self._data_type, flush=True)

        self._eq_entropies = None
        self._eq_cp = None

        self._is_heat_capacity = self._check_heat_capacity()
        self._scan_data()

    def _check_heat_capacity(self):
        """Check whether heat capacity has been calculated."""
        for d in self._data:
            if d.heat_capacity is not None:
                return True
        return False

    def _scan_data(self):
        """Scan and reconstruct data."""
        self._volumes = np.unique([d.volume for d in self._data])
        self._temperatures = np.unique([d.temperature for d in self._data])
        self._models = FittedModels(self._volumes, self._temperatures)

        shape = (len(self._volumes), len(self._temperatures))
        self._grid = np.full(shape, None, dtype=GridPointData)
        for d in self._data:
            ivol = np.where(d.volume == self._volumes)[0][0]
            itemp = np.where(d.temperature == self._temperatures)[0][0]
            self._grid[ivol, itemp] = d

        self._eliminate_temperatures(threshold=5)
        return self

    def _eliminate_temperatures(self, threshold: int = 5):
        """Eliminate data for temperatures where only a small number of data exist."""
        if self._verbose:
            print("temperature_and_n_data:", flush=True)
        ids = []
        id_output = 1
        for itemp, data in enumerate(self._grid.T):
            n_data = len([d for d in data if _exist_attr(d, "free_energy")])
            if n_data >= threshold:
                ids.append(itemp)
                if self._verbose:
                    output_data = (self._temperatures[itemp], n_data)
                    print(output_data, end=" ", flush=True)
                    if id_output % 5 == 0:
                        print(flush=True)
                    id_output += 1
        if self._verbose:
            print(flush=True)

        ids = np.array(ids)
        self._temperatures = self._temperatures[ids]
        self._grid = self._grid[:, ids]
        return self

    def calculate_reference(self):
        """Calculate reference properties."""
        if self._verbose:
            print("Calculate reference properties.", flush=True)

        for g in self._grid:
            g = calculate_reference(g, self._temperatures)
        return self

    def copy_reference(self, grid: np.ndarray):
        """Copy reference properties."""
        for g1, g2 in zip(grid, self._grid):
            for p, q in zip(g1, g2):
                if q is not None:
                    q.copy_reference(p)
        return self

    def calculate_harmonic_free_energies(self):
        """Calculate harmonic free energies."""
        if self._verbose:
            print("Calculate harmonic free energies.", flush=True)

        for g, vol in zip(self._grid, self._volumes):
            if self._verbose:
                print("- Volume:", np.round(vol, 3), flush=True)
            g = calculate_harmonic_free_energies(g)
        return self

    def fit_free_energy_volume(self):
        """Fit volume-free energy data to Vinet EOS."""
        if self._verbose:
            print("Volume-FreeEnergy fitting.", flush=True)

        fv_fits = []
        for itemp, data in enumerate(self._grid.T):
            volumes = [d.volume for d in data if _exist_attr(d, "free_energy")]
            energies = [d.free_energy for d in data if _exist_attr(d, "free_energy")]
            try:
                eos = EOS(volumes, energies)
            except:
                eos = None
            fv_fits.append(eos)
        self._models.fv_fits = fv_fits
        return self

    def fit_entropy_volume(self, max_order: int = 6, assign_fit_values: bool = False):
        """Fit volume-entropy data using polynomial."""
        if self._verbose:
            print("Volume-Entropy fitting.", flush=True)
        self._models.sv_fits = self._fit_wrt_volume(
            max_order=max_order,
            attr="entropy",
            assign_fit_values=assign_fit_values,
        )
        return self

    def fit_cv_volume(self, max_order: int = 4):
        """Fit volume-Cv data using polynomial."""
        if not self._is_heat_capacity:
            return self
        if self._verbose:
            print("Volume-Cv fitting.", flush=True)
        self._models.cv_fits = self._fit_wrt_volume(
            max_order=max_order,
            attr="heat_capacity",
            check_distribution=True,
        )
        return self

    def _fit_wrt_volume(
        self,
        max_order: int = 6,
        attr: str = "entropy",
        assign_fit_values: bool = False,
        check_distribution: bool = False,
    ):
        """Fit volume-property data using polynomial."""
        fits = []
        for itemp, data in enumerate(self._grid.T):
            props = np.array([getattr(d, attr) for d in data if _exist_attr(d, attr)])
            if len(props) > max_order + 1:
                volumes = np.array([d.volume for d in data if _exist_attr(d, attr)])
                if check_distribution and not np.allclose(props, 0.0):
                    volumes, props = self._check_distribution(volumes, props)

                if volumes is None:
                    fits.append(None)
                else:
                    polyfit = Polyfit(volumes, props)
                    polyfit.fit(max_order=max_order, intercept=True, add_sqrt=False)
                    fits.append(polyfit)
                    if self._verbose:
                        print("- temperature:", self._temperatures[itemp], flush=True)
                        print(
                            "  model_rmse: ",
                            polyfit.best_model,
                            polyfit.error,
                            flush=True,
                        )

                    self._print_predictions(volumes, props, polyfit)
                    if assign_fit_values:
                        pred = polyfit.eval(volumes)
                        idx = 0
                        for d in data:
                            if _exist_attr(d, attr):
                                setattr(d, attr, pred[idx])
                                idx += 1
            else:
                fits.append(None)
        return fits

    def _check_distribution(self, volumes: np.ndarray, props: np.ndarray):
        """Check property distribution before fitting."""
        idx = np.where(props > 0.0)[0]
        volumes, props = volumes[idx], props[idx]

        cbegin = int(len(volumes) * 0.45)
        cend = 2 * cbegin

        ave, std = np.mean(props[cbegin:cend]), np.std(props[cbegin:cend])
        if std > ave * 0.2:
            return None, None

        cond = (props < ave + 1.5 * std) & (props > ave - 1.5 * std)
        target = cond[cbegin:cend]

        if np.count_nonzero(target) > len(target) / 5:
            cond = (props < ave + 2.0 * std) & (props > ave - 2.0 * std)
            target = cond[cbegin:cend]
            if np.count_nonzero(target) > len(target) / 5:
                cond = (props < ave + 2.5 * std) & (props > ave - 2.5 * std)
                if self._verbose:
                    print("Volume-Cv data is largely scattering.")
                    print(" Average, Std:", ave, std, flush=True)
                    print(" Cv:", flush=True)
                    print(props, flush=True)
                    print(" Selected Cv:", flush=True)
                    print(props[cond], flush=True)

        volumes, props = volumes[cond], props[cond]
        return volumes, props

    def fit_free_energy_temperature(self, max_order: int = 6, intercept: bool = False):
        """Fit temperature-free-energy data using polynomial."""
        if self._verbose:
            print("Temperature-FreeEnergy fitting.", flush=True)

        ft_fits = []
        for ivol, data in enumerate(self._grid):
            points = np.array([d for d in data if _exist_attr(d, "free_energy")])
            if len(points) > max_order + 1:
                temperatures = np.array([p.temperature for p in points])
                free_energies = np.array([p.free_energy for p in points])
                ref = np.array([p.reference_free_energy for p in points])

                polyfit = Polyfit(temperatures, free_energies - ref)
                polyfit.fit(
                    max_order=max_order,
                    intercept=intercept,
                    first_order=False,
                    add_sqrt=False,
                )
                ft_fits.append(polyfit)
                if self._verbose:
                    print("- volume:", np.round(self._volumes[ivol], 3), flush=True)
                    print(
                        "  model_rmse:  ", polyfit.best_model, polyfit.error, flush=True
                    )

                # self._print_predictions(temperatures, free_energies - ref, polyfit)
                # entropy calculations
                entropies = -polyfit.eval_derivative(temperatures)
                for p, val in zip(points, entropies):
                    p.entropy = p.reference_entropy + val
            else:
                ft_fits.append(None)
        self._models.ft_fits = ft_fits
        return self

    def fit_entropy_temperature(self, max_order: int = 4):
        """Fit temperature-entropy data using polynomial."""
        if self._verbose:
            print("Temperature-Entropy fitting.", flush=True)

        st_fits = []
        for ivol, data in enumerate(self._grid):
            points = np.array([d for d in data if _exist_attr(d, "entropy")])
            if len(points) > max_order + 1:
                temperatures = np.array([p.temperature for p in points])
                entropies = np.array([p.entropy for p in points])
                ref = np.array([p.reference_entropy for p in points])
                polyfit = Polyfit(temperatures, entropies - ref)
                if np.isclose(temperatures[0], 0.0):
                    polyfit.fit(max_order=max_order, intercept=False, add_sqrt=True)
                else:
                    polyfit.fit(max_order=max_order, intercept=True, add_sqrt=True)

                st_fits.append(polyfit)
                if self._verbose:
                    print("- volume:", np.round(self._volumes[ivol], 3), flush=True)
                    print(
                        "  model_rmse:  ", polyfit.best_model, polyfit.error, flush=True
                    )

                # self._print_predictions(temperatures, entropies - ref, polyfit)
                # Cv calculations
                cv_from_ref = temperatures * polyfit.eval_derivative(temperatures)
                for p, val in zip(points, cv_from_ref):
                    p.heat_capacity = p.reference_heat_capacity + val * EVtoJmol
            else:
                st_fits.append(None)
        self._models.st_fits = st_fits
        self._is_heat_capacity = True
        return self

    def _print_predictions(self, x: np.ndarray, y: np.ndarray, polyfit: Polyfit):
        """Print prediction values."""
        pred = polyfit.eval(x)
        for x1, f1, f2 in zip(x, y, pred):
            print(" ", np.round(x1, 3), f1, f2)

    def eval_entropy_equilibrium(self):
        """Evaluate entropies at equilibrium volumes."""
        self._eq_entropies = np.array(
            [self._models.eval_eq_entropy(i) for i, _ in enumerate(self._temperatures)]
        )
        return self._eq_entropies

    def eval_cp_equilibrium(self):
        """Evaluate Cp from S and Cv functions."""
        if not self._is_heat_capacity:
            return None
        self._eq_cp = np.array(
            [self._models.eval_eq_cp(i) for i, _ in enumerate(self._temperatures)]
        )
        return self._eq_cp

    def get_data(self, attr: str = "free_energy"):
        """Retrun data of given attribute."""
        props = []
        for data1 in self._grid:
            array = []
            for d in data1:
                p = None if d is None else getattr(d, attr)
                array.append(p)
            props.append(array)

        return np.array(props)

    def reshape(self, ix_v: np.ndarray, ix_t: np.ndarray):
        """Reshape using ixgrid."""
        self._volumes = self._volumes[ix_v]
        self._temperatures = self._temperatures[ix_t]
        self._grid = self._grid[np.ix_(ix_v, ix_t)]
        self._models.reshape(ix_v, ix_t)
        return self

    def replace_free_energies(self, free_energies: np.ndarray, reset_fit: bool = True):
        """Replace free energies."""
        if reset_fit:
            self._models.fv_fits = None
            self._models.ft_fits = None
        self._replace(free_energies, attr="free_energy")
        return self

    def replace_entropies(self, entropies: np.ndarray, reset_fit: bool = True):
        """Replace entropies."""
        if reset_fit:
            self._models.sv_fits = None
            self._models.st_fits = None
            self._eq_entropies = None
        self._replace(entropies, attr="entropy")
        return self

    def replace_heat_capacities(
        self, heat_capacities: np.ndarray, reset_fit: bool = True
    ):
        """Replace heat capacities."""
        if reset_fit:
            self._models.cv_fits = None
            self._eq_cp = None
        self._replace(heat_capacities, attr="heat_capacity")
        return self

    def _replace(self, properties: np.ndarray, attr: str = "free_energy"):
        """Replace properties."""
        if properties.shape != self._grid.shape:
            raise RuntimeError("Different grid points in two objects.")

        for i, g1 in enumerate(self._grid):
            for j, g2 in enumerate(g1):
                if g2 is None:
                    self._grid[i, j] = GridPointData(
                        volume=self._volumes[i],
                        temperature=self._temperatures[j],
                        data_type=self._data_type,
                    )
                setattr(self._grid[i, j], attr, properties[i, j])
        return self

    def save_thermodynamics_yaml(self, filename: str = "polymlp_thermodynamics.yaml"):
        """Save fitted thermodynamics properties."""
        save_thermodynamics_yaml(
            self._volumes,
            self._temperatures,
            self._models,
            self._eq_entropies,
            self._eq_cp,
            self.get_data(attr="free_energy"),
            filename=filename,
        )

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

    @property
    def grid(self):
        """Return grid points.

        Rows and columns correspond to volumes and temperatures, respectively.
        """
        return self._grid

    @grid.setter
    def grid(self, _grid: np.ndarray):
        """Set grid points."""
        self._grid = _grid

    @property
    def volumes(self):
        """Return volumes."""
        return self._volumes

    @volumes.setter
    def volumes(self, _volumes: np.ndarray):
        """Set volumes."""
        self._volumes = _volumes

    @property
    def temperatures(self):
        """Return temperatures."""
        return self._temperatures

    @temperatures.setter
    def temperatures(self, _temperatures: np.ndarray):
        """Set temperatures."""
        self._temperatures = _temperatures

    @property
    def fitted_models(self):
        """Return fitted models."""
        return self._models

    @property
    def is_heat_capacity(self):
        """Return whether heat capacity exists or not."""
        return self._is_heat_capacity

    def save_data(self, filename: str = "polymlp_thermodynamics_grid.yaml"):
        """Save grid data to file."""
        with open(filename, "w") as f:
            print("grid_data:", file=f)
            for grid, temp in zip(self._grid.T, self._temperatures):
                print("- temperature:", temp, file=f)
                for g2 in grid:
                    if g2 is not None and g2.entropy is not None:
                        print("  - volume:       ", g2.volume, file=f)
                        print("    free_energy:  ", g2.free_energy, file=f)
                        print("    entropy:      ", g2.entropy * EVtoJmol, file=f)
                        print("    heat_capacity:", g2.heat_capacity, file=f)
                        print(file=f)


def _adjust_to_common_grid(thermo1: Thermodynamics, thermo2: Thermodynamics):
    """Reshape objects with common grid."""
    (ix1_v, ix1_t), (ix2_v, ix2_t) = get_common_grid(
        thermo1.volumes,
        thermo2.volumes,
        thermo1.temperatures,
        thermo2.temperatures,
    )
    thermo1.reshape(ix1_v, ix1_t)
    thermo2.reshape(ix2_v, ix2_t)
    return thermo1, thermo2


def load_yamls(
    yamls_sscha: list[str],
    yamls_electron: Optional[list[str]] = None,
    yamls_ti: Optional[list[str]] = None,
    yamls_electron_phonon: Optional[list[str]] = None,
    verbose: bool = False,
):
    """Load yaml files needed for calculating thermodynamics."""
    if verbose:
        print("Loading sscha.yaml files.", flush=True)
    data = load_sscha_yamls(yamls_sscha)
    sscha = Thermodynamics(data=data, data_type="sscha", verbose=verbose)

    if yamls_electron is not None:
        if verbose:
            print("Loading electron.yaml files.", flush=True)
        data2 = load_electron_yamls(yamls_electron)
        electron = Thermodynamics(data=data2, data_type="electron", verbose=verbose)
        sscha, electron = _adjust_to_common_grid(sscha, electron)
    else:
        electron = None

    if yamls_ti is not None:
        if verbose:
            print("Loading ti.yaml files.", flush=True)
        data3 = load_ti_yamls(yamls_ti, verbose=verbose)
        ti = Thermodynamics(data=data3, data_type="ti", verbose=verbose)
        sscha, ti = _adjust_to_common_grid(sscha, ti)
        if yamls_electron is not None:
            sscha, electron = _adjust_to_common_grid(sscha, electron)
    else:
        ti = None
        ti_ref = None

    # Set reference
    sscha.calculate_reference()
    if electron is not None:
        electron.copy_reference(sscha.grid)
    if ti is not None:
        ti.copy_reference(sscha.grid)

    # Set reference term for TI (multiple reference states)
    # if ti is not None:
    #     ti_ref = copy.deepcopy(sscha)
    #     f1 = sscha.get_data(attr="harmonic_free_energy")
    #     ti_ref.replace_free_energies(f1)
    #     ti_ref.fit_free_energy_temperature(max_order=4, intercept=True)

    #     f1 = sscha.get_data(attr="static_potential")
    #     f2 = ti_ref.get_data(attr="free_energy")
    #     f_sum = sum_matrix_data(f1, f2)
    #     ti_ref.replace_free_energies(f_sum)

    # Set reference term for TI (single reference state)
    if ti is not None:
        ti_ref = copy.deepcopy(sscha)
        f1 = sscha.get_data(attr="reference_free_energy")
        s1 = sscha.get_data(attr="reference_entropy")
        f2 = sscha.get_data(attr="static_potential")
        f_sum = sum_matrix_data(f1, f2)
        ti_ref.replace_free_energies(f_sum)
        ti_ref.replace_entropies(s1)

    if yamls_electron_phonon is not None:
        if verbose:
            print("Loading electron.yaml (sscha) files.", flush=True)
        data4 = load_electron_yamls(yamls_electron_phonon, data_type="electron_ph")
        electron_ph = Thermodynamics(
            data=data4, data_type="electron_ph", verbose=verbose
        )
        sscha, electron_ph = _adjust_to_common_grid(sscha, electron_ph)
        if yamls_electron is not None:
            electron, electron_ph = _adjust_to_common_grid(electron, electron_ph)
        if yamls_ti is not None:
            ti, electron_ph = _adjust_to_common_grid(ti, electron_ph)
            ti_ref, electron_ph = _adjust_to_common_grid(ti_ref, electron_ph)
    else:
        electron_ph = None

    return sscha, electron, ti, ti_ref, electron_ph
