"""Functions for calculating thermodynamic properties."""

from typing import Literal, Optional

import numpy as np
import yaml
from phono3py.file_IO import read_fc2_from_hdf5
from phonopy import Phonopy

from pypolymlp.calculator.md.md_utils import load_thermodynamic_integration_yaml
from pypolymlp.calculator.sscha.sscha_utils import Restart
from pypolymlp.calculator.thermodynamics.fit_utils import Polyfit
from pypolymlp.calculator.thermodynamics.io_utils import save_thermodynamics_yaml
from pypolymlp.calculator.thermodynamics.thermodynamics_utils import (
    FittedModels,
    GridPointData,
    get_common_grid,
)
from pypolymlp.calculator.utils.eos_utils import EOS
from pypolymlp.core.units import EVtoJmol
from pypolymlp.utils.phonopy_utils import structure_to_phonopy_cell


class Thermodynamics:
    """Class for calculating properties from datasets on  grid points."""

    def __init__(
        self,
        data: list[GridPointData],
        data_type: Optional[Literal["sscha", "ti", "electron"]] = None,
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
        self._volumes = None
        self._temperatures = None
        self._grid = None
        self._models = None

        self._scan_data()

        self._eq_entropies = None
        self._eq_cp = None

    def _scan_data(self):
        """Scan and reconstruct data."""
        self._volumes = np.unique([d.volume for d in self._data])
        self._temperatures = np.unique([d.temperature for d in self._data])
        self._grid = np.full(
            (len(self._volumes), len(self._temperatures)),
            None,
            dtype=GridPointData,
        )
        for d in self._data:
            ivol = np.where(d.volume == self._volumes)[0][0]
            itemp = np.where(d.temperature == self._temperatures)[0][0]
            self._grid[ivol, itemp] = d

        if self._verbose:
            print("Dataset type:", self._data_type, flush=True)

        self._eliminate_temperatures(threshold=5)
        self._models = FittedModels(self._volumes, self._temperatures)
        return self

    def _eliminate_temperatures(self, threshold: int = 5):
        """Eliminate data for temperatures where only a small number of data exist."""
        ids = []
        for itemp, data in enumerate(self._grid.T):
            n_data = len(
                [d for d in data if d is not None and d.free_energy is not None]
            )
            if n_data >= threshold:
                ids.append(itemp)
                if self._verbose:
                    print("- temperature:", self._temperatures[itemp], flush=True)
                    print("  n_data:     ", n_data, flush=True)

        ids = np.array(ids)
        self._temperatures = self._temperatures[ids]
        self._grid = self._grid[:, ids]
        return self

    def fit_eos(self):
        """Fit volume-free energy data to Vinet EOS."""
        if self._verbose:
            print("Volume-FreeEnergy fitting.", flush=True)

        eos_fits = []
        for itemp, data in enumerate(self._grid.T):
            volumes = [
                d.volume for d in data if d is not None and d.free_energy is not None
            ]
            free_energies = [
                d.free_energy
                for d in data
                if d is not None and d.free_energy is not None
            ]
            try:
                eos = EOS(volumes, free_energies)
            except:
                eos = None
            eos_fits.append(eos)
        self._models.eos_fits = eos_fits
        return self

    def fit_entropy_volume(self, max_order: int = 6):
        """Fit volume-entropy data using polynomial."""
        if self._verbose:
            print("Volume-Entropy fitting.", flush=True)
        self._models.sv_fits = self._fit_wrt_volume(max_order=max_order, attr="entropy")
        return self

    def fit_cv_volume(self, max_order: int = 4):
        """Fit volume-Cv data using polynomial."""
        if self._verbose:
            print("Volume-Cv fitting.", flush=True)
        attr = "heat_capacity"
        self._models.cv_fits = self._fit_wrt_volume(max_order=max_order, attr=attr)
        return self

    def _fit_wrt_volume(self, max_order: int = 6, attr="entropy"):
        """Fit volume-property data using polynomial."""
        fits = []
        for itemp, data in enumerate(self._grid.T):
            volumes = [
                d.volume for d in data if d is not None and getattr(d, attr) is not None
            ]
            props = np.array(
                [
                    getattr(d, attr)
                    for d in data
                    if d is not None and getattr(d, attr) is not None
                ]
            )
            # temporarily revised.
            props[np.abs(props) == np.inf] = 0.0
            ###
            if len(props) > 4:
                polyfit = Polyfit(volumes, props)
                polyfit.fit(max_order=max_order, intercept=True, add_sqrt=False)
                fits.append(polyfit)
                if self._verbose:
                    print("- temperature:", self._temperatures[itemp], flush=True)
                    print("  rmse:       ", polyfit.error, flush=True)
                    print("  model:      ", polyfit.best_model, flush=True)
            else:
                fits.append(None)
        return fits

    def fit_free_energy_temperature(self, max_order: int = 6):
        """Fit temperature-free-energy data using polynomial."""
        if self._verbose:
            print("Temperature-FreeEnergy fitting.", flush=True)

        ft_fits = []
        for ivol, data in enumerate(self._grid):
            points = np.array(
                [d for d in data if d is not None and d.free_energy is not None]
            )

            if len(points) > 4:
                temperatures = np.array([p.temperature for p in points])
                free_energies = np.array([p.free_energy for p in points])
                polyfit = Polyfit(temperatures, free_energies)
                polyfit.fit(
                    max_order=max_order,
                    intercept=False,
                    first_order=False,
                    add_sqrt=False,
                )
                ft_fits.append(polyfit)
                if self._verbose:
                    print("- Volume:", np.round(self._volumes[ivol], 3), flush=True)
                    print("  RMSE:  ", polyfit.error, flush=True)
                    print("  model: ", polyfit.best_model, flush=True)

                # entropy calculations
                entropies = -polyfit.eval_derivative(temperatures)
                for p, val in zip(points, entropies):
                    p.entropy = val
            else:
                ft_fits.append(None)
        self._models.ft_fits = ft_fits
        return self

    def fit_entropy_temperature(self, max_order: int = 4, reference: bool = True):
        """Fit temperature-entropy data using polynomial."""
        if self._verbose:
            print("Temperature-Entropy fitting.", flush=True)

        st_fits = []
        for ivol, data in enumerate(self._grid):
            points = np.array(
                [d for d in data if d is not None and d.entropy is not None]
            )
            if len(points) > 4:
                if reference:
                    points = calculate_reference(points)
                temperatures = np.array([p.temperature for p in points])
                entropies = np.array([p.entropy for p in points])
                if reference:
                    ref = np.array([p.reference_entropy for p in points])
                    del_entropies = entropies - ref
                    polyfit = Polyfit(temperatures, del_entropies)
                else:
                    polyfit = Polyfit(temperatures, entropies)

                polyfit.fit(max_order=max_order, intercept=False, add_sqrt=True)
                st_fits.append(polyfit)
                if self._verbose:
                    print("- Volume:", np.round(self._volumes[ivol], 3), flush=True)
                    print("  RMSE:  ", polyfit.error, flush=True)
                    print("  model: ", polyfit.best_model, flush=True)

                # Cv calculations
                cv_from_ref = temperatures * polyfit.eval_derivative(temperatures)
                if reference:
                    for p, val in zip(points, cv_from_ref):
                        p.heat_capacity = p.reference_heat_capacity + val * EVtoJmol
                else:
                    for p, val in zip(points, cv_from_ref):
                        p.heat_capacity = val * EVtoJmol
            else:
                st_fits.append(None)
        self._models.st_fits = st_fits
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

    def add_cp(self, cp: np.array):
        """Add  Cp ."""
        if self._eq_cp is None:
            raise RuntimeError("Cp at V_eq not found.")

        self._eq_cp += np.array(cp)
        return self._eq_cp

    def fit_eval_entropy(self, max_order: int = 6, from_free_energy: bool = False):
        """Evaluate entropy from data."""
        if from_free_energy:
            self.fit_free_energy_temperature(max_order=max_order)
        self.fit_entropy_volume(max_order=max_order)
        self.eval_entropy_equilibrium()
        return self

    def fit_eval_cp(
        self,
        max_order: int = 4,
        from_entropy: bool = True,
        reference: bool = True,
    ):
        """Evaluate Cp from entropy data."""
        if from_entropy:
            self.fit_entropy_temperature(max_order=max_order, reference=reference)
        self.fit_cv_volume(max_order=max_order)
        self.eval_cp_equilibrium()
        return self

    def fit_eval_sscha(self):
        """Calculate thermodynamic properties from SSCHA."""
        self.fit_eos()
        self.fit_eval_entropy(max_order=6)
        self.fit_eval_cp(max_order=4, from_entropy=True)
        return self

    def save_thermodynamics_yaml(self, filename="polymlp_thermodynamics.yaml"):
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

    def add(self, thermo_add: np.ndarray):
        """Add grid data to the current grid data."""
        if self._grid.shape != thermo_add.grid.shape:
            (ix1_v, ix1_t), (ix2_v, ix2_t) = get_common_grid(
                self._volumes,
                thermo_add.volumes,
                self._temperatures,
                thermo_add.temperatures,
            )
            self.reshape(ix1_v, ix1_t)
            grid_add = thermo_add.grid[np.ix_(ix2_v, ix2_t)]
        else:
            grid_add = thermo_add.grid

        mask = np.equal(self._grid, None) | np.equal(grid_add, None)
        self._grid[mask] = None
        for g1, g2 in zip(self._grid[~mask], grid_add[~mask]):
            g1 = g1.add(g2)

        self._models = FittedModels(self._volumes, self._temperatures)
        return self

    def replace_free_energies(self, free_energies: np.ndarray, reset_fit: bool = True):
        """Replace free energies."""
        if reset_fit:
            self._models.eos_fits = None
            self._models.ft_fits = None
        self._replace(free_energies, attr="free_energy")
        return self

    def replace_entropies(self, entropies: np.ndarray, reset_fit: bool = True):
        """Replace entropies."""
        if reset_fit:
            self._models.sv_fits = None
            self._models.st_fits = None
        self._replace(entropies, attr="entropy")
        return self

    def replace_heat_capacities(
        self, heat_capacities: np.ndarray, reset_fit: bool = True
    ):
        """Replace heat capacities."""
        if reset_fit:
            self._models.cv_fits = None
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

    def reshape(self, ix_v: np.ndarray, ix_t: np.ndarray):
        """Reshape using ixgrid."""
        self._volumes = self._volumes[ix_v]
        self._temperatures = self._temperatures[ix_t]
        self._grid = self._grid[np.ix_(ix_v, ix_t)]
        self._models.reshape(ix_v, ix_t)
        return self


def calculate_reference(
    grid_points: list[GridPointData],
    mesh: np.ndarray = (10, 10, 10),
):
    """Return reference entropies.

    Harmonic phonon entropies calculated with SSCHA FC2 and frequencies
    is used as reference entropies to fit entropies with respect to temperature.
    """
    ref_id = 0
    if grid_points[ref_id].path_fc2 is None:
        raise RuntimeError("Reference state not found.")

    temperatures = np.array([p.temperature for p in grid_points])
    res = grid_points[ref_id].restart
    n_atom = len(res.unitcell.elements)
    ph = Phonopy(structure_to_phonopy_cell(res.unitcell), res.supercell_matrix)
    ph.force_constants = read_fc2_from_hdf5(grid_points[ref_id].path_fc2)
    ph.run_mesh(mesh)
    ph.run_thermal_properties(temperatures=temperatures)
    tp_dict = ph.get_thermal_properties_dict()

    for s, cv, point in zip(tp_dict["entropy"], tp_dict["heat_capacity"], grid_points):
        point.reference_entropy = s / EVtoJmol / n_atom
        point.reference_heat_capacity = cv / n_atom

    return grid_points


def adjust_to_common_grid(thermo1: Thermodynamics, thermo2: Thermodynamics):
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


def load_sscha_yamls(filenames: tuple[str], verbose: bool = False) -> Thermodynamics:
    """Load sscha_results.yaml files."""
    data = []
    for yamlfile in filenames:
        res = Restart(yamlfile, unit="eV/atom")
        n_atom = len(res.unitcell.elements)
        volume = np.round(res.volume, decimals=12) / n_atom
        temp = np.round(res.temperature, decimals=3)
        grid = GridPointData(
            volume=volume,
            temperature=temp,
            data_type="sscha",
            restart=res,
            path_yaml=yamlfile,
            path_fc2="/".join(yamlfile.split("/")[:-1]) + "/fc2.hdf5",
        )
        if res.converge and not res.imaginary:
            grid.free_energy = res.free_energy + res.static_potential
            grid.entropy = res.entropy
            grid.harmonic_heat_capacity = res.harmonic_heat_capacity
        else:
            grid.free_energy = None
            grid.entropy = None
            grid.harmonic_heat_capacity = None
        data.append(grid)

    return Thermodynamics(data=data, data_type="sscha", verbose=verbose)


def _check_melting(log: np.ndarray):
    """Check whether MD simulation converges to a melting state."""
    try:
        displacement_ratio = log[-1, 2] / log[0, 2]
        return displacement_ratio > 2.0
    except:
        return False


def load_ti_yamls(filenames: tuple[str], verbose: bool = False) -> Thermodynamics:
    """Load polymlp_ti.yaml files."""
    data = []
    for yamlfile in filenames:
        temp, volume, free_e, cv, log = load_thermodynamic_integration_yaml(yamlfile)
        if _check_melting(log):
            if verbose:
                message = yamlfile + " was eliminated (found to be in a melting state)."
                print(message, flush=True)
        else:
            grid = GridPointData(
                volume=volume,
                temperature=temp,
                data_type="ti",
                free_energy=free_e,
                heat_capacity=cv,
                path_yaml=yamlfile,
            )
            data.append(grid)
    return Thermodynamics(data=data, data_type="ti", verbose=verbose)


def load_electron_yamls(filenames: tuple[str], verbose: bool = False) -> Thermodynamics:
    """Load electron.yaml files."""
    data = []
    for yamlfile in filenames:
        yml = yaml.safe_load(open(yamlfile))
        n_atom = len(yml["structure"]["elements"])
        volume = float(yml["structure"]["volume"]) / n_atom
        for prop in yml["properties"]:
            temp = float(prop["temperature"])
            free_e = float(prop["free_energy"]) / n_atom
            entropy = float(prop["entropy"]) / n_atom
            cv = float(prop["specific_heat"]) * EVtoJmol / n_atom
            grid = GridPointData(
                volume=volume,
                temperature=temp,
                data_type="electron",
                free_energy=free_e,
                entropy=entropy,
                heat_capacity=cv,
                path_yaml=yamlfile,
            )
            data.append(grid)
    return Thermodynamics(data=data, data_type="electron", verbose=verbose)


def load_yamls(
    yamls_sscha: list[str],
    yamls_electron: Optional[list[str]] = None,
    yamls_ti: Optional[list[str]] = None,
    verbose: bool = False,
):
    """Load yaml files needed for calculating thermodynamics."""
    sscha = load_sscha_yamls(yamls_sscha, verbose=verbose)
    if yamls_electron is not None:
        electron = load_electron_yamls(yamls_electron, verbose=verbose)
        sscha, electron = adjust_to_common_grid(sscha, electron)
    else:
        electron = None
    if yamls_ti is not None:
        ti = load_ti_yamls(yamls_ti, verbose=verbose)
        sscha, ti = adjust_to_common_grid(sscha, ti)
        if yamls_electron is not None:
            sscha, electron = adjust_to_common_grid(sscha, electron)
    else:
        ti = None
    return sscha, electron, ti
