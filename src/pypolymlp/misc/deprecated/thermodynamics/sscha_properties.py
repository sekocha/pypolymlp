"""Class for calculating thermodynamic properties from SSCHA results."""

import copy
import itertools
import os
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import scipy
from phono3py.file_IO import read_fc2_from_hdf5
from phonopy import Phonopy
from phonopy.qha.core import BulkModulus
from phonopy.units import EVAngstromToGPa

from pypolymlp.calculator.compute_phonon import PolymlpPhonon
from pypolymlp.calculator.sscha.sscha_utils import Restart
from pypolymlp.calculator.sscha.utils.lsq import loocv
from pypolymlp.core.units import Avogadro, EVtoJ
from pypolymlp.core.utils import rmse
from pypolymlp.utils.phonopy_utils import structure_to_phonopy_cell
from pypolymlp.utils.vasp_utils import write_poscar_file


@dataclass
class GridVolTemp:
    """Dataclass for properties on a volume-temperature grid point."""

    temperature: float
    volume: float
    restart: Optional[Restart] = None
    free_energy: Optional[float] = None
    entropy: Optional[float] = None
    reference_entropy: Optional[float] = None
    heat_capacity: Optional[float] = None
    harmonic_heat_capacity: Optional[float] = None
    reference_heat_capacity: Optional[float] = None
    path_yaml: Optional[float] = None
    path_fc2: Optional[float] = None


@dataclass
class GridTemp:
    """Dataclass for properties on a temperature grid point."""

    temperature: float
    eqm_volume: Optional[float] = None
    eqm_free_energy: Optional[float] = None
    eqm_entropy: Optional[float] = None
    eqm_entropy_vol_deriv: Optional[float] = None
    eqm_heat_capacity: Optional[float] = None
    eqm_cp: Optional[float] = None
    bulk_modulus: Optional[float] = None
    free_energies: Optional[np.ndarray] = None
    entropies: Optional[np.ndarray] = None
    eos_fit_data: Optional[np.ndarray] = None
    gibbs_free_energies: Optional[np.ndarray] = None
    volume_entropy_fit: Optional[np.ndarray] = None
    volume_cv_fit: Optional[np.ndarray] = None


class SSCHAProperties:
    """Class for calculating thermodynamic properties from SSCHA results."""

    def __init__(self, filenames: list[str], verbose: bool = False):
        """Init method."""
        self._verbose = verbose
        self._volumes = None
        self._volumes_all = None
        self._temperatures = None
        self._grid_vt = None
        self._grid_t = None

        self._find_grid(filenames)
        self._load_sscha_yamls(filenames)
        if self._verbose:
            self._print_grid()

    def _find_grid(self, filenames):
        """Find unique volumes and temperatures."""
        self._temperatures = []
        self._volumes = []
        self._volumes_all = []
        for yamlfile in filenames:
            res = Restart(yamlfile)
            if res.converge and not res.imaginary:
                self._temperatures.append(np.round(res.temperature, decimals=3))
                self._volumes.append(np.round(res.volume, decimals=12))
            self._volumes_all.append(np.round(res.volume, decimals=12))
        self._volumes = np.unique(self._volumes)
        self._volumes_all = np.unique(self._volumes_all)
        self._temperatures = np.unique(self._temperatures)

        shape = (self._volumes.shape[0], self._temperatures.shape[0])
        self._grid_vt = np.zeros(shape, dtype=GridVolTemp)
        self._grid_t = np.zeros(shape[1], dtype=GridTemp)
        for ivol, vol in enumerate(self._volumes):
            for itemp, temp in enumerate(self._temperatures):
                self._grid_vt[ivol, itemp] = GridVolTemp(temperature=temp, volume=vol)
        for itemp, temp in enumerate(self._temperatures):
            self._grid_t[itemp] = GridTemp(temperature=temp)
        return self

    def _load_sscha_yamls(self, filenames: list[str]):
        """Load sscha_results.yaml files."""
        for yamlfile in filenames:
            res = Restart(yamlfile)
            volume = np.round(res.volume, decimals=12)
            temp = np.round(res.temperature, decimals=3)
            try:
                ivol = np.where(volume == self._volumes)[0][0]
                itemp = np.where(temp == self._temperatures)[0][0]
            except:
                continue

            grid = self._grid_vt[ivol][itemp]
            grid.restart = res
            grid.path_yaml = yamlfile
            grid.path_fc2 = "/".join(yamlfile.split("/")[:-1]) + "/fc2.hdf5"
            if res.converge and not res.imaginary:
                res.unit = "eV/cell"
                grid.free_energy = res.free_energy + res.static_potential
                res.unit = "kJ/mol"
                grid.entropy = res.entropy
                grid.harmonic_heat_capacity = res.harmonic_heat_capacity

        for itemp, temp in enumerate(self._temperatures):
            volumes, free_energies, entropies = [], [], []
            for g in self._grid_vt[:, itemp]:
                if g.free_energy is not None:
                    volumes.append(g.volume)
                    free_energies.append(g.free_energy)
                    entropies.append(g.entropy)
            self._grid_t[itemp].free_energies = np.stack([volumes, free_energies]).T
            self._grid_t[itemp].entropies = np.stack([volumes, entropies]).T

        return self

    def _print_grid(self):
        """Print SSCHA status on grid points."""
        print("SSCHA status:", flush=True)
        volumes_nodata = list(set(self._volumes_all) - set(self._volumes))
        for itemp, temp in enumerate(self._temperatures):
            print("- temperature:", temp, flush=True)
            n_data = 0
            imag_vol = []
            for ivol, vol in enumerate(self._volumes):
                grid = self._grid_vt[ivol, itemp]
                if grid.restart is not None:
                    if grid.free_energy is not None:
                        n_data += 1
                    if grid.restart.imaginary:
                        imag_vol.append(vol)
            imag_vol = np.round(volumes_nodata + imag_vol, 3)
            print("  - n_data:", n_data, flush=True)
            print("  - volumes (imag. freq.):", imag_vol, flush=True)
        return self

    def load_electron_yamls(self):
        """Add electronic free energy contribution."""
        if self._free_energies is None:
            raise RuntimeError("Call load_sscha_yamls in advance.")
        return self

    def _eos_function(self, bm: BulkModulus, volumes: np.ndarray):
        """EOS function.

        Return
        ------
        Energies.
        """
        parameters = bm.get_parameters()
        energies = bm._eos(volumes, *parameters)
        return energies

    def _fit_eos(self):
        """Fit EOS curves."""
        for itemp, temp in enumerate(self._temperatures):
            grid = self._grid_t[itemp]
            volumes = grid.free_energies[:, 0]
            free_energies = grid.free_energies[:, 1]
            bm = BulkModulus(volumes=volumes, energies=free_energies, eos="vinet")
            grid.eqm_volume = bm.equilibrium_volume
            grid.eqm_free_energy = bm.energy
            grid.bulk_modulus = bm.bulk_modulus * EVAngstromToGPa

            grid_volumes = np.linspace(min(volumes) * 0.9, max(volumes) * 1.1, 200)
            fitted = self._eos_function(bm, grid_volumes)
            grid.eos_fit_data = np.stack([grid_volumes, fitted]).T
            grid.gibbs_free_energies = self._transformFVTtoGPT(bm, grid_volumes, fitted)

        return self

    def _transformFVTtoGPT(
        self,
        bm: BulkModulus,
        volumes: np.ndarray,
        free_energies: np.ndarray,
        eps: float = 1e-4,
    ):
        """Transform Helmholtz free energy to Gibbs free energy."""
        pressure_gibbs_free_energies = []
        for vol, fe in zip(volumes, free_energies):
            eos_f = self._eos_function(bm, vol + eps)
            eos_b = self._eos_function(bm, vol - eps)
            deriv = (eos_f - eos_b) / (2 * eps)
            press = -deriv  # in eV/ang^3
            gibbs = fe + press * vol
            press_gpa = press * EVAngstromToGPa  # in GPa
            pressure_gibbs_free_energies.append([press_gpa, gibbs])
        return pressure_gibbs_free_energies

    def _fit_volume_entropy(self):
        """Fit volume-entropy curves."""
        if self._verbose:
            print("RMSE (V-S fit):", flush=True)
        for itemp, temp in enumerate(self._temperatures):
            volumes, entropies, _ = self._get_data(itemp=itemp, attr="entropy")
            (coeffs, _, error), _ = self._polyfit(volumes, entropies, max_order=6)
            deriv = self._get_poly_deriv(coeffs)
            if self._verbose:
                print("- temperature:", temp, ", rmse", np.round(error, 5), flush=True)

            grid = self._grid_t[itemp]
            grid.eqm_entropy = np.polyval(coeffs, grid.eqm_volume)
            grid.volume_entropy_fit = coeffs
            grid.eqm_entropy_vol_deriv = np.polyval(deriv, grid.eqm_volume)

        return self

    def _fit_heat_capacity(self):
        """Fit properties for calculating heat capacity."""
        self._set_reference_entropies()
        self._fit_temperature_entropy()
        self._fit_volume_heat_capacity()
        self._compute_cp()
        return self

    def _fit_temperature_entropy_spline(self):
        """Fit temperature-entropy at volumes."""
        if self._verbose:
            print("RMSE (T-S fit):", flush=True)
        for ivol, vol in enumerate(self._volumes):
            temperatures, entropies, itemps = self._get_data(ivol=ivol, attr="entropy")
            _, ref_entropies, _ = self._get_data(ivol=ivol, attr="reference_entropy")
            del_entropies = entropies - ref_entropies

            sp1 = scipy.interpolate.make_interp_spline(temperatures, del_entropies, k=2)
            pred = sp1(temperatures)
            error = rmse(del_entropies, pred)
            if self._verbose:
                error = np.round(error, 5)
                print("- volume:", np.round(vol, 3), ", rmse:", error, flush=True)

            cv_from_ref = temperatures * sp1.derivative()(temperatures)
            for itemp, val in zip(itemps, cv_from_ref):
                g = self._grid_vt[ivol, itemp]
                g.heat_capacity = g.reference_heat_capacity + val

        return self

    def _fit_temperature_entropy(self):
        """Fit temperature-entropy at volumes."""
        if self._verbose:
            print("RMSE (T-S fit):", flush=True)
        for ivol, vol in enumerate(self._volumes):
            temperatures, entropies, itemps = self._get_data(ivol=ivol, attr="entropy")
            _, ref_entropies, _ = self._get_data(ivol=ivol, attr="reference_entropy")
            del_entropies = entropies - ref_entropies

            (coeffs, pred, error), (order, add_sqrt) = self._polyfit(
                temperatures,
                del_entropies,
                add_sqrt=None,
                intercept=False,
                max_order=4,
            )
            if self._verbose:
                error = np.round(error, 5)
                print("- volume:", np.round(vol, 3), ", rmse:", error, flush=True)

            if add_sqrt:
                deriv_poly = self._get_poly_deriv(coeffs[1:])
                cv_from_ref = temperatures * np.polyval(deriv_poly, temperatures)
                cv_from_ref += 0.5 * coeffs[0] * np.power(temperatures, 0.5)
            else:
                deriv = self._get_poly_deriv(coeffs)
                cv_from_ref = temperatures * np.polyval(deriv, temperatures)

            for itemp, val in zip(itemps, cv_from_ref):
                g = self._grid_vt[ivol, itemp]
                g.heat_capacity = g.reference_heat_capacity + val

        return self

    def _fit_volume_heat_capacity(self):
        """Fit volume-Cv at temperatures."""
        if self._verbose:
            print("RMSE (V-Cv fit):", flush=True)
        for itemp, temp in enumerate(self._temperatures):
            volumes, cvs, _ = self._get_data(itemp=itemp, attr="heat_capacity")
            (coeffs, _, error), _ = self._polyfit(volumes, cvs, max_order=6)
            if self._verbose:
                print("- temperature:", temp, ", rmse", np.round(error, 5), flush=True)

            grid = self._grid_t[itemp]
            grid.eqm_heat_capacity = np.polyval(coeffs, grid.eqm_volume)
            grid.volume_cv_fit = coeffs
        return self

    def _compute_cp(self):
        """Calculate Cp - Cv."""
        for itemp, temp in enumerate(self._temperatures):
            g = self._grid_t[itemp]
            bm = g.bulk_modulus / EVAngstromToGPa
            add = temp * g.eqm_volume * (g.eqm_entropy_vol_deriv**2) / bm
            add /= EVtoJ * Avogadro
            g.eqm_cp = g.eqm_heat_capacity + add
        return self

    def _get_data(
        self,
        ivol: Optional[int] = None,
        itemp: Optional[int] = None,
        attr: Optional[str] = None,
    ):
        """Slice grid data."""
        if ivol is not None:
            grid = self._grid_vt[ivol]
            x = [g.temperature for g in grid if getattr(g, attr) is not None]
        elif itemp is not None:
            grid = self._grid_vt[:, itemp]
            x = [g.volume for g in grid if getattr(g, attr) is not None]
        else:
            raise RuntimeError("Set ivol or itemp.")

        y = [getattr(g, attr) for g in grid if getattr(g, attr) is not None]
        ids = [i for i, g in enumerate(grid) if getattr(g, attr) is not None]
        return np.array(x), np.array(y), np.array(ids)

    def _polyfit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        order: Optional[int] = None,
        max_order: int = 4,
        intercept: bool = True,
        add_sqrt: bool = False,
    ):
        """Fit data to a polynomial.

        If order is None, the optimal value of order will be automatically
        determined by minimizing the leave-one-out cross validation score.
        """
        orders = list(range(2, max_order + 1)) if order is None else [order]
        sqrts = [True, False] if add_sqrt is None else [add_sqrt]
        if len(orders) == 1 and len(sqrts) == 1:
            best_order = orders[0]
            best_add_sqrt = sqrts[0]
        else:
            min_loocv = 1e10
            best_order, best_add_sqrt = None, None
            params = list(itertools.product(sqrts, orders))
            for add_sqrt, order in params:
                (poly_coeffs, y_pred, y_rmse), X = self._polyfit_single(
                    x,
                    y,
                    order,
                    intercept=intercept,
                    add_sqrt=add_sqrt,
                )
                cv = loocv(X, y, y_pred)
                if min_loocv > cv:
                    min_loocv = cv
                    best_order = order
                    best_add_sqrt = add_sqrt

        (poly_coeffs, y_pred, y_rmse), _ = self._polyfit_single(
            x,
            y,
            best_order,
            intercept=intercept,
            add_sqrt=best_add_sqrt,
        )
        if not intercept:
            poly_coeffs = list(poly_coeffs)
            poly_coeffs.append(0.0)
            poly_coeffs = np.array(poly_coeffs)

        return (poly_coeffs, y_pred, y_rmse), (best_order, best_add_sqrt)

    def _polyfit_single(
        self,
        x: np.ndarray,
        y: np.ndarray,
        order: int,
        intercept: bool = True,
        add_sqrt: bool = False,
    ):
        """Fit data to a polynomial with a given order."""
        X = []
        if add_sqrt:
            X.append(np.sqrt(x))
        for power in np.arange(order, 0, -1, dtype=int):
            X.append(x**power)
        if intercept:
            X.append(np.ones(x.shape))
        X = np.array(X).T

        poly_coeffs = np.linalg.solve(X.T @ X, X.T @ y)
        y_pred = X @ poly_coeffs
        y_rmse = rmse(y, y_pred)
        return (poly_coeffs, y_pred, y_rmse), X

    def run(self, reference: Literal["auto", "harmonic"] = "auto"):
        """Fit all properties."""
        self._fit_eos()
        self._fit_volume_entropy()
        self._fit_heat_capacity()
        return self

    def _set_reference_entropies(self, reference: Literal["auto", "harmonic"] = "auto"):
        """Set reference entropies."""
        if reference == "auto":
            self._set_reference_entropies_mintemp()
        elif reference == "harmonic":
            self._set_reference_entropies_harmonic()
        else:
            raise RuntimeError("auto or harmonic is available as reference entropy.")
        return self

    def _set_reference_entropies_mintemp(self, mesh: np.ndarray = (10, 10, 10)):
        """Use reference entropies automatically.

        Harmonic phonon model with SSCHA frequencies is used to
        calculate reference entropies.
        """
        t_min, t_max, t_step = self._check_temperatures()
        print("Reference entropy:", flush=True)
        for ivol, vol in enumerate(self._volumes):
            res = None
            itemp = -1
            while res is None:
                itemp += 1
                grid = self._grid_vt[ivol, itemp]
                if grid.restart is not None and not grid.restart.imaginary:
                    res = grid.restart

            if self._verbose:
                temp = self._temperatures[itemp]
                print("V =", np.round(vol, 3), "T =", temp, flush=True)

            ph = Phonopy(structure_to_phonopy_cell(res.unitcell), res.supercell_matrix)
            ph.force_constants = read_fc2_from_hdf5(grid.path_fc2)
            ph.run_mesh(mesh)
            ph.run_thermal_properties(t_step=t_step, t_max=t_max, t_min=t_min)
            tp_dict = ph.get_thermal_properties_dict()

            for itemp, val in enumerate(tp_dict["entropy"]):
                grid = self._grid_vt[ivol, itemp]
                if grid.entropy is not None:
                    grid.reference_entropy = val
            for itemp, val in enumerate(tp_dict["heat_capacity"]):
                grid = self._grid_vt[ivol, itemp]
                if grid.entropy is not None:
                    grid.reference_heat_capacity = val
        return self

    def _set_reference_entropies_harmonic(
        self,
        distance: float = 0.01,
        mesh: np.ndarray = (10, 10, 10),
    ):
        """Use harmonic entropies as reference entropies."""
        t_min, t_max, t_step = self._check_temperatures()
        print("Reference entropy (harmonic):", flush=True)
        for ivol, vol in enumerate(self._volumes):
            if self._verbose:
                print("- vol:", vol, flush=True)
            res = None
            itemp = -1
            while res is None:
                itemp += 1
                res = self._grid_vt[ivol, itemp].restart

            if os.path.exists(res.parameters["pot"]):
                pot = res.parameters["pot"]
            else:
                pot = "/".join(res.parameters["pot"].split("/")[-2:])

            ph = PolymlpPhonon(
                unitcell=res.unitcell,
                supercell_matrix=res.supercell_matrix,
                pot=pot,
            )
            ph.produce_force_constants(distance=distance)
            ph.compute_properties(mesh=mesh, t_min=t_min, t_max=t_max, t_step=t_step)
            tp_dict = ph.phonopy.get_thermal_properties_dict()

            for itemp, val in enumerate(tp_dict["entropy"]):
                grid = self._grid_vt[ivol, itemp]
                if grid.entropy is not None:
                    grid.reference_entropy = val
            for itemp, val in enumerate(tp_dict["heat_capacity"]):
                grid = self._grid_vt[ivol, itemp]
                if grid.entropy is not None:
                    grid.reference_heat_capacity = val
        return self

    def _check_temperatures(self):
        """Convert temperatures to temperature min., max., and interval."""
        t_min = np.min(self._temperatures)
        t_max = np.max(self._temperatures)
        diff = np.array(self._temperatures)[1:] - np.array(self._temperatures[:-1])
        if np.allclose(diff, diff[0]):
            t_step = diff[0]
            return t_min, t_max, t_step
        raise RuntimeError("Constant temperature step not found.")

    def save_properties(self, filename: str = "sscha_properties.yaml"):
        """Save properties to a file."""
        f = open(filename, "w")
        print("units:", file=f)
        print("  temperature:   K", file=f)
        print("  volume:        angstroms^3/unitcell", file=f)
        print("  pressure:      GPa", file=f)
        print("  free_energy:   eV/unitcell", file=f)
        print("  entropy:       J/K/mol", file=f)
        print("  bulk_modulus:  GPa", file=f)
        print("  heat_capacity: J/K/mol", file=f)
        print("", file=f)

        print("equilibrium_properties:", file=f)
        for grid in self._grid_t:
            print("- temperature:      ", grid.temperature, file=f)
            print("  volume:           ", grid.eqm_volume, file=f)
            print("  bulk_modulus:     ", grid.bulk_modulus, file=f)
            print("  free_energy:      ", grid.eqm_free_energy, file=f)
            print("  entropy:          ", grid.eqm_entropy, file=f)
            print("  heat_capacity_cp: ", grid.eqm_cp, file=f)
            print("", file=f)

        self._save_2d_array(f, tag="free_energies")
        self._save_2d_array(f, tag="eos_fit_data")
        self._save_2d_array(f, tag="entropies")
        self._save_2d_array(f, tag="gibbs_free_energies")

        print("cv:", file=f)
        for ivol, vol in enumerate(self._volumes):
            print("- volume:", vol, file=f)
            print("  values:", file=f)
            for itemp, temp in enumerate(self._temperatures):
                grid = self._grid_vt[ivol, itemp]
                print("  -", [temp, grid.heat_capacity], file=f)
            print("", file=f)

        f.close()
        return self

    def save_equilibrium_structures(self, path: str = "sscha_eqm_poscars"):
        """Save structures with equilibrium volumes to files."""
        os.makedirs(path, exist_ok=True)
        unitcell = self._grid_vt[int(len(self._volumes) / 2), 0].restart.unitcell
        for itemp, grid in enumerate(self._grid_t):
            expand = grid.eqm_volume / np.linalg.det(unitcell.axis)
            expand = np.power(expand, 1.0 / 3.0)
            unitcell_expand = copy.deepcopy(unitcell)
            unitcell_expand.axis *= expand
            filename = path + "/POSCAR-" + str(self._temperatures[itemp])
            write_poscar_file(unitcell_expand, filename=filename)
        return self

    def _save_2d_array(self, f, tag: str):
        """Save dict of 2D array"""
        print(tag + ":", file=f)
        for itemp, temp in enumerate(self._temperatures):
            array1d = getattr(self._grid_t[itemp], tag)
            print("- temperature:", temp, file=f)
            print("  values:", file=f)
            for vals in array1d:
                print("  -", list(vals), file=f)
            print("", file=f)
        return self

    def _get_poly_deriv(self, coeffs: np.ndarray):
        """Return derivatives of coefficients from polynomial fits."""
        deriv = coeffs * np.arange(len(coeffs) - 1, -1, -1, dtype=int)
        return deriv[:-1]

    @property
    def grid_data_temperature(self):
        """Return properties at temperatures."""
        return self._grid_t

    @property
    def grid_data_volume_temperature(self):
        """Return properties at grid points (V, T)."""
        return self._grid_vt
