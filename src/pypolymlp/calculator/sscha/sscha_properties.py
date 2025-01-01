"""Class for calculating thermodynamic properties from SSCHA results."""

import os
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from phonopy.qha.core import BulkModulus
from phonopy.units import EVAngstromToGPa

from pypolymlp.calculator.compute_phonon import PolymlpPhonon
from pypolymlp.calculator.sscha.sscha_utils import Restart
from pypolymlp.core.utils import rmse


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
        self._temperatures = None
        self._grid_vt = None
        self._grid_t = None
        self._entropy_polyfit = None

        self._find_grid(filenames)
        self._load_sscha_yamls(filenames)

    def _load_sscha_yamls(self, filenames: list[str]):
        """Load sscha_results.yaml files."""
        for yamlfile in filenames:
            res = Restart(yamlfile)
            volume = np.round(res.volume, decimals=12)
            temp = np.round(res.temperature, decimals=3)
            ivol = np.where(volume == self._volumes)[0][0]
            itemp = np.where(temp == self._temperatures)[0][0]
            grid = self._grid_vt[ivol][itemp]
            grid.restart = res
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

    def _find_grid(self, filenames):
        """Find unique volumes and temperatures."""
        self._temperatures = []
        self._volumes = []
        for yamlfile in filenames:
            res = Restart(yamlfile)
            self._temperatures.append(np.round(res.temperature, decimals=3))
            self._volumes.append(np.round(res.volume, decimals=12))
        self._volumes = np.unique(self._volumes)
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

    def load_electron_yamls(self):
        """Add electronic free energy contribution."""
        if self._free_energies is None:
            raise RuntimeError("Call load_sscha_yamls in advance.")
        return self

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

    def _eos_function(self, bm: BulkModulus, volumes: np.ndarray):
        """EOS function.

        Return
        ------
        Energies.
        """
        parameters = bm.get_parameters()
        energies = bm._eos(volumes, *parameters)
        return energies

    def _fit_volume_entropy(self, order: int = 4):
        """Fit volume-entropy curves."""
        for itemp, temp in enumerate(self._temperatures):
            volumes, entropies, _ = self._get_data(itemp=itemp, attr="entropy")
            coeffs, _, error = self._polyfit(volumes, entropies, order)
            deriv = self._get_poly_deriv(coeffs)
            if self._verbose:
                print("RMSE (V-S Fit, T = " + str(temp) + "):", error)

            grid = self._grid_t[itemp]
            grid.eqm_entropy = np.polyval(coeffs, grid.eqm_volume)
            grid.volume_entropy_fit = coeffs
            grid.eqm_entropy_vol_deriv = np.polyval(deriv, grid.eqm_volume)

        return self

    def _fit_heat_capacity(self):
        """Fit properties for calculating heat capacity."""
        self._fit_temperature_entropy(order=4)
        self._fit_volume_heat_capacity(order=4)
        self._compute_cp()
        return self

    def _fit_temperature_entropy(self, order: int = 4):
        """Fit temperature-entropy at volumes."""
        for ivol, vol in enumerate(self._volumes):
            temperatures, entropies, itemps = self._get_data(ivol=ivol, attr="entropy")
            _, ref_entropies, _ = self._get_data(ivol=ivol, attr="reference_entropy")
            del_entropies = entropies - ref_entropies
            # TODO: Use fit without intercept.
            coeffs, _, error = self._polyfit(temperatures, del_entropies, order)
            deriv = self._get_poly_deriv(coeffs)
            if self._verbose:
                print("RMSE (T-S Fit, V = " + str(vol) + "):", error)

            heat_capacity_from_ref = temperatures * np.polyval(deriv, temperatures)
            for itemp, val in zip(itemps, heat_capacity_from_ref):
                g = self._grid_vt[ivol, itemp]
                g.heat_capacity = g.reference_heat_capacity + val

        return self

    def _fit_volume_heat_capacity(self, order: int = 4):
        """Fit volume-Cv at temperatures."""
        for itemp, temp in enumerate(self._temperatures):
            volumes, cvs, _ = self._get_data(itemp=itemp, attr="heat_capacity")
            coeffs, _, error = self._polyfit(volumes, cvs, order)
            if self._verbose:
                print("RMSE (V-Cv Fit, T = " + str(temp) + "):", error)

            grid = self._grid_t[itemp]
            grid.eqm_heat_capacity = np.polyval(coeffs, grid.eqm_volume)
            grid.volume_cv_fit = coeffs
        return self

    def _compute_cp(self):
        """Calculate Cp - Cv."""
        for itemp, temp in enumerate(self._temperatures):
            g = self._grid_t[itemp]
            bm = g.bulk_modulus / EVAngstromToGPa
            # TODO: consider units or eqm_entropy_vol_deriv (too large)
            add = temp * g.eqm_volume * (g.eqm_entropy_vol_deriv**2) / bm
            g.eqm_cp = g.eqm_heat_capacity + add
            print(g.eqm_volume, g.eqm_entropy_vol_deriv, bm)
            print(temp, g.eqm_cp, add)
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

    def _polyfit(self, x: np.ndarray, y: np.ndarray, order: int = 4):
        """Fit data to a polynomial using numpy."""
        poly_coeffs = np.polyfit(x, y, order)
        y_pred = np.polyval(poly_coeffs, x)
        y_rmse = rmse(y, y_pred)
        return (poly_coeffs, y_pred, y_rmse)

    def run(
        self,
        entropy_order: int = 4,
        reference: Literal["harmonic", "auto"] = "harmonic",
    ):
        """Fit all properties."""
        self._fit_eos()
        self._fit_volume_entropy(order=entropy_order)

        if reference == "harmonic":
            self.compute_harmonic_entropies()
        else:
            pass
        self._fit_heat_capacity()
        return self

    def compute_harmonic_entropies(
        self,
        distance: float = 0.001,
        mesh: np.ndarray = (10, 10, 10),
    ):
        """Compute harmonic entropies."""
        t_min, t_max, t_step = self._check_temperatures()
        for ivol, vol in enumerate(self._volumes):
            if self._verbose:
                print("Harmonic phonon: vol =", vol)
            res = None
            itemp = 0
            while res is None:
                res = self._grid_vt[ivol, itemp].restart
                itemp += 1
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
                self._grid_vt[ivol, itemp].reference_heat_capacity = val

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
        print("", file=f)

        print("equilibrium_properties:", file=f)
        for grid in self._grid_t:
            print("- temperature:  ", grid.temperature, file=f)
            print("  volume:       ", grid.eqm_volume, file=f)
            print("  free_energy:  ", grid.eqm_free_energy, file=f)
            print("  entropy:      ", grid.eqm_entropy, file=f)
            print("  bulk_modulus: ", grid.bulk_modulus, file=f)
            print("", file=f)

        self._save_dict_of_2d_array(f, tag="free_energies")
        self._save_dict_of_2d_array(f, tag="eos_fit_data")
        self._save_dict_of_2d_array(f, tag="entropies")
        self._save_dict_of_2d_array(f, tag="gibbs_free_energies")
        f.close()

    def _save_dict_of_2d_array(self, f, tag: str):
        """Save dict of 2D array"""
        print(tag + ":", file=f)
        for itemp, temp in enumerate(self._temperatures):
            array1d = getattr(self._grid_t[itemp], tag)
            print(" - temperature:", temp, file=f)
            for vals in array1d:
                print("   -", list(vals), file=f)
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


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml",
        nargs="*",
        type=str,
        default=None,
        help="sscha_results.yaml files",
    )
    args = parser.parse_args()

    np.set_printoptions(legacy="1.21")
    sscha = SSCHAProperties(args.yaml, verbose=True)
    sscha.run(entropy_order=4)
    sscha.save_properties(filename="sscha_properties.yaml")
