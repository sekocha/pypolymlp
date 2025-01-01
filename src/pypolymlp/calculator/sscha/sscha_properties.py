"""Class for calculating thermodynamic properties from SSCHA results."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from phonopy.qha.core import BulkModulus
from phonopy.units import EVAngstromToGPa

# from pypolymlp.calculator.compute_phonon import PolymlpPhonon
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
    harmonic_entropy: Optional[float] = None
    anharmonic_entropy: Optional[float] = None
    heat_capacity: Optional[float] = None
    harmonic_heat_capacity: Optional[float] = None
    anharmonic_heat_capacity: Optional[float] = None


@dataclass
class GridTemp:
    """Dataclass for properties on a temperature grid point."""

    temperature: float
    eqm_volume: Optional[float] = None
    eqm_free_energy: Optional[float] = None
    eqm_entropy: Optional[float] = None
    eqm_entropy_vol_deriv: Optional[float] = None
    bulk_modulus: Optional[float] = None
    free_energies: Optional[np.ndarray] = None
    entropies: Optional[np.ndarray] = None
    eos_fit_data: Optional[np.ndarray] = None
    gibbs_free_energies: Optional[np.ndarray] = None


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

        #        self.compute_harmonic_entropies()
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
            volumes = self._grid_t[itemp].free_energies[:, 0]
            free_energies = self._grid_t[itemp].free_energies[:, 1]
            bm = BulkModulus(volumes=volumes, energies=free_energies, eos="vinet")
            self._grid_t[itemp].eqm_volume = bm.equilibrium_volume
            self._grid_t[itemp].eqm_free_energy = bm.energy
            self._grid_t[itemp].bulk_modulus = bm.bulk_modulus * EVAngstromToGPa

            grid_volumes = np.linspace(min(volumes) * 0.9, max(volumes) * 1.1, 200)
            fitted = self._eos_function(bm, grid_volumes)
            self._grid_t[itemp].eos_fit_data = np.stack([grid_volumes, fitted]).T
            gibbs_free_energies = self._transform_FVT_to_GPT(bm, grid_volumes, fitted)
            self._grid_t[itemp].gibbs_free_energies = gibbs_free_energies

        return self

    def _transform_FVT_to_GPT(
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

    def _fit_entropy(self, order: int = 4):
        """Fit volume-entropy curves."""
        self._entropy_polyfit = dict()
        for itemp, temp in enumerate(self._temperatures):
            volumes = self._grid_t[itemp].entropies[:, 0]
            entropies = self._grid_t[itemp].entropies[:, 1]
            coeffs = np.polyfit(volumes, entropies, order)
            if self._verbose:
                entropies_pred = np.polyval(coeffs, volumes)
                rmse_entropy = rmse(entropies, entropies_pred)
                print("RMSE (EntropyFit, T = " + str(temp) + "):", rmse_entropy)

            eqm_entropy = np.polyval(coeffs, self._grid_t[itemp].eqm_volume)
            self._grid_t[itemp].eqm_entropy = eqm_entropy
            self._entropy_polyfit[itemp] = coeffs

        for itemp, deriv in self.entropy_polyfit_derivative.items():
            eqm_entropy_vol_deriv = np.polyval(deriv, self._grid_t[itemp].eqm_volume)
            self._grid_t[itemp].eqm_entropy_vol_deriv = eqm_entropy_vol_deriv

        return self

    def run(self, entropy_order: int = 4):
        """Fit all properties."""
        self._fit_eos()
        self._fit_entropy(order=entropy_order)
        return self

        #    def compute_harmonic_entropies(
        #        self,
        #        distance: float = 0.001,
        #        mesh: np.ndarray = (10, 10, 10),
        #    ):
        #        """Compute harmonic entropies."""
        #        self._harmonic_entropies = defaultdict(list)
        #        self._anharmonic_entropies = defaultdict(list)
        #        self._harmonic_heat_capacities = defaultdict(list)
        #
        #        t_min, t_max, t_step = self._check_temperatures()
        #        for ivol in range(len(self._volumes)):
        #            res = self._grid[ivol][0]
        #            if os.path.exists(res.parameters["pot"]):
        #                pot = res.parameters["pot"]
        #            else:
        #                pot = "/".join(res.parameters["pot"].split("/")[-2:])
        #
        #            ph = PolymlpPhonon(
        #                unitcell=res.unitcell,
        #                supercell_matrix=res.supercell_matrix,
        #                pot=pot,
        #            )
        #            ph.produce_force_constants(distance=distance)
        #            ph.compute_properties(mesh=mesh, t_min=t_min, t_max=t_max, t_step=t_step)
        #            tp_dict = ph.phonopy.get_thermal_properties_dict()
        #            print(tp_dict)
        #
        #            for temp, val in zip(tp_dict["temperatures"], tp_dict["entropy"]):
        #                self._harmonic_entropies[temp].append([res.volume, val])
        #            for temp, val in zip(tp_dict["temperatures"], tp_dict["heat_capacity"]):
        #                self._harmonic_heat_capacities[temp].append([res.volume, val])
        #
        #        for temp in self._temperatures:
        #            anh = np.array(self._entropies[temp])[:, 1] - np.array(self._harmonic_entropies[temp])[:, 1]
        #            self._anharmonic_entropies[temp] = np.stack([np.array(self._entropies[temp])[:, 0], anh]).T
        #            print(self._anharmonic_entropies[temp].shape)
        ##

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

    @property
    def grid_data_temperature(self):
        """Return properties at temperatures."""
        return self._grid_t

    @property
    def grid_data_volume_temperature(self):
        """Return properties at grid points (V, T)."""
        return self._grid_vt

    @property
    def entropy_polyfit(self):
        """Return coefficients from polynomial fits.

        The coefficients are obtained using np.polyfit.
        """
        return self._entropy_polyfit

    @property
    def entropy_polyfit_derivative(self):
        """Return volume derivatives of coefficients from polynomial fits."""
        derivatives = dict()
        for temp, coeffs in self._entropy_polyfit.items():
            deriv = coeffs * np.arange(len(coeffs) - 1, -1, -1, dtype=int)
            derivatives[temp] = deriv[:-1]
        return derivatives


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

#    print(sscha.bulk_moduli)
#    print(sscha.equilibrium_free_energies)
#    print(sscha.equilibrium_volumes)
#    print(sscha.equilibrium_entropies)
#
#    print(sscha.entropy_polyfit)
#    print(sscha.eos_fit_data)
