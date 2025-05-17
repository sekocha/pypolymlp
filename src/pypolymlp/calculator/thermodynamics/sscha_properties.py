"""Class for calculating thermodynamic properties from SSCHA results."""

import copy
import os
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import scipy
from phono3py.file_IO import read_fc2_from_hdf5
from phonopy import Phonopy
from phonopy.units import EVAngstromToGPa

from pypolymlp.calculator.compute_phonon import PolymlpPhonon
from pypolymlp.core.units import Avogadro, EVtoJ
from pypolymlp.core.utils import rmse
from pypolymlp.utils.phonopy_utils import structure_to_phonopy_cell
from pypolymlp.utils.vasp_utils import write_poscar_file

# @dataclass
# class GridVolTemp:
#     """Dataclass for properties on a volume-temperature grid point."""
#
#     temperature: float
#     volume: float
#     restart: Optional[Restart] = None
#     free_energy: Optional[float] = None
#     entropy: Optional[float] = None
#     reference_entropy: Optional[float] = None
#     heat_capacity: Optional[float] = None
#     harmonic_heat_capacity: Optional[float] = None
#     reference_heat_capacity: Optional[float] = None
#     path_yaml: Optional[float] = None
#     path_fc2: Optional[float] = None


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

    #    def run(self, reference: Literal["auto", "harmonic"] = "auto"):
    #        """Fit all properties."""
    #        self._fit_eos()
    #        self._fit_volume_entropy()
    #        self._fit_heat_capacity()
    #        return self

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
