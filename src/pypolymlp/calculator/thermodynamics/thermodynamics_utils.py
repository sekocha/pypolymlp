"""Utility functions for calculating thermodynamic properties."""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from phono3py.file_IO import read_fc2_from_hdf5
from phonopy import Phonopy
from phonopy.units import EVAngstromToGPa

from pypolymlp.calculator.sscha.sscha_utils import Restart
from pypolymlp.calculator.thermodynamics.fit_utils import Polyfit
from pypolymlp.calculator.utils.eos_utils import EOS
from pypolymlp.core.units import EVtoJmol
from pypolymlp.utils.phonopy_utils import structure_to_phonopy_cell


@dataclass
class GridPointData:
    """Dataclass for properties on a volume-temperature grid point."""

    volume: float
    temperature: float
    data_type: Optional[Literal["sscha", "ti", "electron"]] = None
    restart: Optional[Restart] = None
    free_energy: Optional[float] = None
    entropy: Optional[float] = None
    heat_capacity: Optional[float] = None
    harmonic_heat_capacity: Optional[float] = None

    reference_free_energy: Optional[float] = None
    reference_entropy: Optional[float] = None
    reference_heat_capacity: Optional[float] = None

    path_yaml: Optional[float] = None
    path_fc2: Optional[float] = None


class GridData:
    """Dataclass for properties on grid points."""

    def __init__(
        self,
        data: list[GridPointData],
        data_type: Optional[Literal["sscha", "ti", "electron"]] = None,
        verbose: bool = False,
    ):
        """Init method."""
        self._data = data
        self._data_type = data_type
        self._verbose = verbose
        self._volumes = None
        self._temperatures = None
        self._grid = None
        self._scan_data()

        self._eos_fits = None
        self._sv_fits = None
        self._st_fits = None
        self._cv_fits = None

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
            for itemp, data in enumerate(self._grid.T):
                n_data = len([d.volume for d in data if d is not None])
                print("- Temp.:", self._temperatures[itemp], flush=True)
                print("  n_data:", n_data, flush=True)

        return self

    def fit_eos(self):
        """Fit volume-free energy data to Vinet EOS."""
        if self._verbose:
            print("Volume-FreeEnergy fitting.", flush=True)

        self._eos_fits = []
        for itemp, data in enumerate(self._grid.T):
            volumes = [d.volume for d in data if d is not None]
            free_energies = [d.free_energy for d in data if d is not None]
            try:
                eos = EOS(volumes, free_energies)
            except:
                eos = None
            self._eos_fits.append(eos)
        return self

    def fit_entropy_volume(self, max_order: int = 6):
        """Fit volume-entropy data using polynomial."""
        if self._verbose:
            print("Volume-Entropy fitting.", flush=True)

        self._sv_fits = []
        for itemp, data in enumerate(self._grid.T):
            volumes = [d.volume for d in data if d is not None]
            entropies = [d.entropy for d in data if d is not None]
            if len(entropies) > 4:
                polyfit = Polyfit(volumes, entropies)
                polyfit.fit(max_order=max_order, intercept=True, add_sqrt=False)
                self._sv_fits.append(polyfit)
                if self._verbose:
                    print("- Temp.:", self._temperatures[itemp], flush=True)
                    print("  RMSE: ", polyfit.error, flush=True)
                    print("  model:", polyfit.best_model, flush=True)
            else:
                self._sv_fits.append(None)

        return self

    def fit_cv_volume(self, max_order: int = 4):
        """Fit volume-Cv data using polynomial."""
        if self._verbose:
            print("Volume-Cv fitting.", flush=True)
        self._cv_fits = []
        for itemp, data in enumerate(self._grid.T):
            volumes = [d.volume for d in data if d is not None]
            cvs = [d.heat_capacity for d in data if d is not None]
            if len(cvs) > 4:
                polyfit = Polyfit(volumes, cvs)
                polyfit.fit(max_order=max_order, intercept=True, add_sqrt=False)
                self._cv_fits.append(polyfit)
                if self._verbose:
                    print("- Temp.:", self._temperatures[itemp], flush=True)
                    print("  RMSE: ", polyfit.error, flush=True)
                    print("  model:", polyfit.best_model, flush=True)
            else:
                self._cv_fits.append(None)
        return self

    def fit_entropy_temperature(self, max_order: int = 4, reference: bool = True):
        """Fit temperature-entropy data using polynomial."""
        self._st_fits = []
        for ivol, data in enumerate(self._grid):
            points = np.array([d for d in data if d is not None])
            if len(points) > 4:
                points = calculate_reference(points)
                temperatures = np.array([p.temperature for p in points])
                entropies = np.array([p.entropy for p in points])
                ref = np.array([p.reference_entropy for p in points])
                del_entropies = entropies - ref

                polyfit = Polyfit(temperatures, del_entropies)
                polyfit.fit(max_order=max_order, intercept=False, add_sqrt=True)
                self._st_fits.append(polyfit)
                if self._verbose:
                    print("- Volume:", np.round(self._volumes[ivol], 3), flush=True)
                    print("  RMSE:  ", polyfit.error, flush=True)
                    print("  model: ", polyfit.best_model, flush=True)

                cv_from_ref = temperatures * polyfit.eval_derivative(temperatures)
                for p, val in zip(points, cv_from_ref):
                    p.heat_capacity = p.reference_heat_capacity + val * EVtoJmol
            else:
                self._st_fits.append(None)

        return self

    def eval_cp(self):
        """Evaluate Cp - Cv."""
        if self._eos_fits is None:
            raise RuntimeError("EOS functions not found.")
        if self._sv_fits is None:
            raise RuntimeError("S-V functions not found.")
        if self._cv_fits is None:
            raise RuntimeError("Cv-V functions not found.")

        self._eq_cp = []
        for itemp, data in enumerate(self._grid.T):
            temp = self._temperatures[itemp]
            eos = self._eos_fits[itemp]
            sv = self._sv_fits[itemp]
            cv = self._cv_fits[itemp]
            if eos is not None and sv is not None and cv is not None:
                v0, b0 = eos.v0, eos.b0
                s_deriv = sv.eval_derivative(v0)
                bm = b0 / EVAngstromToGPa
                add = temp * v0 * (s_deriv**2) / bm
                add *= EVtoJmol
                cp = cv.eval(v0) + add
                self._eq_cp.append(cp)
            else:
                self._eq_cp.append(None)
        return np.array(self._eq_cp)

    def eval_entropy(self):
        """Evaluate entropies at equilibrium volumes."""
        self._eq_entropies = []
        for itemp, data in enumerate(self._grid.T):
            eos = self._eos_fits[itemp]
            sv = self._sv_fits[itemp]
            self._eq_entropies.append(sv.eval(eos.v0))
        return np.array(self._eq_entropies)

    def save_data(self, filename="polymlp_thermodynamics.yaml"):
        """Save raw data and fitted functions."""
        f = open(filename, "w")
        print("units:", file=f)
        print("  temperature:   K", file=f)
        print("  volume:        angstroms^3/atom", file=f)
        print("  bulk_modulus:  GPa", file=f)
        print("  pressure:      GPa", file=f)
        print("  free_energy:   eV/atom", file=f)
        print("  entropy:       J/K/mol (/Avogadro's number of atoms)", file=f)
        print("  heat_capacity: J/K/mol (/Avogadro's number of atoms)", file=f)
        print("", file=f)

        print("equilibrium_properties:", file=f)
        for itemp, data in enumerate(self._grid.T):
            print("- temperature:      ", self._temperatures[itemp], file=f)
            if self._eos_fits is not None and self._eos_fits[itemp] is not None:
                print("  volume:           ", self._eos_fits[itemp].v0, file=f)
                print("  bulk_modulus:     ", self._eos_fits[itemp].b0, file=f)
                print("  free_energy:      ", self._eos_fits[itemp].e0, file=f)
            if self._eq_entropies is not None and self._eq_entropies[itemp] is not None:
                val = self._eq_entropies[itemp] * EVtoJmol
                print("  entropy:          ", val, file=f)
            if self._eq_cp is not None and self._eq_cp[itemp] is not None:
                print("  heat_capacity_cp: ", self._eq_cp[itemp], file=f)
            print("", file=f)

        #        self._save_2d_array(f, tag="free_energies")
        #        self._save_2d_array(f, tag="eos_fit_data")
        #        self._save_2d_array(f, tag="entropies")
        #        self._save_2d_array(f, tag="gibbs_free_energies")

        #        print("cv:", file=f)
        #        for ivol, vol in enumerate(self._volumes):
        #            print("- volume:", vol, file=f)
        #            print("  values:", file=f)
        #            for itemp, temp in enumerate(self._temperatures):
        #                grid = self._grid_vt[ivol, itemp]
        #                print("  -", [temp, grid.heat_capacity], file=f)
        #            print("", file=f)

        f.close()

    #    def _save_2d_array(self, f, tag: str):
    #        """Save dict of 2D array"""
    #        print(tag + ":", file=f)
    #        for itemp, temp in enumerate(self._temperatures):
    #            array1d = getattr(self._grid_t[itemp], tag)
    #            print("- temperature:", temp, file=f)
    #            print("  values:", file=f)
    #            for vals in array1d:
    #                print("  -", list(vals), file=f)
    #            print("", file=f)
    #        return self
    #
    @property
    def grid(self):
        """Return grid points.

        Rows and columns correspond to volumes and temperatures, respectively.
        """
        return self._grid

    @property
    def temperatures(self):
        """Retrun temperatures."""
        return self._temperatures

    @property
    def volumes(self):
        """Retrun volumes."""
        return self._volumes

    @property
    def eos_fits(self):
        """Retrun EOS fits."""
        return self._eos_fits

    @property
    def sv_fits(self):
        """Retrun S-V fits."""
        return self._sv_fits

    @property
    def st_fits(self):
        """Retrun S-T fits."""
        return self._st_fits

    @property
    def cv_fits(self):
        """Retrun Cv-V fits."""
        return self._cv_fits

    def get_gibbs_free_energies(self, volumes: np.ndarray):
        """Return Gibbs free energy.

        Return
        ------
        Gibbs free energies.
            Array of (temperature index, pressure in GPa, Gibbs free energy).
        """
        if self._eos_fits is None:
            raise RuntimeError("EOS functions not found.")
        return np.array([eos.eval_gibbs_pressure(volumes) for eos in self._eos_fits])


def calculate_reference(
    grid_points: list[GridPointData],
    mesh: np.ndarray = (10, 10, 10),
):
    """Return reference entropies.

    Harmonic phonon entropies calculated with SSCHA FC2 and frequencies
    is used as reference entropies to fit entropies with respect to temperature.
    """
    temperatures = np.array([p.temperature for p in grid_points])

    ref_id = 0
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


def calculate_entropies_from_free_energy(grid_data: np.array):
    """Compute entropies from temperature-free-energy fitting."""
    pass


def calculate_heat_capacities_from_entropies(grid_data: np.array):
    """Compute heat capacities from temperature-entropy fitting."""
    pass


def load_sscha_yamls(filenames: tuple[str], verbose: bool = False) -> GridData:
    """Load sscha_results.yaml files."""
    data = []
    for yamlfile in filenames:
        res = Restart(yamlfile, unit="eV/atom")
        if res.converge and not res.imaginary:
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
            grid.free_energy = res.free_energy + res.static_potential
            grid.entropy = res.entropy
            grid.harmonic_heat_capacity = res.harmonic_heat_capacity
            data.append(grid)
    return GridData(data=data, data_type="sscha", verbose=verbose)


def load_ti_yamls(self, filenames: tuple[str]) -> GridData:
    """Load polymlp_ti.yaml files."""
    pass


def load_electron_yamls(self, filenames: tuple[str]) -> GridData:
    """Load electron.yaml files."""
    pass
