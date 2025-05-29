"""Utility functions for calculating thermodynamic properties."""

from dataclasses import dataclass

import numpy as np
import yaml

from pypolymlp.calculator.thermodynamics.thermodynamics_utils import FittedModels
from pypolymlp.core.units import EVtoJmol


@dataclass
class ThermodynamicProperties:
    """Dataclass of thermodynamic properties."""

    temperatures: np.ndarray
    eq_volumes: np.ndarray
    bm: np.ndarray
    eq_helmholtz: np.ndarray
    eq_entropy: np.ndarray
    eq_cp: np.ndarray
    eos_data: list
    eos_fit_data: np.ndarray
    gibbs: np.ndarray

    def get_T_F(self):
        """Return temperature-Helmholtz data at equilibrium volumes."""
        return np.array([self.temperatures, self.eq_helmholtz]).T


def save_thermodynamics_yaml(
    volumes: np.ndarray,
    temperatures: np.ndarray,
    models: FittedModels,
    eq_entropies: np.ndarray,
    eq_cp: np.ndarray,
    free_energies: np.ndarray,
    filename: str = "polymlp_thermodynamics.yaml",
):
    """Save fitted thermodynamics properties."""
    np.set_printoptions(legacy="1.21")
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
    for itemp, temp in enumerate(temperatures):
        eos, _, _ = models.extract(itemp)
        print("- temperature:      ", temp, file=f)
        if eos is not None:
            print("  volume:           ", eos.v0, file=f)
            print("  bulk_modulus:     ", eos.b0, file=f)
            print("  free_energy:      ", eos.e0, file=f)
        if eq_entropies is not None and eq_entropies[itemp] is not None:
            val = eq_entropies[itemp] * EVtoJmol
            print("  entropy:          ", val, file=f)
        if eq_cp is not None and eq_cp[itemp] is not None:
            print("  heat_capacity_cp: ", eq_cp[itemp], file=f)
        print("", file=f)

    print("data_helmholtz_volume:", file=f)
    print("  volume:", list(volumes), file=f)
    print(file=f)
    print("  eos_data:", file=f)
    for itemp, temp in enumerate(temperatures):
        print("  - temperature:", temp, file=f)
        print("    free_energy:", list(free_energies[:, itemp]), file=f)
        print(file=f)

    vol_min = np.round(np.min(volumes) - 1.0, 2)
    vol_max = np.round(np.max(volumes) + 1.0, 2)
    volumes_tofit = np.round(np.arange(vol_min, vol_max, 0.01), 2)
    fit_free_energies = models.eval_helmholtz_free_energies(volumes_tofit)
    print("fit_helmholtz_volume:", file=f)
    print("  volume:", list(volumes_tofit), file=f)
    print(file=f)
    print("  eos:", file=f)
    for itemp, temp in enumerate(temperatures):
        print("  - temperature:", temp, file=f)
        print("    free_energy:", list(fit_free_energies[:, itemp]), file=f)
        print(file=f)

    gibbs = models.eval_gibbs_free_energies(volumes_tofit)
    print("fit_gibbs_pressure:", file=f)
    for itemp, temp in enumerate(temperatures):
        print("- temperature:", temp, file=f)
        print("  pressure:", list(gibbs[itemp][:, 0]), file=f)
        print("  free_energy:", list(gibbs[itemp][:, 1]), file=f)
        print(file=f)
    f.close()


def load_thermodynamics_yaml(filename: str = "polymlp_thermodynamics.yaml"):
    """Load thermodynamics.yaml."""
    data = yaml.safe_load(open(filename))
    temperatures = [float(d["temperature"]) for d in data["equilibrium_properties"]]
    eq_volumes = [float(d["volume"]) for d in data["equilibrium_properties"]]
    bm = [float(d["bulk_modulus"]) for d in data["equilibrium_properties"]]
    eq_helmholtz = [float(d["free_energy"]) for d in data["equilibrium_properties"]]
    eq_entropy = [float(d["entropy"]) for d in data["equilibrium_properties"]]
    eq_cp = [float(d["heat_capacity_cp"]) for d in data["equilibrium_properties"]]

    eos_data, eos_fit_data = [], []
    volumes = np.array(data["data_helmholtz_volume"]["volume"])
    for d in data["data_helmholtz_volume"]["eos_data"]:
        add = [[v, f] for v, f in zip(volumes, d["free_energy"]) if f != "None"]
        eos_data.append(np.array(add).astype(float))

    volumes = np.array(data["fit_helmholtz_volume"]["volume"])
    for d in data["fit_helmholtz_volume"]["eos"]:
        eos_fit_data.append(np.array([volumes, d["free_energy"]]).T)
    eos_fit_data = np.array(eos_fit_data).astype(float)

    gibbs = []
    for d in data["fit_gibbs_pressure"]:
        gibbs.append(np.array([d["pressure"], d["free_energy"]]).T)
    gibbs = np.array(gibbs).astype(float)

    properties = ThermodynamicProperties(
        temperatures=temperatures,
        eq_volumes=eq_volumes,
        bm=bm,
        eq_helmholtz=eq_helmholtz,
        eq_entropy=eq_entropy,
        eq_cp=eq_cp,
        eos_data=eos_data,
        eos_fit_data=eos_fit_data,
        gibbs=gibbs,
    )
    return properties
