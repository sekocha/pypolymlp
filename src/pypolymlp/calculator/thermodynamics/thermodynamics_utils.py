"""Utility functions for calculating thermodynamic properties."""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from phonopy.units import EVAngstromToGPa

from pypolymlp.calculator.sscha.sscha_utils import Restart
from pypolymlp.core.units import EVtoJmol


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


@dataclass
class FittedModels:
    """Dataclass for fitted thermodynamics functions."""

    volumes: np.ndarray
    temperatures: np.ndarray
    eos_fits: Optional[list] = None
    sv_fits: Optional[list] = None
    st_fits: Optional[list] = None
    cv_fits: Optional[list] = None

    def extract(self, itemp: int):
        """Retrun fitted functions for at a temperature index."""
        eos = self.eos_fits[itemp] if self.eos_fits is not None else None
        sv = self.sv_fits[itemp] if self.sv_fits is not None else None
        cv = self.cv_fits[itemp] if self.cv_fits is not None else None
        return eos, sv, cv

    def eval_eq_entropy(self, itemp: int):
        """Evaluate entropy at equilibrium volume."""
        if self.eos_fits is None:
            raise RuntimeError("EOS functions not found.")
        if self.sv_fits is None:
            raise RuntimeError("S-V functions not found.")
        if self.eos_fits[itemp] is None or self.sv_fits[itemp] is None:
            return None

        return self.sv_fits[itemp].eval(self.eos_fits[itemp].v0)

    def eval_eq_cv(self, itemp: int):
        """Evaluate Cv contribution at equilibrium volume."""
        if self.eos_fits is None:
            raise RuntimeError("EOS functions not found.")
        if self.cv_fits is None:
            raise RuntimeError("Cv-V functions not found.")
        if self.eos_fits[itemp] is None or self.cv_fits[itemp] is None:
            return None
        return self.cv_fits[itemp].eval(self.eos_fits[itemp].v0)

    def eval_eq_cp(self, itemp: int):
        """Evaluate Cp at equilibrium volume."""
        cv_val = self.eval_eq_cv(itemp)
        eos, sv, _ = self.extract(itemp)
        v0, b0 = eos.v0, eos.b0
        s_deriv = sv.eval_derivative(v0)
        temp = self.temperatures[itemp]
        try:
            bm = b0 / EVAngstromToGPa
            add = temp * v0 * (s_deriv**2) / bm
            add *= EVtoJmol
        except:
            return None

        return cv_val + add


# def load_sscha_yamls(filenames: tuple[str], verbose: bool = False) -> Thermodynamics:
#    """Load sscha_results.yaml files."""
#    data = []
#    for yamlfile in filenames:
#        res = Restart(yamlfile, unit="eV/atom")
#        if res.converge and not res.imaginary:
#            n_atom = len(res.unitcell.elements)
#            volume = np.round(res.volume, decimals=12) / n_atom
#            temp = np.round(res.temperature, decimals=3)
#            grid = GridPointData(
#                volume=volume,
#                temperature=temp,
#                data_type="sscha",
#                restart=res,
#                path_yaml=yamlfile,
#                path_fc2="/".join(yamlfile.split("/")[:-1]) + "/fc2.hdf5",
#            )
#            grid.free_energy = res.free_energy + res.static_potential
#            grid.entropy = res.entropy
#            grid.harmonic_heat_capacity = res.harmonic_heat_capacity
#            data.append(grid)
#    return Thermodynamics(data=data, data_type="sscha", verbose=verbose)
#
#
# def load_ti_yamls(self, filenames: tuple[str]) -> GridData:
#    """Load polymlp_ti.yaml files."""
#    pass
#
#
# def load_electron_yamls(self, filenames: tuple[str]) -> GridData:
#    """Load electron.yaml files."""
#    pass
