"""Utility functions for calculating thermodynamic properties."""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from pypolymlp.calculator.sscha.sscha_utils import Restart


@dataclass
class GridPointData:
    """Dataclass for properties on a volume-temperature grid point."""

    volume: float
    temperature: float
    data_type: Optional[Literal["sscha", "ti", "electron"]] = None
    restart: Optional[Restart] = None
    free_energy: Optional[float] = None
    entropy: Optional[float] = None
    reference_entropy: Optional[float] = None
    heat_capacity: Optional[float] = None
    harmonic_heat_capacity: Optional[float] = None
    reference_heat_capacity: Optional[float] = None
    path_yaml: Optional[float] = None
    path_fc2: Optional[float] = None


class GridData:
    """Dataclass for properties on grid points."""

    def __init__(
        self,
        data: list[GridPointData],
        data_type: Optional[Literal["sscha", "ti", "electron"]] = None,
    ):
        """Init method."""
        self._data = data
        self._data_type = data_type
        self._scan_data()

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

        return self

    @property
    def grid(self):
        """Return grid points.

        Rows and columns correspond to volumes and temperatures, respectively.
        """
        return self._grid


def load_sscha_yamls(filenames: tuple[str]) -> GridData:
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

        data.append(grid)
    return GridData(data=data, data_type="sscha")


def load_ti_yamls(self, filenames: tuple[str]):
    """Load polymlp_ti.yaml files."""
    pass


def load_electron_yamls(self, filenames: tuple[str]):
    """Load electron.yaml files."""
    pass
