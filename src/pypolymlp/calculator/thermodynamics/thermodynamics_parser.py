"""Functions for loading thermodynamic properties."""

from collections import defaultdict
from typing import Optional

import numpy as np
import yaml

from pypolymlp.calculator.sscha.sscha_restart import Restart
from pypolymlp.calculator.thermodynamics.thermodynamics_grid import (
    GridPointData,
    GridVT,
)
from pypolymlp.calculator.thermodynamics.ti_utils import load_ti_yaml


def load_sscha_yamls(filenames: tuple[str]) -> list[GridPointData]:
    """Load sscha_results.yaml files."""
    data = []
    for yamlfile in filenames:
        fc2file = "/".join(yamlfile.split("/")[:-1]) + "/fc2.hdf5"
        res = Restart(yamlfile, unit="eV/atom")
        grid = GridPointData(
            volume=res.volume,
            temperature=res.temperature,
            restart=res,
            path_yaml=yamlfile,
            path_fc2=fc2file,
        )
        if res.converge and not res.imaginary:
            grid.static_potential = res.static_potential
            grid.free_energy = res.free_energy + res.static_potential
            grid.entropy = res.entropy
        data.append(grid)
    return data


def load_electron_yamls(filenames: tuple[str]) -> list[GridPointData]:
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
            grid = GridPointData(
                volume=volume,
                temperature=temp,
                free_energy=free_e,
                entropy=entropy,
                path_yaml=yamlfile,
            )
            data.append(grid)
    return data


def load_ti_yamls(filenames: tuple[str], verbose: bool = False) -> list[GridPointData]:
    """Load polymlp_ti.yaml files."""
    data, data_ext = [], []
    for yamlfile in filenames:
        res = load_ti_yaml(yamlfile, verbose=verbose)
        if res is None:
            continue

        data_ti, data_ti_ext = res
        grid = GridPointData(
            volume=data_ti.volume,
            temperature=data_ti.temperature,
            free_energy=data_ti.free_energy,
            entropy=data_ti.entropy,
            energy=data_ti.energy,
            path_yaml=yamlfile,
        )
        data.append(grid)
        grid = GridPointData(
            volume=data_ti_ext.volume,
            temperature=data_ti_ext.temperature,
            free_energy=data_ti_ext.free_energy,
            entropy=data_ti_ext.entropy,
            energy=data_ti_ext.energy,
            path_yaml=yamlfile,
        )
        data_ext.append(grid)
    return data, data_ext


def _count_data_size(data: list, decimals: int = 3):
    """Count number of data entries."""
    count_volumes = defaultdict(int)
    count_temperatures = defaultdict(int)
    for d in data:
        vol = np.round(d.volume, decimals)
        temp = int(np.rint(d.temperature))
        if d.free_energy is not None:
            count_volumes[vol] += 1
            count_temperatures[temp] += 1
    return count_volumes, count_temperatures


def _count_data_minimum_size(data_all: list, decimals: int = 3):
    """Count minimum number of data entries in multiple datasets."""
    num_data = 0
    count_volumes, count_temperatures = None, None
    for data in data_all:
        if data is None:
            continue

        cvols, ctemps = _count_data_size(data, decimals=decimals)
        if num_data == 0:
            count_volumes = cvols
            count_temperatures = ctemps
        else:
            for vol in list(count_volumes.keys()):
                if vol not in cvols:
                    del count_volumes[vol]
                elif count_volumes[vol] > cvols[vol]:
                    count_volumes[vol] = cvols[vol]

            for temp in list(count_temperatures.keys()):
                if temp not in ctemps:
                    del count_temperatures[temp]
                elif count_temperatures[temp] > ctemps[temp]:
                    count_temperatures[temp] = ctemps[temp]
        num_data += 1
    return count_volumes, count_temperatures


def _get_common_grid(data_all: list, decimals: int = 3, n_require: int = 10):
    """Get common grid points of volumes and temperatures."""
    count_vols, count_temps = _count_data_minimum_size(data_all, decimals=decimals)
    volumes = sorted([vol for vol, n in count_vols.items() if n >= n_require])
    temperatures = sorted([t for t, n in count_temps.items()])
    return np.array(volumes), np.array(temperatures)


def _get_grid_data(
    data: list,
    volumes: np.ndarray,
    temperatures: np.ndarray,
    decimals: int = 3,
):
    """Get only data on common grid."""
    if data is None:
        return None

    map_volumes = {v: i for i, v in enumerate(volumes)}
    map_temperatures = {v: i for i, v in enumerate(temperatures)}
    arr = np.full((volumes.shape[0], temperatures.shape[0]), None, dtype=object)
    for d in data:
        vol = np.round(d.volume, decimals)
        temp = np.round(d.temperature, decimals)
        try:
            arr[map_volumes[vol], map_temperatures[temp]] = d
        except:
            pass

    arr[np.equal(arr, None)] = GridPointData(volume=None, temperature=None)
    grid = GridVT(volumes, temperatures, arr)
    return grid


def load_yamls(
    yamls_sscha: list[str],
    yamls_electron: Optional[list[str]] = None,
    yamls_ti: Optional[list[str]] = None,
    yamls_electron_phonon: Optional[list[str]] = None,
    decimals: int = 3,
    n_require: int = 10,
    verbose: bool = False,
):
    """Load yaml files needed for calculating thermodynamics."""
    if verbose:
        print("Loading sscha.yaml files.", flush=True)
    data_sscha = load_sscha_yamls(yamls_sscha)

    data_electron = None
    if yamls_electron is not None:
        if verbose:
            print("Loading electron.yaml files.", flush=True)
        data_electron = load_electron_yamls(yamls_electron)

    data_ti, data_ti_ext = None, None
    if yamls_ti is not None:
        if verbose:
            print("Loading ti.yaml files.", flush=True)
        data_ti, data_ti_ext = load_ti_yamls(yamls_ti, verbose=verbose)

    data_ele_ph = None
    if yamls_electron_phonon is not None:
        if verbose:
            print("Loading electron.yaml files for ele-ph.", flush=True)
        data_ele_ph = load_electron_yamls(yamls_electron_phonon)

    data_all = [data_sscha, data_electron, data_ti, data_ele_ph]
    volumes, temps = _get_common_grid(data_all, decimals=decimals, n_require=n_require)

    grid_sscha = _get_grid_data(data_sscha, volumes, temps, decimals=decimals)
    grid_electron = _get_grid_data(data_electron, volumes, temps, decimals=decimals)
    grid_ti = _get_grid_data(data_ti, volumes, temps, decimals=decimals)
    grid_ti_ext = _get_grid_data(data_ti_ext, volumes, temps, decimals=decimals)
    grid_ele_ph = _get_grid_data(data_ele_ph, volumes, temps, decimals=decimals)

    return (grid_sscha, grid_electron, grid_ti, grid_ti_ext, grid_ele_ph)
