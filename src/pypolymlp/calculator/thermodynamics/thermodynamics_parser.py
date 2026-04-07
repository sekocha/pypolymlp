"""Functions for loading thermodynamic properties."""

# import copy
from collections import defaultdict
from typing import Literal, Optional

import numpy as np
import yaml

# from pypolymlp.calculator.thermodynamics.init_ti import load_ti_yaml
from pypolymlp.calculator.sscha.sscha_restart import Restart
from pypolymlp.calculator.thermodynamics.thermodynamics_grid import (
    GridPointData,
    GridVT,
)

# from pypolymlp.calculator.thermodynamics.init import (
#     calculate_harmonic_free_energies,
#     calculate_reference,
# )

# from pypolymlp.calculator.thermodynamics.thermodynamics_utils import (
#     sum_matrix_data,
# )


def load_sscha_yamls(filenames: tuple[str]) -> list[GridPointData]:
    """Load sscha_results.yaml files."""
    data = []
    for yamlfile in filenames:
        fc2file = "/".join(yamlfile.split("/")[:-1]) + "/fc2.hdf5"
        res = Restart(yamlfile, unit="eV/atom")
        grid = GridPointData(
            volume=np.round(res.volume, decimals=12),
            temperature=np.round(res.temperature, decimals=3),
            data_type="sscha",
            restart=res,
            path_yaml=yamlfile,
            path_fc2=fc2file,
        )
        if res.converge and not res.imaginary:
            grid.static_potential = res.static_potential
            grid.free_energy = res.free_energy + res.static_potential
            grid.harmonic_free_energy = res.harmonic_free_energy
            grid.entropy = res.entropy
        else:
            grid.free_energy = None
            grid.entropy = None
        data.append(grid)
    return data


def load_electron_yamls(
    filenames: tuple[str],
    data_type: Literal["electron", "electron_ph"] = "electron",
) -> list[GridPointData]:
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
                data_type=data_type,
                free_energy=free_e,
                entropy=entropy,
                path_yaml=yamlfile,
            )
            data.append(grid)
    return data


# def load_ti_yamls(
#     filenames: tuple[str],
#     extrapolation: bool = False,
#     verbose: bool = False,
# ) -> list[GridPointData]:
#     """Load polymlp_ti.yaml files."""
#     data = []
#     for yamlfile in filenames:
#         res = load_ti_yaml(
#             yamlfile,
#             extrapolation=extrapolation,
#             verbose=verbose,
#         )
#         if res is not None:
#             temp, volume, free_e, energy, entropy, cv = res
#             grid = GridPointData(
#                 volume=volume,
#                 temperature=temp,
#                 data_type="ti",
#                 free_energy=free_e,
#                 entropy=entropy,
#                 energy=energy,
#                 heat_capacity=cv,
#                 path_yaml=yamlfile,
#             )
#             data.append(grid)
#         else:
#             if verbose:
#                 message = " was eliminated (failed or in a melting state)."
#                 print(yamlfile + message, flush=True)
#     return data


def _count_data_size(data: list, decimals: int = 3):
    """Count number of data entries."""
    count_volumes = defaultdict(int)
    count_temperatures = defaultdict(int)
    for d in data:
        vol = np.round(d.volume, decimals)
        temp = np.round(d.temperature, decimals)
        if d.free_energy is not None:
            count_volumes[vol] += 1
            count_temperatures[temp] += 1
    return count_volumes, count_temperatures


def _count_data_minimum_size(data_all: list, decimals: int = 3):
    """Count minimum number of data entries in multiple datasets."""
    count_volumes, count_temperatures = dict(), dict()
    for data in data_all:
        if data is None:
            continue
        cvols, ctemps = _count_data_size(data, decimals=decimals)
        for vol, n in cvols.items():
            if vol in count_volumes and n > count_volumes[vol]:
                continue
            count_volumes[vol] = n
        for temp, n in ctemps.items():
            if vol in count_temperatures and n > count_temperatures[temp]:
                continue
            count_temperatures[temp] = n
    return count_volumes, count_temperatures


def _get_common_grid(data_all: list, decimals: int = 3, n_require: int = 10):
    """Get common grid points of volumes and temperatures."""
    count_vols, count_temps = _count_data_minimum_size(data_all, decimals=decimals)
    volumes = sorted([vol for vol, n in count_vols.items() if n >= n_require])
    temperatures = sorted([t for t, n in count_temps.items() if n >= n_require])
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

    grid = GridVT(volumes, temperatures, arr)
    return grid


def load_yamls(
    yamls_sscha: list[str],
    yamls_electron: Optional[list[str]] = None,
    yamls_ti: Optional[list[str]] = None,
    yamls_electron_phonon: Optional[list[str]] = None,
    extrapolation_ti: bool = False,
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

    data_ti = None
    if yamls_ti is not None:
        if verbose:
            print("Loading ti.yaml files.", flush=True)
        # data3 = load_ti_yamls(
        #     yamls_ti,
        #     extrapolation=extrapolation_ti,
        #     verbose=verbose,
        # )

    data_all = [data_sscha, data_sscha, data_electron, data_ti]

    volumes, temps = _get_common_grid(data_all, decimals=decimals, n_require=n_require)
    grid_sscha = _get_grid_data(data_sscha, volumes, temps, decimals=decimals)
    grid_electron = _get_grid_data(data_electron, volumes, temps, decimals=decimals)
    grid_ti = _get_grid_data(data_ti, volumes, temps, decimals=decimals)
    print(grid_sscha)
    print(grid_electron)
    print(grid_ti)
    print(grid_sscha.volumes)

    return (grid_sscha, grid_electron, grid_ti)


#    # Set reference
#    sscha.calculate_reference()
#    if electron is not None:
#        electron.copy_reference(sscha.grid)
#    if ti is not None:
#        ti.copy_reference(sscha.grid)
#
#    # Set reference term for TI (multiple reference states)
#    # if ti is not None:
#    #     ti_ref = copy.deepcopy(sscha)
#    #     f1 = sscha.get_data(attr="harmonic_free_energy")
#    #     ti_ref.replace_free_energies(f1)
#    #     ti_ref.fit_free_energy_temperature(max_order=4, intercept=True)
#
#    #     f1 = sscha.get_data(attr="static_potential")
#    #     f2 = ti_ref.get_data(attr="free_energy")
#    #     f_sum = sum_matrix_data(f1, f2)
#    #     ti_ref.replace_free_energies(f_sum)
#
#    # Set reference term for TI (single reference state)
#    if ti is not None:
#        ti_ref = copy.deepcopy(sscha)
#        f1 = sscha.get_data(attr="reference_free_energy")
#        s1 = sscha.get_data(attr="reference_entropy")
#        f2 = sscha.get_data(attr="static_potential")
#        f_sum = sum_matrix_data(f1, f2)
#        ti_ref.replace_free_energies(f_sum)
#        ti_ref.replace_entropies(s1)
#
#    if yamls_electron_phonon is not None:
#        if verbose:
#            print("Loading electron.yaml (sscha) files.", flush=True)
#        data4 = load_electron_yamls(yamls_electron_phonon, data_type="electron_ph")
#        electron_ph = Thermodynamics(
#            data=data4, data_type="electron_ph", verbose=verbose
#        )
#        sscha, electron_ph = _adjust_to_common_grid(sscha, electron_ph)
#        if yamls_electron is not None:
#            electron, electron_ph = _adjust_to_common_grid(electron, electron_ph)
#        if yamls_ti is not None:
#            ti, electron_ph = _adjust_to_common_grid(ti, electron_ph)
#            ti_ref, electron_ph = _adjust_to_common_grid(ti_ref, electron_ph)
#    else:
#        electron_ph = None
#
#    return sscha, electron, ti, ti_ref, electron_ph
