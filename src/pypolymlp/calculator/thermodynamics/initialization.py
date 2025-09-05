"""Utility functions for initializing thermodynamic property calculation."""

# from dataclasses import dataclass

import numpy as np
import yaml

from pypolymlp.calculator.md.md_utils import load_thermodynamic_integration_yaml
from pypolymlp.calculator.sscha.sscha_utils import Restart
from pypolymlp.calculator.thermodynamics.thermodynamics_utils import GridPointData

# from pypolymlp.core.units import EVtoJmol

# from phono3py.file_IO import read_fc2_from_hdf5
# from phonopy import Phonopy


def load_sscha_yamls(filenames: tuple[str]) -> list[GridPointData]:
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
        else:
            grid.free_energy = None
            grid.entropy = None
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
            # cv = float(prop["specific_heat"]) * EVtoJmol / n_atom
            grid = GridPointData(
                volume=volume,
                temperature=temp,
                data_type="electron",
                free_energy=free_e,
                entropy=entropy,
                # heat_capacity=cv,
                path_yaml=yamlfile,
            )
            data.append(grid)
    return data


def _check_melting(log: np.ndarray):
    """Check whether MD simulation converges to a melting state."""
    if np.isclose(log[0, 2], 0.0):
        return False
    try:
        displacement_ratio = log[-1, 2] / log[0, 2]
        return displacement_ratio > 2.0
    except:
        return False


def load_ti_yamls(filenames: tuple[str], verbose: bool = False) -> list[GridPointData]:
    """Load polymlp_ti.yaml files."""
    data = []
    for yamlfile in filenames:
        res = load_thermodynamic_integration_yaml(yamlfile)
        temp, volume, free_e, ent, cv, eng, log = res
        if _check_melting(log):
            if verbose:
                message = yamlfile + " was eliminated (found to be in a melting state)."
                print(message, flush=True)
        else:
            grid = GridPointData(
                volume=volume,
                temperature=temp,
                data_type="ti",
                free_energy=free_e,
                entropy=ent,
                # heat_capacity=cv,
                # energy=eng,
                path_yaml=yamlfile,
            )
            data.append(grid)
    return data
