"""Interfaces for pypolymlp yaml files."""

from typing import Literal, Optional

import numpy as np
import yaml

from pypolymlp.core.data_format import PolymlpDataDFT
from pypolymlp.core.interface_datasets import set_dataset_from_structures
from pypolymlp.core.units import EVtoKJmol
from pypolymlp.utils.yaml_utils import load_cell


def set_dataset_from_sscha_yamls(
    yamlfiles: list[str],
    element_order: Optional[bool] = None,
) -> PolymlpDataDFT:
    """Return DFT dataset by loading sscha_results.yaml files."""
    structures, free_energies, forces = parse_sscha_yamls(yamlfiles)
    dft = set_dataset_from_structures(
        structures,
        free_energies,
        forces=forces,
        stresses=None,
        element_order=element_order,
    )
    return dft


def parse_sscha_yamls(yamlfiles: list[str]):
    """Parse sscha_results.yaml files."""
    free_energies, structures = [], []
    forces = []
    for yfile in yamlfiles:
        yml = yaml.safe_load(open(yfile))
        if yml["status"]["converge"] and not yml["status"]["imaginary"]:
            if "average_forces" in yml and "supercell" in yml:
                supercell = load_cell(yaml_data=yml, tag="supercell")
                supercell.name = yfile
                n_cells = int(yml["supercell"]["n_unitcells"])
                fvib = float(yml["properties"]["free_energy"])
                free_energies.append(fvib * n_cells / EVtoKJmol)  # kJ/mol->eV/supercell
                forces.append(np.array(yml["average_forces"]).T)
                structures.append(supercell)

    return structures, np.array(free_energies), forces


def parse_electron_yamls(yamlfiles: list[str]):
    """Parse electron.yaml files."""
    yaml_data = []
    for yfile in yamlfiles:
        yml = yaml.safe_load(open(yfile))
        yml["name"] = yfile
        yaml_data.append(yml)
    return yaml_data


def extract_electron_properties(
    yaml_data: list[dict],
    temperature: float = 300.0,
    target: Literal[
        "free_energy",
        "energy",
        "entropy",
        "specific_heat",
    ] = "free_energy",
):
    """Extract property data from electron.yaml files."""
    properties, structures = [], []
    for yml in yaml_data:
        unitcell = load_cell(yaml_data=yml, tag="structure")
        unitcell.name = yml["name"]
        structures.append(unitcell)
        for prop in yml["properties"]:
            if np.isclose(float(prop["temperature"]), temperature):
                properties.append(float(prop[target]))
                break
    return structures, np.array(properties)


def set_dataset_from_electron_yamls(
    yaml_data: list[dict],
    temperature: float = 300.0,
    target: Literal[
        "free_energy",
        "energy",
        "entropy",
        "specific_heat",
    ] = "free_energy",
    element_order: Optional[bool] = None,
) -> PolymlpDataDFT:
    """Return DFT dataset by loading electron.yaml files."""
    structures, properties = extract_electron_properties(
        yaml_data,
        temperature=temperature,
        target=target,
    )
    dft = set_dataset_from_structures(
        structures,
        properties,
        forces=None,
        stresses=None,
        element_order=element_order,
    )
    return dft
