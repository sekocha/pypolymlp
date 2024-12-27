"""Interfaces for pypolymlp yaml files."""

from typing import Optional

import numpy as np
import yaml

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpStructure
from pypolymlp.core.interface_datasets import set_dataset_from_structures


def set_dataset_from_sscha_yamls(
    yamlfiles: list[str],
    element_order: Optional[bool] = None,
) -> PolymlpDataDFT:
    """Return DFT dataset by loading sscha_results.yaml files."""
    structures, free_energies = parse_sscha_yamls(yamlfiles)
    dft = set_dataset_from_structures(
        structures,
        free_energies,
        forces=None,
        stresses=None,
        element_order=element_order,
    )
    return dft


def parse_sscha_yamls(yamlfiles: list[str]):
    """Parse sscha_results.yaml files."""
    free_energies, structures = [], []
    for yfile in yamlfiles:
        yml = yaml.safe_load(open(yfile))
        fvib = float(yml["properties"]["free_energy"])
        free_energies.append(fvib * 0.01036426965)  # kJ/mol -> eV/unitcell
        unitcell = parse_structure_from_yaml(yml, tag="unitcell")
        unitcell.name = yfile
        structures.append(unitcell)

    return structures, np.array(free_energies)


def parse_structure_from_yaml(yml_data: yaml, tag: str = "unitcell"):
    """Parse structure in yaml data."""
    cell_dict = yml_data[tag]
    cell_dict["axis"] = np.array(cell_dict["axis"]).T
    cell_dict["positions"] = np.array(cell_dict["positions"]).T
    cell = PolymlpStructure(**cell_dict)
    return cell


def set_dataset_from_electron_yamls(
    yamlfiles: list[str],
    temperature: float = 300.0,
    element_order: Optional[bool] = None,
) -> PolymlpDataDFT:
    """Return DFT dataset by loading electron.yaml files."""
    structures, free_energies = parse_electron_yamls(
        yamlfiles,
        temperature=temperature,
    )
    dft = set_dataset_from_structures(
        structures,
        free_energies,
        forces=None,
        stresses=None,
        element_order=element_order,
    )
    return dft


def parse_electron_yamls(yamlfiles: list[str], temperature: float = 300.0):
    """Parse electron.yaml files."""
    free_energies, structures = [], []
    for yfile in yamlfiles:
        yml = yaml.safe_load(open(yfile))
        for prop in yml["properties"]:
            if np.isclose(float(prop["temperature"]), temperature):
                free_energies.append(float(prop["free_energy"]))
                unitcell = parse_structure_from_yaml(yml, tag="structure")
                unitcell.name = yfile
                structures.append(unitcell)
                break
    return structures, np.array(free_energies)
