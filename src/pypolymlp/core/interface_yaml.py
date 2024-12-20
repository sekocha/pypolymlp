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
    cell = PolymlpStructure(**yml_data[tag])
    cell.axis = np.array(cell.axis).T
    cell.positions = np.array(cell.positions).T
    return cell
