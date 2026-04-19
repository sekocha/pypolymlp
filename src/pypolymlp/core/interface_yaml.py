"""Interfaces for pypolymlp yaml files."""

from typing import Literal, Optional

import numpy as np
import yaml

from pypolymlp.core.dataset_utils import DatasetDFT
from pypolymlp.core.units import EVtoKJmol
from pypolymlp.utils.yaml_utils import load_cell


def set_dataset_from_sscha_yamls(
    yamlfiles: list[str],
    elements: Optional[tuple] = None,
    symmetrized: bool = True,
) -> DatasetDFT:
    """Return DFT dataset by loading sscha_results.yaml files."""
    structures, free_energies, forces, stress_tensors = parse_sscha_yamls(
        yamlfiles,
        symmetrized=symmetrized,
    )
    dft = DatasetDFT(
        structures,
        free_energies,
        forces=forces,
        stresses=stress_tensors,
        elements=elements,
    )
    return dft


def parse_sscha_yamls(yamlfiles: list[str], symmetrized: bool = True):
    """Parse sscha_results.yaml files."""
    free_energies, structures, forces, stress_tensors = [], [], [], []
    for yfile in yamlfiles:
        yml = yaml.safe_load(open(yfile))
        if not yml["status"]["converge"]:
            continue
        if "supercell" not in yml:
            continue
        if "average_forces" not in yml:
            continue
        if "average_stress_tensor" not in yml:
            continue

        res = _get_sscha_properties(yml, yfile, symmetrized=symmetrized)
        structures.append(res[0])
        free_energies.append(res[1])
        forces.append(res[2])
        stress_tensors.append(res[3])

    return (structures, np.array(free_energies), forces, stress_tensors)


def _get_sscha_properties(yml: dict, name: str, symmetrized: bool = True):
    """Get SSCHA properties from a single yaml data."""
    fvib = float(yml["properties"]["free_energy"])
    if symmetrized:
        unitcell = load_cell(yaml_data=yml, tag="unitcell")
        unitcell.name = name
        free_energy = fvib / EVtoKJmol
        force = np.array(yml["symmetrized_average_forces"]).T
        stress_tensor = np.array(yml["symmetrized_average_stress_tensor"])
        return unitcell, free_energy, force, stress_tensor
    else:
        supercell = load_cell(yaml_data=yml, tag="supercell")
        supercell.name = name
        n_cells = int(yml["supercell"]["n_unitcells"])
        free_energy = fvib * n_cells / EVtoKJmol  # kJ/mol->eV/supercell
        force = np.array(yml["average_forces"]).T
        stress_tensor = np.array(yml["average_stress_tensor"]) * n_cells
        return supercell, free_energy, force, stress_tensor


def split_imaginary(yamlfiles: list[str]):
    """Check imaginary frequency in sscha_results.yaml files."""
    no_imag, imag = [], []
    for yfile in yamlfiles:
        yml = yaml.safe_load(open(yfile))
        if yml["status"]["imaginary"]:
            imag.append(yfile)
        else:
            no_imag.append(yfile)

    if len(no_imag) == 0:
        no_imag = None

    if len(imag) == 0:
        imag = None

    return no_imag, imag


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
    elements: Optional[bool] = None,
) -> DatasetDFT:
    """Return DFT dataset by loading electron.yaml files."""
    structures, properties = extract_electron_properties(
        yaml_data,
        temperature=temperature,
        target=target,
    )
    dft = DatasetDFT(
        structures,
        properties,
        forces=None,
        stresses=None,
        elements=elements,
    )
    return dft


def parse_property_yamls(yamlfiles: list[str]):
    """Parse polymlp_property.yaml files."""
    energies, forces, stresses, structures = [], [], [], []
    for yfile in yamlfiles:
        yml = yaml.safe_load(open(yfile))
        st = load_cell(yaml_data=yml, tag="structure")
        st.name = yfile
        structures.append(st)
        energies.append(yml["energy"])
        forces.append(np.array(yml["forces"]).T)
        s = yml["stress"]
        stress = np.array([[s[0], s[3], s[5]], [s[3], s[1], s[4]], [s[5], s[4], s[2]]])
        stresses.append(stress)
    return structures, np.array(energies), forces, np.array(stresses)
