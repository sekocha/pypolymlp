"""Utility functions for KIM-API."""

import datetime
import os
import shutil
from typing import Optional

import numpy as np

from pypolymlp.core.io_polymlp import convert_to_yaml, is_legacy, load_mlp


def generate_kim_files(
    path: str,
    elements: list,
    polymlp_year: int = 2026,
    performance_level: int = 1,
    project_id: int = 0,
    project_version: int = 0,
    description: Optional[str] = None,
    model_driver: str = "Polymlp__MD_000000123456_000",
):
    """Generate potential model files required for KIM model."""
    system = "-".join(elements)
    project = (
        "Polymlp_Seko",
        str(polymlp_year) + "p" + str(performance_level),
        "".join(elements) + "__MO",
        str(project_id).zfill(12),
        str(project_version).zfill(3),
    )
    project = "_".join(project)

    with open(path + "polymlp.settings", "w") as f:
        print(" ".join(elements), file=f)
        print(file=f)

    np.set_printoptions(formatter={"str_kind": lambda x: f'"{x}"'})
    with open(path + "CMakeLists.txt", "w") as f:
        print("#", file=f)
        print("# Author: Atsuto Seko", file=f)
        print("#", file=f)
        print(file=f)
        print("cmake_minimum_required(VERSION 3.10)", file=f)
        print(file=f)
        print("list(APPEND CMAKE_PREFIX_PATH $ENV{KIM_API_CMAKE_PREFIX_DIR})", file=f)
        print("find_package(KIM-API-ITEMS 2.2 REQUIRED CONFIG)", file=f)
        print(file=f)
        print('kim_api_items_setup_before_project(ITEM_TYPE "portableModel")', file=f)
        print("project(" + project + ")", file=f)
        print('kim_api_items_setup_after_project(ITEM_TYPE "portableModel")', file=f)
        print(file=f)
        print("add_kim_api_model_library(", file=f)
        print("  NAME            ${PROJECT_NAME}", file=f)
        print('  DRIVER_NAME     "' + model_driver + '"', file=f)
        print('  PARAMETER_FILES "polymlp.settings"', file=f)
        print('                  "polymlp.yaml"', file=f)
        print("  )", file=f)

    title = (
        "Polynomial machine learning potential for",
        system,
        "developed by Seko (" + str(polymlp_year) + ")",
        "v" + str(project_version).zfill(3),
    )
    title = " ".join(title)

    with open(path + "kimspec.edn", "w") as f:
        print('{"domain" "openkim.org"', file=f)
        if description is not None:
            print(' "description" "' + description + '"', file=f)
        print(' "extended-id" "' + project + '"', file=f)
        print(' "kim-api-version" "2.2"', file=f)
        print(' "model-driver" "' + model_driver + '"', file=f)
        print(' "potential-type" "polymlp"', file=f)
        print(' "publication-year" "' + str(polymlp_year) + '"', file=f)
        print(' "species"', np.array(elements), file=f)
        print(' "title" "' + title + '"', file=f)
        print(" }", file=f)

    return project


def convert_polymlp_to_kim_model(
    polymlp_file: str,
    performance_level: int = 1,
    project_id: int = 0,
    project_version: int = 0,
    model_driver: str = "Polymlp__MD_000000123456_000",
):
    """Convert polymlp to KIM-API model."""
    tmp_path = "./Polymlp__MO_tmp/"
    polymlp_yaml = tmp_path + "polymlp.yaml"
    os.makedirs(tmp_path, exist_ok=True)

    if is_legacy(polymlp_file):
        convert_to_yaml(polymlp_file, yaml=polymlp_yaml)
    else:
        shutil.copy(polymlp_file, polymlp_yaml)

    params, _ = load_mlp(polymlp_yaml)
    elements = params.elements
    dt = datetime.datetime.now()
    polymlp_year = dt.year

    description = (
        "Polynomial machine learning potential (MLP) for",
        "-".join(elements),
        "system.",
    )
    description = " ".join(description)

    project = generate_kim_files(
        tmp_path,
        elements,
        polymlp_year=polymlp_year,
        performance_level=performance_level,
        project_id=project_id,
        project_version=project_version,
        description=description,
        model_driver=model_driver,
    )
    shutil.move(tmp_path, project)
