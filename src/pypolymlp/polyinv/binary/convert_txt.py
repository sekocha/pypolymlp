"""Generate C++ polymlp_gtinv_model.cpp files.

How to use
----------
from pypolymlp.polyinv.cxx_utils import convert_polyinv_yaml_c

convert_polyinv_yaml_c(
    order=2,
    headerfile: str = "polymlp_gtinv_data_ver2.h",
    classname="GtinvDataVer2",
    filename="polyinv_coeffs_2.yaml",
)
"""

import io

import yaml


def print_data1d(
    array: list,
    prefix: str = "",
    require_comma: bool = True,
    is_float=False,
    file: io.IOBase | None = None,
):
    """Print one-dimensional array."""
    if is_float:
        print(prefix, "{", sep="", file=file)
        for i, d in enumerate(array[:-1]):
            print(prefix, d, ",", sep="", file=file)
        print(prefix, array[-1], sep="", file=file)

        if require_comma:
            print(prefix, "},", sep="", file=file)
        else:
            print(prefix, "}", sep="", file=file)
    else:
        print(prefix, "{", sep="", end="", file=file)
        for i, d in enumerate(array[:-1]):
            print(d, end=",", file=file)
        print(array[-1], end="", file=file)

        if require_comma:
            print("},", file=file)
        else:
            print("}", file=file)


def print_data2d(data, tag: str, is_float: bool = False, file: io.IOBase | None = None):
    """Print 2D array."""
    for d in data[:-1]:
        print_data1d(d[tag], prefix="  ", is_float=is_float, file=file)
    print_data1d(
        data[-1][tag], prefix="  ", require_comma=False, is_float=is_float, file=file
    )


def print_data3d(data, tag: str, is_float: bool = False, file: io.IOBase | None = None):
    """Print 3D array."""
    for i, d in enumerate(data):
        data2d = d[tag]
        print(" ", "{", sep="", file=file)
        for d2 in data2d[:-1]:
            print_data1d(d2, prefix="  ", is_float=is_float, file=file)
        print_data1d(
            data2d[-1], prefix="  ", require_comma=False, is_float=is_float, file=file
        )
        if i == len(data) - 1:
            print(" ", "}", sep="", file=file)
        else:
            print(" ", "},", sep="", file=file)


def convert_polyinv_yaml_c(
    order: int = 3,
    headerfile: str = "polymlp_gtinv_data_v2.h",
    classname: str = "GtinvDataVer2",
    filename: str = "polyinv_coeffs.yaml",
):
    """Load yaml file for polynomial invariant enumeratation."""
    data = yaml.safe_load(open(filename))
    data = data["invariants"]
    lmax = max([max(d["l"]) for d in data])
    data_div = [[] for _ in range(lmax + 1)]
    for d in data:
        lmax = max(d["l"])
        data_div[lmax].append(d)

    for data in data_div:
        generate_cfiles(data, order=order, headerfile=headerfile, classname=classname)


def generate_cfiles(
    data: list,
    order: int = 3,
    headerfile: str = "polymlp_gtinv_data_v2.h",
    classname: str = "GtinvDataVer2",
):
    """Load yaml file for polynomial invariant enumeratation."""
    lmax = max(data[0]["l"])
    print(lmax)
    filename = "polymlp_gtinv_data_v2_order" + str(order) + "_l" + str(lmax) + ".cpp"
    with open(filename, "w") as f:
        print('#include "' + headerfile + '"', file=f)
        print(file=f)

        class_str = classname + "<" + str(order) + "," + str(lmax) + ">"
        print("template<>", file=f)
        print("const vector2i", class_str + "::L_ARRAY_ALL = {", file=f)
        print_data2d(data, "l", file=f)
        print("};", file=f)
        print(file=f)

        print("template<>", file=f)
        print("const vector2d", class_str + "::COEFFS_ALL = {", file=f)
        print_data2d(data, "coeffs", is_float=True, file=f)
        print("};", file=f)
        print(file=f)

        print("template<>", file=f)
        print("const vector3i", class_str + "::M_ARRAY_ALL = {", file=f)
        print_data3d(data, "m", file=f)
        print("};", file=f)
        print(file=f)

        print("template class", class_str + ";", file=f)
