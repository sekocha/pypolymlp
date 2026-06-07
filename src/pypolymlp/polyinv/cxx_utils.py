"""Generate C++ polymlp_gtinv_model.cpp files."""

# import io
# import sys
#
# import numpy as np
import yaml

# from numpy.typing import NDArray


def print_list_nospace(array: list, prefix: str, require_comma: bool = True):
    """Print one-dimensional array."""
    print(prefix + " {", end="", flush=True)
    for i, d in enumerate(array[:-1]):
        print(d, end=",", flush=True)
    print(array[-1], end="", flush=True)
    if require_comma:
        print("},", flush=True)
    else:
        print("}", flush=True)


def print_data_2d_int(data, tag: str):
    """Print 2D array with int type."""
    for d in data[:-1]:
        print_list_nospace(d[tag], prefix=" ")
    print_list_nospace(data[-1][tag], prefix=" ", require_comma=False)


def print_data_2d_float(data, tag: str):
    """Print 2D array with float type."""
    for d in data[:-1]:
        print_list_nospace(d[tag], prefix=" ")
    print_list_nospace(data[-1][tag], prefix=" ", require_comma=False)


def convert_polyinv_yaml_c(
    order: int = 3,
    headerfile: str = "polymlp_gtinv_data_ver2.h",
    classname: str = "GtinvDataVer2",
    filename: str = "polyinv_coeffs.yaml",
):
    """Load yaml file for polynomial invariant enumeratation."""
    data = yaml.safe_load(open(filename))
    data = data["invariants"]
    # l_array = [d["l"] for d in data]

    print('#include "' + headerfile + '"', flush=True)
    print(flush=True)

    class_str = classname + "<" + str(order) + ">"
    print("template<>", flush=True)
    print("const vector2i", class_str + ">::L_ARRAY_ALL = {", flush=True)
    print_data_2d_int(data, "l")
    # for d in data[:-1]:
    #     print_list_nospace(d['l'], prefix=" ")
    # print_list_nospace(data[-1]['l'], prefix=" ", require_comma=False)
    print("};", flush=True)
    print(flush=True)

    # [d['coeffs'] for d in data]
    print("template<>", flush=True)
    print("const vector2d", class_str + ">::COEFFS_ALL = {", flush=True)
    print_data_2d_float(data, "coeffs")
    print("};", flush=True)
    print(flush=True)

    # m_array = [d['m'] for d in data]
    print("template<>", flush=True)
    print("const vector3i", class_str + ">::M_ARRAY_ALL = {", flush=True)
    # TODO: Insert data.
    print("};", flush=True)
    print(flush=True)

    print("template class", class_str + ";", flush=True)
