"""Convert polynomial invariant yaml file to binary data file."""

import struct

import numpy as np
import yaml

MAGIC = b"DATA"

TYPE_2D_INT = 1
TYPE_2D_DOUBLE = 2
TYPE_3D_INT = 3


def write_int32(f, x):
    """Write integer."""
    f.write(struct.pack("<i", x))


def write_double_array(f, arr):
    """Write float."""
    arr = np.asarray(arr, dtype=np.float64)
    f.write(arr.astype("<f8").tobytes())


def save_2d_int(f, v):
    """Save 2D array with integer."""
    write_int32(f, TYPE_2D_INT)
    write_int32(f, len(v))

    for row in v:
        write_int32(f, len(row))
        arr = np.asarray(row, dtype=np.int32)
        f.write(arr.astype("<i4").tobytes())


def save_2d_double(f, v):
    """Save 2D array with float."""
    write_int32(f, TYPE_2D_DOUBLE)
    write_int32(f, len(v))

    for row in v:
        write_int32(f, len(row))
        arr = np.asarray(row, dtype=np.float64)
        f.write(arr.astype("<f8").tobytes())


def save_3d_int(f, v):
    """Save 3D array with integer."""
    write_int32(f, TYPE_3D_INT)
    write_int32(f, len(v))

    for mat in v:
        write_int32(f, len(mat))
        for row in mat:
            write_int32(f, len(row))
            arr = np.asarray(row, dtype=np.int32)
            f.write(arr.astype("<i4").tobytes())


def save_all(v_int2d: list, v_double2d: list, v_int3d: list, path: str):
    """Save all data to binary."""
    with open(path, "wb") as f:
        f.write(MAGIC)

        write_int32(f, 3)
        save_2d_int(f, v_int2d)
        save_2d_double(f, v_double2d)
        save_3d_int(f, v_int3d)


def convert_polyinv_yaml_binary(
    order: int = 3,
    filename: str = "polyinv_coeffs.yaml",
    version: int = 2,
):
    """Convert polyinv_coeff.yaml file into binary data file.

    How to use
    ----------
    convert_polyinv_yaml_binary(
        order=2,
        filename="polyinv_coeffs_2.yaml",
    )
    """
    data = yaml.safe_load(open(filename))
    data = data["invariants"]
    l_all = [d["l"] for d in data]
    m_all = [d["m"] for d in data]
    c_all = [d["coeffs"] for d in data]

    binary_file = "polymlp_gtinv_data_v" + str(version) + "_order" + str(order) + ".bin"
    save_all(l_all, c_all, m_all, binary_file)
