"""Check binary data."""

import struct


def read_int32(f):
    """Read integer data."""
    return struct.unpack("<i", f.read(4))[0]


def read_double(f):
    """Read float data."""
    return struct.unpack("<d", f.read(8))[0]


def read_2d_int(f):
    """Read integer-type 2D array."""
    n = read_int32(f)
    v = []
    for _ in range(n):
        m = read_int32(f)
        row = [read_int32(f) for _ in range(m)]
        v.append(row)

    return v


def read_2d_double(f):
    """Read float-type 2D array."""
    n = read_int32(f)
    v = []

    for _ in range(n):
        m = read_int32(f)
        row = [read_double(f) for _ in range(m)]
        v.append(row)

    return v


def read_3d_int(f):
    """Read integer-type 3D array."""
    a = read_int32(f)
    v = []

    for _ in range(a):
        b = read_int32(f)
        mat = []

        for _ in range(b):
            c = read_int32(f)
            row = [read_int32(f) for _ in range(c)]
            mat.append(row)

        v.append(mat)

    return v


def load_all(path):
    """Load binary data file.

    How to use
    ----------
    l, c, m = load_all("polymlp_gtinv_data_v2_order3.bin")

    for l1, c1, m1 in zip(l, c, m):
        print(l1)
        print(c1)
        print(m1)
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"DATA":
            raise ValueError("Incorrect format.")

        blocks = read_int32(f)
        assert blocks == 3

        t_data = read_int32(f)
        assert t_data == 1
        v1 = read_2d_int(f)

        t_data = read_int32(f)
        assert t_data == 2
        v2 = read_2d_double(f)

        t_data = read_int32(f)
        assert t_data == 3
        v3 = read_3d_int(f)

        return v1, v2, v3
