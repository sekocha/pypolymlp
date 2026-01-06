"""Utility functions for generating texts in sphinx format."""

import io
from typing import Optional

import numpy as np


def include_image(
    image: str,
    title: str = Optional[None],
    height: int = 300,
    file: Optional[io.IOBase] = None,
):
    """Generate txt to include image file."""
    if title is not None:
        print("**" + title + "**", file=file)
        print(file=file)

    print(".", end="", file=file)
    print(".", end="", file=file)
    print(" image::", image, file=file)
    print("   :height:", str(height) + "px", file=file)
    print(file=file)
    print(file=file)


def array_to_csv_table(
    data: np.ndarray,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    header: list = None,
    widths: list = None,
    stub_columns: Optional[int] = None,
    file: Optional[io.IOBase] = None,
):
    """Convert numpy data to csv_table."""
    if title is not None:
        print("**" + title + "**", file=file)
        print(file=file)

    print(".", end="", file=file)
    print(".", end="", file=file)
    print(" csv-table::", subtitle, file=file)
    print("    :header:", header, file=file)
    print("    :widths:", end=" ", file=file)
    for w in widths[:-1]:
        print(w, end=",", file=file)
    print(widths[-1], file=file)

    if stub_columns is not None:
        print("    :stub-columns:", stub_columns, file=file)
    print(file=file)

    for d in data:
        print("    ", end="", file=file)
        for d1 in d[:-1]:
            print(d1, end=",", file=file)
        print(d[-1], file=file)
    print(file=file)
    print(file=file)
