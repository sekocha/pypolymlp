"""Utility functions for IO."""

import io
import os
from typing import Optional, Union

import numpy as np


def print_pot(
    pot: Union[str, list],
    tag: str = "polymlp",
    indent: int = 0,
    file: Optional[io.IOBase] = None,
):
    """Print potential file names."""
    header = " " * indent + tag + ":"
    prefix = " " * indent + "- "
    print(header, file=file)
    if isinstance(pot, (list, tuple, np.ndarray)):
        for p in pot:
            print(prefix, os.path.abspath(p), file=file)
    else:
        print(prefix, os.path.abspath(pot), file=file)
