"""Class of input parser."""

import warnings
from typing import Any, Optional

import numpy as np

from pypolymlp.core.utils import strtobool


class InputParser:
    """Class of input parser."""

    def __init__(self, filename: str):
        """Init method."""
        with open(filename) as f:
            lines = f.readlines()

        self._data = dict()
        self._distance, self._dataset_strings = [], []
        for line in lines:
            d = line.split()
            if len(d) > 1:
                if "distance" == d[0]:
                    self._distance.append(d[1:])
                elif "data" in d[0] and "dataset_type" not in d[0]:
                    self._dataset_strings.append(d)
                else:
                    self._data[d[0]] = d[1:]

    def get_params(
        self,
        tag: str,
        size: int = 1,
        default: Optional[Any] = None,
        required: bool = False,
        dtype: Any = str,
        return_array: bool = False,
        use_warnings: bool = True,
    ):
        """Get parameters specified by tag."""
        if tag not in self._data:
            if required:
                raise KeyError("Tag", tag, "is not found.")
            return default

        params = list(self._data[tag])
        if use_warnings and len(params) != size:
            sentence = "Length " + tag + " is not compatible with its required size."
            warnings.warn(sentence)

        params = params[:size]
        if dtype == bool:
            params = [strtobool(x) for x in params]
        elif dtype == int:
            params = [int(x) for x in params]
        elif dtype == float:
            params = [float(x) for x in params]
        elif dtype == str:
            params = [str(x) for x in params]
        else:
            params = np.array(params).astype(dtype)

        if size == 1 and return_array == False:
            return params[0]
        return params

    def get_sequence(self, tag: str, default: Optional[tuple] = None):
        """Return linspace sequence for tag."""
        params = self.get_params(tag, size=3, default=default, dtype=str)
        return np.linspace(float(params[0]), float(params[1]), int(params[2]))

    @property
    def distance(self):
        """Return distances activating atomic pairs."""
        elements = self._data["elements"]
        distance_dict = dict()
        for data in self._distance:
            pair = data[:2]
            for p in pair:
                if p not in elements:
                    raise RuntimeError(
                        "Elements in distance not found in potential model."
                    )
            pair = tuple(sorted(pair, key=lambda x: elements.index(x)))
            distance_dict[pair] = [float(dis) for dis in data[2:]]
        return distance_dict

    @property
    def dataset_strings(self):
        """Return dataset strings from file."""
        return self._dataset_strings
