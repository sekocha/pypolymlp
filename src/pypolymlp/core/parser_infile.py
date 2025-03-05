"""Class of input parser."""

import warnings
from typing import Any, Optional

import numpy as np

from pypolymlp.core.dataset import Dataset
from pypolymlp.core.utils import strtobool


class InputParser:
    """Class of input parser."""

    def __init__(self, fname: str, prefix: Optional[str] = None):
        """Init method."""
        self._prefix = prefix
        self._parse_infile(fname)

    def _parse_infile(self, fname: str):
        """Parse parameters from input file."""
        f = open(fname)
        lines = f.readlines()
        f.close()

        self._data = dict()
        self._distance = []
        self._train, self._test = [], []
        self._train_test_data = []
        self._md = []
        for line in lines:
            d = line.split()
            if len(d) > 1:
                self._data[d[0]] = d[1:]
                if "distance" == d[0]:
                    self._distance.append(d[1:])
                elif d[0] in ["train_data", "test_data", "data", "data_md"]:
                    dataset = Dataset(string_list=d[1:], prefix=self._prefix)
                    if d[0] == "train_data":
                        self._train.append(dataset)
                    elif d[0] == "test_data":
                        self._test.append(dataset)
                    elif d[0] == "data":
                        self._train_test_data.append(dataset)
                    elif d[0] == "data_md":
                        self._md.append(dataset)
        return self

    def get_params(
        self,
        tag: str,
        size: int = 1,
        default: Optional[Any] = None,
        required: bool = False,
        dtype: Any = str,
        return_array: bool = False,
    ):
        """Get parameters specified by tag."""
        if tag not in self._data:
            if required:
                raise KeyError("Tag", tag, "is not found.")
            return default

        params = list(self._data[tag])
        if len(params) != size:
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
            pair = tuple(sorted(data[:2], key=lambda x: elements.index(x)))
            distance_dict[pair] = [float(dis) for dis in data[2:]]
        return distance_dict

    @property
    def train(self):
        """Return training datasets."""
        return self._train

    @property
    def test(self):
        """Return test datasets."""
        return self._test

    @property
    def train_test(self):
        """Return datasets including training and test data."""
        return self._train_test_data

    @property
    def md(self):
        """Return MD datasets."""
        return self._md
