"""Class of input parser."""

import warnings
from typing import Any, Optional

import numpy as np
from setuptools._distutils.util import strtobool


class InputParser:
    """Class of input parser."""

    def __init__(self, fname: str):

        self._parse_infile(fname)

    def _parse_infile(self, fname: str):

        f = open(fname)
        lines = f.readlines()
        f.close()

        self._train, self._test = [], []
        self._distance = []
        self._data = dict()
        for line in lines:
            d = line.split()
            if len(d) > 1:
                self._data[d[0]] = d[1:]
                if "distance" == d[0]:
                    self._distance.append(d[1:])
                elif "train_data" == d[0]:
                    self._train.append(d[1:])
                elif "test_data" == d[0]:
                    self._test.append(d[1:])

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

    def get_sequence(self, tag, default=None):
        params = self.get_params(tag, size=3, default=default, dtype=str)
        return np.linspace(float(params[0]), float(params[1]), int(params[2]))

    @property
    def distance(self):
        elements = self._data["elements"]
        distance_dict = dict()
        for data in self._distance:
            pair = tuple(sorted(data[:2], key=lambda x: elements.index(x)))
            distance_dict[pair] = [float(dis) for dis in data[2:]]
        return distance_dict

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test

    def get_train(self):
        return self._train

    def get_test(self):
        return self._test

    # # deprecated
    # def _get_gaussian_params(self, tag, gauss, default):

    #     if tag in self._data:
    #         params = self._data[tag]
    #     else:
    #         params = default
    #     if len(params) != 3:
    #         sentence = "Length " + tag + " is not compatible with its required size."
    #         warnings.warn(sentence)
    #     params = np.linspace(float(params[0]), float(params[1]), int(params[2]))

    #     bool_conditional = False if len(gauss) == 0 else True
    #     elements = self._data["elements"]
    #     conditional_params = dict()
    #     for g in gauss:
    #         pair = tuple(sorted([g[3], g[4]], key=lambda x: elements.index(x)))
    #         conditional_params[pair] = np.linspace(float(g[0]), float(g[1]), int(g[2]))
    #         if len(g) != 5:
    #             sentence = (
    #                 "Length gaussian_params (conditional) is not compatible "
    #                 "with its required size."
    #             )
    #             warnings.warn(sentence)

    #     element_pairs = itertools.combinations_with_replacement(elements, 2)
    #     for pair in element_pairs:
    #         pair = tuple(sorted(pair, key=lambda x: elements.index(x)))
    #         if pair not in conditional_params:
    #             conditional_params[pair] = params

    #     return params, conditional_params, bool_conditional

    # # deprecated
    # def get_gaussian_params(self, tag: str, default: Optional[Any] = None):
    #     """Get Gaussian parameters specified by tag."""
    #     if tag == "gaussian_params1":
    #         gauss = self._gauss1
    #     elif tag == "gaussian_params2":
    #         gauss = self._gauss2
    #     return self._get_gaussian_params(tag, gauss, default)
