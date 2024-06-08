#!/usr/bin/env python
from distutils.util import strtobool

import numpy as np


class InputParser:

    def __init__(self, fname):

        f = open(fname)
        lines = f.readlines()
        f.close()

        self.train, self.test = [], []
        self.__data = dict()
        for line in lines:
            d = line.split()
            if len(d) > 1:
                self.__data[d[0]] = d[1:]
                if "train_data" == d[0]:
                    self.train.append(d[1:])
                elif "test_data" == d[0]:
                    self.test.append(d[1:])

    def get_params(
        self,
        tag,
        size=1,
        default=None,
        required=False,
        dtype=str,
        return_array=False,
    ):
        try:
            params = list(self.__data[tag])
        except KeyError:
            if required:
                raise KeyError(" Tag", tag, "is not found.")
            return default

        if size is not None:
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

        if size == 1 and return_array is False:
            return params[0]
        return params

    def get_sequence(self, tag, default=None):
        params = self.get_params(tag, size=3, default=default, dtype=str)
        return np.linspace(float(params[0]), float(params[1]), int(params[2]))

    def get_train(self):
        return self.train

    def get_test(self):
        return self.test
