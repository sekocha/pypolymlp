#!/usr/bin/env python
import numpy as np
from distutils.util import strtobool

class InputParser:

    def __init__(self, fname):

        f = open(fname)
        lines = f.readlines() 
        f.close()

        self.__data = dict()
        for line in lines:
            d = line.split()
            if len(d) > 1:
                if 'data_append' in d[0]: 
                    tag = d[0].replace('_append','')
                    if tag in self.__data:
                        self.__data[tag].append(d[1])
                        self.__data[tag+'_force'].append(d[2])
                        self.__data[tag+'_weight'].append(d[3])
                    else:
                        self.__data[tag] = [d[1]]
                        self.__data[tag+'_force'] = [d[2]]
                        self.__data[tag+'_weight'] = [d[3]]
                elif 'data' in d[0]: 
                    if d[0] in self.__data:
                        self.__data[d[0]].extend(d[1:])
                    else:
                        self.__data[d[0]] = d[1:]
                else:
                    self.__data[d[0]] = d[1:]

    def get_params(self, 
                   tag, 
                   size=1, 
                   default=None, 
                   dtype=str,
                   return_array=False):
        try:
            params = list(self.__data[tag])
        except:
            if default is None:
                raise KeyError(' Tag', name, 'is not found.')
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

        if size == 1 and return_array == False:
            return params[0]
        return params

    def get_sequence(self, tag, default=None):
        params = self.get_params(tag, size=3, default=default, dtype=str)
        return np.linspace(float(params[0]), float(params[1]), int(params[2]))

