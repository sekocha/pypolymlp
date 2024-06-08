#!/usr/bin/env python
import numpy as np
import yaml


def get_frequency(yamlname):
    data = yaml.load(open(yamlname))
    freq = [b["frequency"] for d in data["phonon"] for b in d["band"]]
    weight = [d["weight"] for d in data["phonon"] for b in d["band"]]
    return np.array(freq), np.array(weight)


def get_band_frequency(yamlname):
    data = yaml.load(open(yamlname))
    dist = [d["distance"] for d in data["phonon"]]
    freq = [[b["frequency"] for b in d["band"]] for d in data["phonon"]]
    return dist, freq
