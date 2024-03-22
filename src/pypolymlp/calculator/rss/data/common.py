#!/usr/bin/env python 
import numpy as np
import sys, re, os

import yaml

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def readfile(filename):
    f = open(filename)
    lines = f.readlines()
    f.close()
    return lines

def parse_log_summary_yaml(fname, return_numbers=False):

    data = yaml.safe_load(open(fname))
    try:
        numbers = data['numbers']
    except:
        numbers = None

    summary_dict = dict()
    for d in data['nonequiv_structures']:
        comp = tuple(d['composition'])
        summary_dict[comp] = dict()
        ids = [d2['id'] for d2 in d['structures']]
        summary_dict[comp]['poscars'] = np.array(ids)
        energy = [d2['e'] for d2 in d['structures']]
        summary_dict[comp]['energies'] = np.array(energy)

        if 'spg' in d['structures'][0]:
            spgs = [d2['spg'] for d2 in d['structures']]
            summary_dict[comp]['space_groups'] = spgs
        else:
            summary_dict[comp]['space_groups'] = None

        if 'prob' in d['structures'][0]:
            prob = [d2['prob'] for d2 in d['structures']]
            summary_dict[comp]['probabilities'] = np.array(prob)

    if return_numbers:
        return summary_dict, numbers
    return summary_dict

def estimate_n_locals(n_trials, n_locals):
    denom = n_trials - n_locals - 2
    if denom > 0:
        return n_locals * (n_trials - 1) / denom
    return None


