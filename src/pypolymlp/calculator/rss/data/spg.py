#!/usr/bin/env python

from pypolymlp.utils.spglib_utils import SymCell


def __process(poscar):

    symprecs = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]

    sym = SymCell(poscar_name=poscar)
    spgs = sym.get_spacegroup_multiple_prec(symprecs=symprecs)
    spgs = [i for i in spgs if i is not None]
    spgs = tuple(sorted(set(spgs)))
    return spgs


def get_space_groups(poscars, parallel=True):

    if parallel:
        from joblib import Parallel, delayed

        spgs = Parallel(n_jobs=-1)(delayed(__process)(p) for p in poscars)
        return list(spgs)
    return [__process(p) for p in poscars]
