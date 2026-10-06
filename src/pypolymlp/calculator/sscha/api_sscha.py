"""Class for performing SSCHA."""

import copy
from typing import Optional, Union

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.sscha.sscha_core import SSCHACore
from pypolymlp.calculator.sscha.sscha_params import SSCHAParams
from pypolymlp.calculator.sscha.sscha_restart import Restart


def run_sscha(
    sscha_params: SSCHAParams,
    properties: Properties,
    verbose: bool = False,
):
    """Run sscha iterations for multiple temperatures.

    Parameters
    ----------
    sscha_params: Parameters for SSCHA in SSCHAParams.
    properties: Properties instance.
    """
    if sscha_params.use_temporal_cutoff:
        sscha = run_sscha_large_system(sscha_params, properties, verbose=verbose)
    else:
        sscha = run_sscha_standard(sscha_params, properties, verbose=verbose)
    return sscha


def run_sscha_standard(
    sscha_params: SSCHAParams,
    properties: Properties,
    verbose: bool = False,
):
    """Run sscha iterations for multiple temperatures.

    Parameters
    ----------
    sscha_params: Parameters for SSCHA in SSCHAParams.
    properties: Properties instance.
    """
    sscha = SSCHACore(sscha_params, properties, verbose=verbose)
    sscha.set_initial_force_constants(fc2=sscha_params.fc2)
    if verbose:
        freq = sscha.run_frequencies()
        print("Frequency (min):      ", np.round(np.min(freq), 5), flush=True)
        print("Frequency (max):      ", np.round(np.max(freq), 5), flush=True)

    if sscha_params.enable_precondition:
        sscha = _run_precondition(sscha, verbose=verbose)

    if verbose:
        print("Size of FC2 basis-set:", sscha.n_fc_basis, flush=True)
    sscha = _run_target_sscha(sscha, verbose=verbose)
    return sscha


def run_sscha_large_system(
    sscha_params: SSCHAParams,
    properties: Properties,
    verbose: bool = False,
):
    """Run sscha iterations for multiple temperatures using cutoff temporarily.

    Parameters
    ----------
    sscha_params: Parameters for SSCHA in SSCHAParams.
    properties: Properties instance.
    """
    sscha_params_target = copy.deepcopy(sscha_params)
    if sscha_params.cutoff_radius is None or sscha_params.cutoff_radius > 7.0:
        sscha_params.cutoff_radius = 6.0
        rerun = True
    else:
        rerun = False

    sscha = SSCHACore(sscha_params, properties, verbose=verbose)
    sscha.set_initial_force_constants(fc2=sscha_params.fc2)
    if verbose:
        freq = sscha.run_frequencies()
        print("Frequency (min):      ", np.round(np.min(freq), 5), flush=True)
        print("Frequency (max):      ", np.round(np.max(freq), 5), flush=True)

    if sscha_params.enable_precondition:
        sscha = _run_precondition(sscha, verbose=verbose)

    if rerun:
        if verbose:
            print("---", flush=True)
            print("Run SSCHA with temporal cutoff.", flush=True)
            print("Temporal cutoff radius:", sscha_params.cutoff_radius, flush=True)
            print("Size of FC2 basis-set: ", sscha.n_fc_basis, flush=True)
        sscha.run(temp=sscha_params.temperatures[0])
        fc2_rerun = sscha.force_constants
        sscha_params.cutoff_radius = sscha_params_target.cutoff_radius

        sscha = SSCHACore(sscha_params_target, properties, verbose=verbose)
        sscha.set_initial_force_constants(fc2=fc2_rerun)

    if verbose:
        print("Size of FC2 basis-set:", sscha.n_fc_basis, flush=True)

    sscha = _run_target_sscha(sscha, verbose=verbose)
    return sscha


def _run_precondition(sscha: SSCHACore, verbose: bool = False):
    """Run a procedure to perform precondition."""
    sscha_params = sscha.sscha_params
    if verbose:
        print("---", flush=True)
        print("Preconditioning.", flush=True)
        print("Size of FC2 basis-set:", sscha.n_fc_basis, flush=True)

    n_samples = max(min(sscha_params.n_samples_init // 50, 100), 5)
    n_iter, delta = 1, 1.0
    while delta > sscha_params.tol * 5 and n_iter < 20:
        if verbose:
            string = "###########"
            print(string, "Preconditioning Iteration:", n_iter, string, flush=True)

        sscha.precondition(
            temp=sscha_params.temperatures[0],
            n_samples=n_samples * n_iter,
            tol=sscha_params.tol,
            max_iter=5,
        )
        delta = sscha.delta
        n_iter += 1

    return sscha


def _run_target_sscha(sscha: SSCHACore, verbose: bool = False):
    """Run SSCHA for target temperatures."""
    for temp in sscha.sscha_params.temperatures:
        if verbose:
            print("************** Temperature:", temp, "**************", flush=True)
        sscha.run(temp=temp)
        # TODO: Include parameters in sschaCore
        sscha_params = sscha.sscha_params
        sscha.save_results(path=sscha_params.path, write_pdos=sscha_params.save_pdos)
    return sscha


def load_restart(
    yaml: str = "sscha_results.yaml",
    parse_fc2: bool = True,
    parse_mlp: bool = True,
    pot: Optional[Union[str, list, tuple, np.ndarray]] = None,
):
    """Parse sscha_results.yaml file.

    If parse_fc2 = True, fc2.hdf5 in the same directory
    as yaml file will be loaded.
    """
    if parse_fc2:
        fc2hdf5 = "/".join(yaml.split("/")[:-1]) + "/fc2.hdf5"
    else:
        fc2hdf5 = None

    res = Restart(yaml, fc2hdf5=fc2hdf5)
    unitcell = res.unitcell
    supercell_matrix = res.supercell_matrix
    if parse_mlp:
        pot = res.polymlp if pot is None else pot
        prop_static = Properties(pot=pot)
    else:
        prop_static = None
    fc2 = res.force_constants
    return (unitcell, supercell_matrix, prop_static, fc2)
