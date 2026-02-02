"""Class for performing SSCHA."""

import copy
from typing import Optional

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.sscha.sscha_core import SSCHACore
from pypolymlp.calculator.sscha.sscha_params import SSCHAParams
from pypolymlp.core.data_format import PolymlpParams

# (SSCHAParams, calculator, fc2) -> run_sscha


def _run_precondition(
    sscha: SSCHACore,
    sscha_params: SSCHAParams,
    verbose: bool = False,
):
    """Run a procedure to perform precondition."""

    if verbose:
        print("---", flush=True)
        print("Preconditioning.", flush=True)
        print("Size of FC2 basis-set:", sscha.n_fc_basis, flush=True)

    n_samples = max(min(sscha_params.n_samples_init // 50, 100), 5)
    sscha.precondition(
        t=sscha_params.temperatures[0],
        n_samples=n_samples,
        tol=0.01,
        max_iter=5,
    )
    if sscha_params.tol < 0.003:
        n_samples = max(min(sscha_params.n_samples_init // 10, 500), 10)
        sscha.precondition(
            t=sscha_params.temperatures[0],
            n_samples=n_samples,
            tol=0.005,
            max_iter=5,
        )
    return sscha


def _run_target_sscha(
    sscha: SSCHACore,
    sscha_params: SSCHAParams,
    verbose: bool = False,
):
    """Run SSCHA for target temperatures."""
    for temp in sscha_params.temperatures:
        if verbose:
            print("************** Temperature:", temp, "**************", flush=True)
            print("Increasing number of samples.", flush=True)
        sscha.run(t=temp, accurate=False)
        if verbose:
            print("Increasing number of samples.", flush=True)
        sscha.run(t=temp, accurate=True, initialize_history=False)
        sscha.save_results()
    return sscha


def run_sscha(
    sscha_params: SSCHAParams,
    pot: Optional[str] = None,
    params: Optional[PolymlpParams] = None,
    coeffs: Optional[np.ndarray] = None,
    properties: Optional[Properties] = None,
    fc2: Optional[np.ndarray] = None,
    precondition: bool = True,
    verbose: bool = False,
):
    """Run sscha iterations for multiple temperatures.

    Parameters
    ----------
    sscha_params: Parameters for SSCHA in SSCHAParams.
    pot: polymlp file.
    params: Parameters for polymlp.
    coeffs: Polymlp coefficients.
    properties: Properties instance.
    """
    sscha = SSCHACore(
        sscha_params,
        pot=pot,
        params=params,
        coeffs=coeffs,
        properties=properties,
        verbose=verbose,
    )
    sscha.set_initial_force_constants(fc2=fc2)
    freq = sscha.run_frequencies()
    if verbose:
        print("Frequency (min):      ", np.round(np.min(freq), 5), flush=True)
        print("Frequency (max):      ", np.round(np.max(freq), 5), flush=True)

    if precondition:
        sscha = _run_precondition(sscha, sscha_params, verbose=verbose)

    if verbose:
        print("Size of FC2 basis-set:", sscha.n_fc_basis, flush=True)
    sscha = _run_target_sscha(sscha, sscha_params, verbose=verbose)
    return sscha


def run_sscha_large_system(
    sscha_params: SSCHAParams,
    pot: Optional[str] = None,
    params: Optional[PolymlpParams] = None,
    coeffs: Optional[np.ndarray] = None,
    properties: Optional[Properties] = None,
    fc2: Optional[np.ndarray] = None,
    precondition: bool = True,
    verbose: bool = False,
):
    """Run sscha iterations for multiple temperatures using cutoff temporarily.

    Parameters
    ----------
    sscha_params: Parameters for SSCHA in SSCHAParams.
    pot: polymlp file.
    params: Parameters for polymlp.
    coeffs: Polymlp coefficients.
    properties: Properties instance.
    """
    sscha_params_target = copy.deepcopy(sscha_params)
    if sscha_params.cutoff_radius is None or sscha_params.cutoff_radius > 7.0:
        sscha_params.cutoff_radius = 6.0
        rerun = True
    else:
        rerun = False

    sscha = SSCHACore(
        sscha_params,
        pot=pot,
        params=params,
        coeffs=coeffs,
        properties=properties,
        verbose=verbose,
    )
    sscha.set_initial_force_constants(fc2=fc2)
    freq = sscha.run_frequencies()
    if verbose:
        print("Frequency (min):      ", np.round(np.min(freq), 5), flush=True)
        print("Frequency (max):      ", np.round(np.max(freq), 5), flush=True)

    if precondition:
        sscha = _run_precondition(sscha, sscha_params, verbose=verbose)

    if rerun:
        if verbose:
            print("---", flush=True)
            print("Run SSCHA with temporal cutoff.", flush=True)
            print("Temporal cutoff radius:", sscha_params.cutoff_radius, flush=True)
            print("Size of FC2 basis-set: ", sscha.n_fc_basis, flush=True)
        sscha.run(t=sscha_params.temperatures[0], accurate=False)
        fc2_rerun = sscha.force_constants
        sscha_params.cutoff_radius = sscha_params_target.cutoff_radius

        sscha = SSCHACore(
            sscha_params_target,
            pot=pot,
            params=params,
            coeffs=coeffs,
            properties=properties,
            verbose=verbose,
        )
        sscha.set_initial_force_constants(fc2=fc2_rerun)

    if verbose:
        print("Size of FC2 basis-set:", sscha.n_fc_basis, flush=True)

    sscha = _run_target_sscha(sscha, sscha_params, verbose=verbose)
    return sscha
