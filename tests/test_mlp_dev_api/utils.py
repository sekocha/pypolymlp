"""Utility functions for running regression."""

from pathlib import Path

from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

cwd = Path(__file__).parent


def _run_fit(files: str):
    """Run fitting using sequential procedure."""
    pypolymlp = Pypolymlp()
    pypolymlp.load_parameter_file(files, train_ratio=0.9, prefix=str(cwd))
    pypolymlp.parameters.solver_batch_size = 100
    pypolymlp.fit()
    pypolymlp.estimate_error(log_energy=False)
    return pypolymlp


def _run_fit_standard(files: str):
    """Run fitting using standard procedure."""
    pypolymlp = Pypolymlp()
    pypolymlp.load_parameter_file(files, train_ratio=0.9, prefix=str(cwd))
    pypolymlp.fit(use_full_x=True)
    pypolymlp.estimate_error(log_energy=False)
    return pypolymlp
