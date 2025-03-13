"""Class for calculating learning curves."""

import copy

import numpy as np

from pypolymlp.mlp_dev.core.accuracy import PolymlpDevAccuracy
from pypolymlp.mlp_dev.standard.mlpdev_dataxy import PolymlpDevDataXY
from pypolymlp.mlp_dev.standard.regression import Regression


class LearningCurve:
    """Class for calculating learning curves."""

    def __init__(
        self,
        polymlp: PolymlpDevDataXY,
        total_n_atoms: np.ndarray,
        verbose: bool = False,
    ):
        """Init method."""
        if len(polymlp.train) > 1:
            raise ValueError(
                "A single dataset is required for calculating learning curve"
            )
        self._polymlp = polymlp
        self._polymlp_copy = copy.deepcopy(polymlp)
        self._train_xy = polymlp.train_xy
        self._total_n_atoms = total_n_atoms
        self._verbose = verbose

        self._error_log = None

    def _find_slices(self, n_samples: int):
        """Return slices for selected data."""
        ids = list(range(n_samples))

        first_id = self._train_xy.first_indices[0][2]
        ids_stress = range(first_id, first_id + n_samples * 6)
        ids.extend(ids_stress)

        first_id = self._train_xy.first_indices[0][1]
        n_forces = sum(self._total_n_atoms[:n_samples]) * 3
        ids_force = range(first_id, first_id + n_forces)
        ids.extend(ids_force)
        return np.array(ids)

    def run(self):
        """Calculate learning curve."""
        self._error_log = []
        n_train = self._train_xy.first_indices[0][2]
        if self._verbose:
            print("Calculating learning curve...", flush=True)

        for n_samples in range(n_train // 10, n_train + 1, n_train // 10):
            if self._verbose:
                print(
                    "------------- n_samples:", n_samples, "-------------", flush=True
                )
            ids = self._find_slices(n_samples)

            self._polymlp_copy.train_xy.x = self._train_xy.x[ids]
            self._polymlp_copy.train_xy.y = self._train_xy.y[ids]
            self._polymlp_copy.train_xy.weight = self._train_xy.weight[ids]
            self._polymlp_copy.train_xy.scales = self._train_xy.scales

            reg = Regression(self._polymlp_copy).fit()
            acc = PolymlpDevAccuracy(reg)
            acc.compute_error()
            for error in acc.error_test_dict.values():
                self._error_log.append((n_samples, error))

        if self._verbose:
            self.print_log()

        return self

    def save_log(self, filename: str = "polymlp_learning_curve.dat"):
        """Save results from learning curve calculations."""
        f = open(filename, "w")
        print(
            "# n_str, RMSE(energy, meV/atom),",
            "RMSE(force, eV/ang), RMSE(stress)",
            file=f,
        )
        for n_samp, error in self._error_log:
            print(
                n_samp,
                error["energy"] * 1000,
                error["force"],
                error["stress"],
                file=f,
            )
        f.close()

    def print_log(self):
        """Generate output for results from learning curve calculations."""
        print("Learning Curve:", flush=True)
        for n_samples, error in self._error_log:
            print("- n_samples:   ", n_samples, flush=True)
            print(
                "  rmse_energy: ",
                "{:.8f}".format(error["energy"] * 1000),
                flush=True,
            )
            print("  rmse_force:  ", "{:.8f}".format(error["force"]), flush=True)
            print("  rmse_stress: ", error["stress"], flush=True)

    @property
    def error(self):
        """Return error for learning curve."""
        return self._error_log
