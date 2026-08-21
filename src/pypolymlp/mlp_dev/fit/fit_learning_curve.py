"""Class for calculating learning curve."""

from pypolymlp.core.dataset import DatasetList
from pypolymlp.core.params import PolymlpParams
from pypolymlp.mlp_dev.errors.api_errors import eval_rmse

from .fit_base import PolymlpFitBase
from .solvers_standard import solver_ridge


def save_learning_curve_log(
    error_log: dict,
    filename: str = "polymlp_learning_curve.dat",
):
    """Save results from learning curve calculations."""
    f = open(filename, "w")
    header = "# n_str, RMSE(energy, meV/atom) RMSE(force, eV/ang), RMSE(stress)"
    print(header, file=f)
    for n_samp, error in error_log:
        error_ev = error["energy"] * 1000
        print(n_samp, error_ev, error["force"], error["stress"], file=f)
    f.close()


def print_learning_curve_log(error_log: dict):
    """Generate output for results from learning curve calculations."""
    print("Learning Curve:", flush=True)
    for n_samples, error in error_log:
        print("- n_samples:   ", n_samples, flush=True)
        print("  rmse_energy: ", "{:.8f}".format(error["energy"] * 1000), flush=True)
        print("  rmse_force:  ", "{:.8f}".format(error["force"]), flush=True)
        print("  rmse_stress: ", error["stress"], flush=True)


class PolymlpFitLearningCurve(PolymlpFitBase):
    """Class for calculating learning curve."""

    def __init__(
        self,
        params: PolymlpParams,
        train: DatasetList,
        test: DatasetList,
        verbose: bool = False,
    ):
        """Init method.

        params: Parameters of polymlp.
        train: Training datasets.
        test: Test datasets.
        """
        if len(train) != 1:
            raise RuntimeError(
                "Number of training datasets must be one for learning curve."
            )
        if len(test) != 1:
            raise RuntimeError(
                "Number of test datasets must be one for learning curve."
            )

        super().__init__(params, train, verbose=verbose)

        self._test = test
        self._error_log = None

    def fit(self):
        """Estimate learning curve."""
        self._polymlp.check_memory_size_in_regression()

        train_xy = self._polymlp.calc_xy(self._train)
        test_xy = self._polymlp.calc_xy(
            self._test,
            scales=train_xy.scales,
            min_energy=train_xy.min_energy,
        )

        if self._verbose:
            print("Calculate learning curve.", flush=True)

        self._error_log = []
        n_train = train_xy.n_structures
        for n_samples in range(n_train // 10, n_train + 1, n_train // 10):
            if self._verbose:
                print(
                    "------------- n_samples:", n_samples, "-------------", flush=True
                )

            x, y = train_xy.slice(n_samples, self._train[0].total_n_atoms)
            coefs = solver_ridge(
                x=x,
                y=y,
                alphas=self._params.alphas,
                verbose=False,
            )
            rmse_train = self._polymlp.compute_rmse(coefs, x=x, y=y)
            rmse_test = self._polymlp.compute_rmse(coefs, test_xy)
            best_model = self._polymlp.get_best_model(
                coefs,
                train_xy.scales,
                rmse_train,
                rmse_test,
                train_xy.cumulative_n_features,
            )
            if self._verbose:
                self._polymlp.print_model_selection_log(rmse_train, rmse_test)

            error = eval_rmse(best_model, self._test, log_energy=False, tag="test")
            for val in error.values():
                self._error_log.append([n_samples, val])

        if self._verbose:
            print_learning_curve_log(self._error_log)
        return self

    @property
    def error_log(self):
        """Return error log."""
        return self._error_log
