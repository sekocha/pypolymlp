"""Class for computing prediction errors."""

import itertools
import math
import os
from typing import Literal, Optional

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpParams
from pypolymlp.core.utils import rmse
from pypolymlp.mlp_dev.core.regression_base import RegressionBase


class PolymlpDevAccuracy:
    """Class for computing prediction errors."""

    def __init__(self, reg: RegressionBase, verbose: bool = False):
        """Init method."""

        if reg.is_hybrid:
            coeffs_rescale = [c / s for c, s in zip(reg.coeffs, reg.scales)]
        else:
            coeffs_rescale = reg.coeffs / reg.scales

        self._prop = Properties(params=reg.params, coeffs=coeffs_rescale)

        self._params = reg.params
        self._train = reg.train
        self._test = reg.test
        self._multiple_datasets = reg.is_multiple_datasets
        self._hybrid = reg.is_hybrid
        self._error_train = dict()
        self._error_test = dict()
        self._verbose = verbose

    def print_error(self, error: dict, key: str = "train"):
        """Print prediction errors."""
        print("prediction:", key, flush=True)
        print(
            "  rmse_energy:",
            "{0:13.5f}".format(error["energy"] * 1000),
            "(meV/atom)",
            flush=True,
        )
        if error["force"] is not None:
            print(
                "  rmse_force: ",
                "{0:13.5f}".format(error["force"]),
                "(eV/ang)",
                flush=True,
            )
        if error["stress"] is not None:
            print(
                "  rmse_stress:",
                "{0:13.5f}".format(error["stress"] * 1000),
                "(meV/atom)",
                flush=True,
            )
        return self

    def write_error_yaml(self, filename: str = "polymlp_error.yaml"):
        """Save errors in yaml format."""
        np.set_printoptions(legacy="1.21")
        self._write_error_yaml(self._error_train, tag="train", filename=filename)
        self._write_error_yaml(
            self._error_test,
            tag="test",
            filename=filename,
            initialize=False,
        )
        return self

    def _write_error_yaml(
        self,
        error,
        tag="train",
        filename="polymlp_error.yaml",
        initialize=True,
    ):
        """Save errors in yaml format."""
        if initialize:
            f = open(filename, "w")
            print("units:", file=f)
            print("  energy: meV/atom", file=f)
            print("  force:  eV/angstrom", file=f)
            print("  stress: meV/atom", file=f)
            print("", file=f)
            f.close()

        f = open(filename, "a")
        print("prediction_errors_" + tag + ":", file=f)
        for key, dict1 in error.items():
            print("- dataset:", key, file=f)
            print("  rmse_energy: ", dict1["energy"] * 1000, file=f)
            if dict1["force"] is not None:
                print("  rmse_force:  ", dict1["force"], file=f)
            if dict1["stress"] is not None:
                print("  rmse_stress: ", dict1["stress"] * 1000, file=f)
            print("", file=f)
        f.close()
        return self

    def _compute_rmse(
        self,
        true_values: np.ndarray,
        pred_values: np.ndarray,
        normalize: Optional[np.ndarray] = None,
        return_values: bool = False,
    ):
        """Compute RMSE."""

        if normalize is None:
            true = true_values
            pred = pred_values
        else:
            true = true_values / np.array(normalize)
            pred = pred_values / np.array(normalize)

        if return_values:
            return rmse(true, pred), true, pred
        return rmse(true, pred)

    def _generate_output_key(
        self,
        dataset_name: str,
        tag: str = Literal["train", "test"],
    ):
        """Generate key used for identify datasets."""
        output_key = dataset_name.replace("*", "-").replace("..", "")
        output_key = output_key.replace(".", "-").replace("/", "-")
        output_key = tag + "-" + output_key
        output_key = output_key.replace("---", "-").replace("--", "-")
        return output_key

    def compute_error(
        self,
        stress_unit: Literal["eV", "GPa"] = "eV",
        log_energy: bool = True,
        log_force: bool = False,
        log_stress: bool = False,
        path_output: str = "./",
    ):
        """Compute errors and predicted values for all datasets."""
        if self._multiple_datasets:
            for dft in self._train:
                output_key = self._generate_output_key(dft.name, tag="train")
                self._error_train[dft.name] = self.compute_error_single(
                    dft,
                    output_key=output_key,
                    path_output=path_output,
                    log_energy=log_energy,
                    log_force=log_force,
                    log_stress=log_stress,
                )
            for dft in self._test:
                output_key = self._generate_output_key(dft.name, tag="test")
                self._error_test[dft.name] = self.compute_error_single(
                    dft,
                    output_key=output_key,
                    path_output=path_output,
                    log_energy=log_energy,
                    log_force=log_force,
                    log_stress=log_stress,
                )
        else:
            self._error_train[self._train.name] = self.compute_error_single(
                self._train,
                output_key="train",
                path_output=path_output,
                log_energy=log_energy,
                log_force=log_force,
                log_stress=log_stress,
            )
            self._error_test[self._test.name] = self.compute_error_single(
                self._test,
                output_key="test",
                path_output=path_output,
                log_energy=log_energy,
                log_force=log_force,
                log_stress=log_stress,
            )

    def compute_error_single(
        self,
        dft: PolymlpDataDFT,
        output_key: str = "train",
        stress_unit: Literal["eV", "GPa"] = "eV",
        log_energy: bool = True,
        log_force: bool = False,
        log_stress: bool = False,
        path_output: bool = "./",
        force_direction: bool = False,
    ):
        """Compute errors and predicted values for single dataset."""
        strs = dft.structures
        energies, forces, stresses = self._prop.eval_multiple(strs)
        forces = np.array(
            list(itertools.chain.from_iterable([f.T.reshape(-1) for f in forces]))
        )
        stresses = stresses.reshape(-1)

        n_total_atoms = [sum(st.n_atoms) for st in strs]
        if log_energy == False:
            rmse_e = self._compute_rmse(
                dft.energies,
                energies,
                normalize=n_total_atoms,
            )
        else:
            rmse_e, true_e, pred_e = self._compute_rmse(
                dft.energies,
                energies,
                normalize=n_total_atoms,
                return_values=True,
            )

        if not dft.exist_force:
            rmse_f = None
            rmse_percent_f_norm = None
            rmse_f_direction = None
        else:
            if log_force == False:
                rmse_f = self._compute_rmse(dft.forces, forces)
            else:
                rmse_f, true_f, pred_f = self._compute_rmse(
                    dft.forces, forces, return_values=True
                )

            if force_direction:
                true_f1 = dft.forces.reshape((-1, 3))
                pred_f1 = forces.reshape((-1, 3))
                norm_t = np.linalg.norm(true_f1, axis=1)
                norm_p = np.linalg.norm(pred_f1, axis=1)

                direction_t = true_f1 / norm_t[:, None]
                direction_p = pred_f1 / norm_p[:, None]
                cosine = [dt @ dp for dt, dp in zip(direction_t, direction_p)]

                rmse_percent_f_norm = np.average(np.abs((norm_p - norm_t) / norm_t))
                rmse_f_direction = np.average(np.abs(cosine))
                rmse_f_direction = math.degrees(math.acos(rmse_f_direction))
            else:
                rmse_f_direction = None
                rmse_percent_f_norm = None

        if stress_unit == "eV":
            normalize = np.repeat(n_total_atoms, 6)
        elif stress_unit == "GPa":
            eV_to_GPa = 160.21766208
            volumes = [st.volume for st in strs]
            normalize = np.repeat(volumes, 6) / eV_to_GPa

        if not dft.exist_stress:
            rmse_s = None
        else:
            if log_stress == False:
                rmse_s = self._compute_rmse(dft.stresses, stresses, normalize=normalize)
            else:
                rmse_s, true_s, pred_s = self._compute_rmse(
                    dft.stresses,
                    stresses,
                    normalize=normalize,
                    return_values=True,
                )

        error_dict = {
            "energy": rmse_e,
            "force": rmse_f,
            "stress": rmse_s,
            "percent_force_norm": rmse_percent_f_norm,
            "force_direction": rmse_f_direction,
        }
        if self._verbose:
            self.print_error(error_dict, key=output_key)

        if log_energy or log_force or log_stress:
            os.makedirs(path_output + "/predictions", exist_ok=True)

            outdata = np.array([true_e, pred_e, (true_e - pred_e) * 1000]).T
            f = open(path_output + "/predictions/energy." + output_key + ".dat", "w")
            print("# DFT(eV/atom), MLP(eV/atom), DFT-MLP(meV/atom)", file=f)
            for d, name in zip(outdata, dft.files):
                print(d[0], d[1], d[2], name, file=f)
            f.close()

            if log_force:
                outdata = np.array([true_f, pred_f, (true_f - pred_f)]).T
                f = open(
                    path_output + "/predictions/force." + output_key + ".dat",
                    "w",
                )
                print("# DFT, MLP, DFT-MLP", file=f)
                for d in outdata:
                    print(d[0], d[1], d[2], file=f)
                f.close()

            if log_stress:
                outdata = np.array([true_s, pred_s, (true_s - pred_s)]).T
                f = open(
                    path_output + "/predictions/stress." + output_key + ".dat",
                    "w",
                )
                print("# DFT, MLP, DFT-MLP", file=f)
                for d in outdata:
                    print(d[0], d[1], d[2], file=f)
                f.close()

        return error_dict

    @property
    def params(self) -> PolymlpParams:
        """Return polymlp parameters."""
        return self._params

    @property
    def is_multiple_datasets(self) -> bool:
        """Return whether multiple datasets are used or not."""
        return self._multiple_datasets

    @property
    def is_hybrid(self) -> bool:
        """Return whether hybrid model is used or not."""
        return self._hybrid

    @property
    def error_train_dict(self) -> dict:
        """Return errors for training datasets."""
        return self._error_train

    @property
    def error_test_dict(self) -> dict:
        """Return errors for test datasets."""
        return self._error_test
