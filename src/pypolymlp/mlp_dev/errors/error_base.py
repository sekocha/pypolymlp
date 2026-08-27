"""Base class for evaluating errors."""

import itertools
from abc import ABC, abstractmethod
from typing import Literal, Optional

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.dataset import Dataset
from pypolymlp.core.utils import rmse
from pypolymlp.mlp_dev.core.dataclass import PolymlpDataMLP


class PolymlpErrorBase(ABC):
    """Base class for evaluating errors."""

    def __init__(self, mlp: PolymlpDataMLP, verbose: bool = False):
        """Init method."""
        self._mlp = mlp
        self._prop = Properties(params=mlp.params, coeffs=mlp.scaled_coeffs)
        self._verbose = verbose

        self._errors = None

    @abstractmethod
    def compute_error(self):
        """Compute errors and predicted values for all datasets."""
        pass

    @abstractmethod
    def compute_error_single(self):
        """Compute errors and predicted values for single dataset."""
        pass

    @property
    def properties(self):
        """Return properties class."""
        return self._prop

    @property
    def errors(self):
        """Return errors in dict of dict.

        Return
        ------
        errors: Predition errors for multiple datasets in dict format.
                For each dataset, the following entries are mandartory.
                "energy": RMSE for energy (eV/atom).
                "force":  RMSE for force (eV/angstrom).
                "stress": RMSE for stress (eV/atom).
                "energy_mae": MAE for energy (eV/atom).
                "force_mae":  MAE for force (eV/angstrom).
                "stress_mae": MAE for stress (eV/atom).
        """
        return self._errors

    def print_error(self, error: dict, key: str = "train"):
        """Print prediction errors."""
        print("prediction:", key, flush=True)

        energy = "{0:13.5f}".format(error["energy"] * 1000)
        print("  rmse_energy:", energy, "(meV/atom)", flush=True)

        if error["force"] is not None:
            force = "{0:13.5f}".format(error["force"])
            print("  rmse_force: ", force, "(eV/ang)", flush=True)

        if error["stress"] is not None:
            stress = "{0:13.5f}".format(error["stress"] * 1000)
            print("  rmse_stress:", stress, "(meV/atom)", flush=True)

        energy = "{0:13.5f}".format(error["energy_mae"] * 1000)
        print("  mae_energy: ", energy, "(meV/atom)", flush=True)

        if error["force_mae"] is not None:
            force = "{0:13.5f}".format(error["force_mae"])
            print("  mae_force:  ", force, "(eV/ang)", flush=True)

        if error["stress_mae"] is not None:
            stress = "{0:13.5f}".format(error["stress_mae"] * 1000)
            print("  mae_stress: ", stress, "(meV/atom)", flush=True)
        return self

    def write_error_yaml(
        self,
        filename: str = "polymlp_error.yaml",
        mode: str = "w",
    ):
        """Save errors in yaml format."""
        if self._errors is None:
            raise RuntimeError("Error dict not found.")

        np.set_printoptions(legacy="1.21")
        f = open(filename, mode)
        if mode == "w":
            print("units:", file=f)
            print("  energy: meV/atom", file=f)
            print("  force:  eV/angstrom", file=f)
            print("  stress: meV/atom", file=f)
            print(file=f)
            print("prediction_errors:", file=f)

        for key, dict1 in self._errors.items():
            print("- dataset:", key, file=f)
            print("  rmse_energy: ", dict1["energy"] * 1000, file=f)
            if dict1["force"] is not None:
                print("  rmse_force:  ", dict1["force"], file=f)
            if dict1["stress"] is not None:
                print("  rmse_stress: ", dict1["stress"] * 1000, file=f)
            print(file=f)

            print("  mae_energy:  ", dict1["energy_mae"] * 1000, file=f)
            if dict1["force_mae"] is not None:
                print("  mae_force:   ", dict1["force_mae"], file=f)
            if dict1["stress_mae"] is not None:
                print("  mae_stress:  ", dict1["stress_mae"] * 1000, file=f)
            print(file=f)
        f.close()

    def _write_energies(
        self,
        dataset: Dataset,
        true_e: np.ndarray,
        pred_e: np.ndarray,
        path_output: str,
        output_key: str,
    ):
        """Write energy values of structures in a dataset."""
        outdata = np.array([true_e, pred_e, (true_e - pred_e) * 1000]).T
        f = open(path_output + "/predictions/energy." + output_key + ".dat", "w")
        print("# DFT(eV/atom), MLP(eV/atom), DFT-MLP(meV/atom)", file=f)
        if dataset.files is None:
            for d in outdata:
                print(d[0], d[1], d[2], file=f)
        else:
            for d, name in zip(outdata, dataset.files):
                print(d[0], d[1], d[2], name, file=f)
        f.close()

    def _write_forces(
        self,
        true_f: np.ndarray,
        pred_f: np.ndarray,
        path_output: str,
        output_key: str,
    ):
        """Write force values of structures in a dataset."""
        outdata = np.array([true_f, pred_f, (true_f - pred_f)]).T
        filename = path_output + "/predictions/force." + output_key + ".dat"
        f = open(filename, "w")
        print("# DFT, MLP, DFT-MLP", file=f)
        for d in outdata:
            print(d[0], d[1], d[2], file=f)
        f.close()

    def _write_stresses(
        self,
        true_s: np.ndarray,
        pred_s: np.ndarray,
        path_output: str,
        output_key: str,
    ):
        """Write stress tensor components of structures in a dataset."""
        outdata = np.array([true_s, pred_s, (true_s - pred_s)]).T
        filename = path_output + "/predictions/stress." + output_key + ".dat"
        f = open(filename, "w")
        print("# DFT, MLP, DFT-MLP", file=f)
        for d in outdata:
            print(d[0], d[1], d[2], file=f)
        f.close()

    def _compute_rmse(
        self,
        true_values: np.ndarray,
        pred_values: np.ndarray,
        normalize: Optional[np.ndarray] = None,
    ):
        """Compute RMSE."""
        if normalize is None:
            true = true_values
            pred = pred_values
        else:
            true = true_values / np.array(normalize)
            pred = pred_values / np.array(normalize)

        return (rmse(true, pred), true, pred)

    def _compute_mae(
        self,
        true_values: np.ndarray,
        pred_values: np.ndarray,
        normalize: Optional[np.ndarray] = None,
    ):
        """Compute MAE."""
        if normalize is None:
            true = true_values
            pred = pred_values
        else:
            true = true_values / np.array(normalize)
            pred = pred_values / np.array(normalize)

        mae = np.mean(np.abs(true - pred))
        return (mae, true, pred)

    def _generate_output_key(
        self,
        dataset_name: str,
        tag: str = Literal["train", "test"],
    ):
        """Generate key used for identify datasets."""
        output_key = dataset_name.replace("*", "-").replace("." + "./", "")
        output_key = output_key.replace(".", "-").replace("/", "-")
        output_key = tag + "-" + output_key
        output_key = output_key.replace("---", "-").replace("--", "-")
        return output_key

    def _stress_normalize_coeffs(
        self,
        structures: list,
        stress_unit: Literal["eV", "GPa"],
    ):
        """Set normalize coefficients for stress entries."""
        if stress_unit == "eV":
            n_total_atoms = [sum(st.n_atoms) for st in structures]
            normalize = np.repeat(n_total_atoms, 6)
        elif stress_unit == "GPa":
            eV_to_GPa = 160.21766208
            volumes = [st.volume for st in structures]
            normalize = np.repeat(volumes, 6) / eV_to_GPa
        return normalize

    def _eval_properties(self, structures: list):
        """Evaluate and flatten properties."""
        pred_e, pred_f, pred_s = self._prop.eval_multiple(structures)
        pred_f = np.array(
            list(itertools.chain.from_iterable([f.T.reshape(-1) for f in pred_f]))
        )
        pred_s = pred_s.reshape(-1)
        return (pred_e, pred_f, pred_s)
