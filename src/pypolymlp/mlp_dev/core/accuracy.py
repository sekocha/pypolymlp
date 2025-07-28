"""Class for computing prediction errors."""

import itertools
import os
from math import acos, degrees
from typing import Literal, Optional

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpDataDFT
from pypolymlp.core.utils import rmse
from pypolymlp.mlp_dev.core.dataclass import PolymlpDataMLP


class PolymlpEvalAccuracy:
    """Class for computing prediction errors."""

    def __init__(self, mlp: PolymlpDataMLP, verbose: bool = False):
        """Init method."""
        self._prop = Properties(params=mlp.params, coeffs=mlp.scaled_coeffs)
        self._verbose = verbose

    def compute_error(
        self,
        dft_all: list[PolymlpDataDFT],
        stress_unit: Literal["eV", "GPa"] = "eV",
        log_energy: bool = True,
        log_force: bool = False,
        log_stress: bool = False,
        path_output: str = "./",
        tag: str = "train",
    ):
        """Compute errors and predicted values for all datasets."""
        errors = dict()
        for dft in dft_all:
            output_key = self._generate_output_key(dft.name, tag=tag)
            errors[dft.name] = self.compute_error_single(
                dft,
                output_key=output_key,
                stress_unit=stress_unit,
                log_energy=log_energy,
                log_force=log_force,
                log_stress=log_stress,
                path_output=path_output,
            )
        return errors

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
        rmse_e, true_e, pred_e = self._compute_rmse(
            dft.energies,
            energies,
            normalize=n_total_atoms,
        )

        if not dft.exist_force:
            rmse_f = None
            rmse_percent_f_norm = None
            rmse_f_direction = None
        else:
            rmse_f, true_f, pred_f = self._compute_rmse(dft.forces, forces)
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
                rmse_f_direction = degrees(acos(rmse_f_direction))
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
            rmse_s, true_s, pred_s = self._compute_rmse(
                dft.stresses,
                stresses,
                normalize=normalize,
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
            if log_energy:
                self._write_energies(dft, true_e, pred_e, path_output, output_key)
            if log_force:
                self._write_forces(dft, true_f, pred_f, path_output, output_key)
            if log_stress:
                self._write_stresses(dft, true_s, pred_s, path_output, output_key)

        return error_dict

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
        return self

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

    def _write_energies(
        self,
        dft: PolymlpDataDFT,
        true_e: np.ndarray,
        pred_e: np.ndarray,
        path_output: str,
        output_key: str,
    ):
        """Write energy values of structures in a dataset."""
        outdata = np.array([true_e, pred_e, (true_e - pred_e) * 1000]).T
        f = open(path_output + "/predictions/energy." + output_key + ".dat", "w")
        print("# DFT(eV/atom), MLP(eV/atom), DFT-MLP(meV/atom)", file=f)
        for d, name in zip(outdata, dft.files):
            print(d[0], d[1], d[2], name, file=f)
        f.close()

    def _write_forces(
        self,
        dft: PolymlpDataDFT,
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
        dft: PolymlpDataDFT,
        true_s: np.ndarray,
        pred_s: np.ndarray,
        path_output: str,
        output_key: str,
    ):
        outdata = np.array([true_s, pred_s, (true_s - pred_s)]).T
        filename = path_output + "/predictions/stress." + output_key + ".dat"
        f = open(filename, "w")
        print("# DFT, MLP, DFT-MLP", file=f)
        for d in outdata:
            print(d[0], d[1], d[2], file=f)
        f.close()


def write_error_yaml(
    error: dict,
    filename: str = "polymlp_error.yaml",
    mode: str = "w",
):
    """Save errors in yaml format."""
    np.set_printoptions(legacy="1.21")
    f = open(filename, mode)
    if mode == "w":
        print("units:", file=f)
        print("  energy: meV/atom", file=f)
        print("  force:  eV/angstrom", file=f)
        print("  stress: meV/atom", file=f)
        print("", file=f)

    print("prediction_errors:", file=f)
    for key, dict1 in error.items():
        print("- dataset:", key, file=f)
        print("  rmse_energy: ", dict1["energy"] * 1000, file=f)
        if dict1["force"] is not None:
            print("  rmse_force:  ", dict1["force"], file=f)
        if dict1["stress"] is not None:
            print("  rmse_stress: ", dict1["stress"] * 1000, file=f)
        print("", file=f)
    f.close()
