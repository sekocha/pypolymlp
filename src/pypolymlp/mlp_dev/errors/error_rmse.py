"""Class for computing RMSE prediction errors."""

import itertools
import os
from math import acos, degrees
from typing import Literal

import numpy as np

from pypolymlp.core.dataset import Dataset, DatasetList
from pypolymlp.mlp_dev.core.dataclass import PolymlpDataMLP

from .error_base import PolymlpErrorBase


class PolymlpErrorRMSE(PolymlpErrorBase):
    """Class for computing RMSE prediction errors."""

    def __init__(self, mlp: PolymlpDataMLP, verbose: bool = False):
        """Init method."""
        super().__init__(mlp, verbose=verbose)

    def compute_error(
        self,
        datasets: DatasetList,
        stress_unit: Literal["eV", "GPa"] = "eV",
        log_energy: bool = True,
        log_force: bool = False,
        log_stress: bool = False,
        path_output: str = "./",
        tag: str = "train",
    ):
        """Compute errors and predicted values for all datasets."""
        self._errors = dict()
        for data in datasets:
            output_key = self._generate_output_key(data.name, tag=tag)
            self._errors[data.name] = self.compute_error_single(
                data,
                output_key=output_key,
                stress_unit=stress_unit,
                log_energy=log_energy,
                log_force=log_force,
                log_stress=log_stress,
                path_output=path_output,
            )
        return self._errors

    def compute_error_single(
        self,
        dataset: Dataset,
        output_key: str = "train",
        stress_unit: Literal["eV", "GPa"] = "eV",
        log_energy: bool = True,
        log_force: bool = False,
        log_stress: bool = False,
        path_output: bool = "./",
        force_direction: bool = False,
    ):
        """Compute errors and predicted values for single dataset."""
        strs = dataset.structures
        energies, forces, stresses = self._prop.eval_multiple(strs)
        forces = np.array(
            list(itertools.chain.from_iterable([f.T.reshape(-1) for f in forces]))
        )
        stresses = stresses.reshape(-1)

        n_total_atoms = [sum(st.n_atoms) for st in strs]
        rmse_e, true_e, pred_e = self._compute_rmse(
            dataset.energies,
            energies,
            normalize=n_total_atoms,
        )
        mae_e, _, _ = self._compute_mae(
            dataset.energies,
            energies,
            normalize=n_total_atoms,
        )

        if not dataset.exist_force:
            rmse_f = None
            mae_f = None
            rmse_percent_f_norm = None
            rmse_f_direction = None
        else:
            rmse_f, true_f, pred_f = self._compute_rmse(dataset.forces, forces)
            mae_f, _, _ = self._compute_mae(dataset.forces, forces)
            if force_direction:
                true_f1 = dataset.forces.reshape((-1, 3))
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

        if not dataset.exist_stress:
            rmse_s = None
            mae_s = None
        else:
            rmse_s, true_s, pred_s = self._compute_rmse(
                dataset.stresses,
                stresses,
                normalize=normalize,
            )
            mae_s, _, _ = self._compute_mae(
                dataset.stresses,
                stresses,
                normalize=normalize,
            )

        error_dict = {
            "energy": rmse_e,
            "force": rmse_f,
            "stress": rmse_s,
            "energy_mae": mae_e,
            "force_mae": mae_f,
            "stress_mae": mae_s,
            "percent_force_norm": rmse_percent_f_norm,
            "force_direction": rmse_f_direction,
        }
        if self._verbose:
            self.print_error(error_dict, key=output_key)

        if log_energy or log_force or log_stress:
            os.makedirs(path_output + "/predictions", exist_ok=True)
            if log_energy:
                self._write_energies(dataset, true_e, pred_e, path_output, output_key)
            if log_force:
                self._write_forces(true_f, pred_f, path_output, output_key)
            if log_stress:
                self._write_stresses(true_s, pred_s, path_output, output_key)

        return error_dict
