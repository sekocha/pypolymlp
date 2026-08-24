"""Class for computing LOOCV prediction errors."""

import os
from typing import Literal

import numpy as np

from pypolymlp.core.dataset import Dataset, DatasetList
from pypolymlp.mlp_dev.core.data_sequential import compute_features_single_batch
from pypolymlp.mlp_dev.core.data_utils import PolymlpDataXY
from pypolymlp.mlp_dev.core.dataclass import PolymlpDataMLP
from pypolymlp.mlp_dev.core.utils_sequential import get_auto_batch_size, get_batch_slice

from .error_base import PolymlpErrorBase


class PolymlpErrorLOOCV(PolymlpErrorBase):
    """Class for computing LOOCV prediction errors."""

    def __init__(self, mlp: PolymlpDataMLP, verbose: bool = False):
        """Init method."""
        super().__init__(mlp, verbose=verbose)

    def compute_error(
        self,
        datasets: DatasetList,
        data_xy: PolymlpDataXY,
        stress_unit: Literal["eV", "GPa"] = "eV",
        log_energy: bool = True,
        log_force: bool = False,
        log_stress: bool = False,
        path_output: bool = "./",
        tag: str = "train",
        batch_size: int = 20,
    ):
        """Compute cross-validation errors and predicted values for all datasets."""
        if data_xy.inv_xtx is None:
            raise RuntimeError("Inverse matrix of X.T @ X not found.")

        inv_xtx = data_xy.inv_xtx
        if batch_size is None:
            n_features = inv_xtx.shape[0]
            batch_size = get_auto_batch_size(n_features, verbose=self._verbose)

        self._errors = dict()
        for data in datasets:
            output_key = self._generate_output_key(data.name, tag=tag)
            self._errors[f"LOOCV:{data.name}"] = self.compute_error_single(
                data,
                inv_xtx,
                batch_size,
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
        inv_xtx: np.ndarray,
        batch_size: int,
        output_key: str = "train",
        stress_unit: Literal["eV", "GPa"] = "eV",
        log_energy: bool = True,
        log_force: bool = False,
        log_stress: bool = False,
        path_output: bool = "./",
    ):
        """Compute cross-validation errors and predicted values for single dataset."""
        # TODO: Needed ?
        # dataset.sort_dft()
        n_str = len(dataset.structures)
        begin_ids, end_ids = get_batch_slice(n_str, batch_size)

        if not dataset.exist_force:
            rmse_f = None
            mae_f = None
        if not dataset.exist_stress:
            rmse_s = None
            mae_s = None

        e_errors = []
        f_errors = []
        s_errors = []
        true_e = []
        true_f = []
        true_s = []
        for begin, end in zip(begin_ids, end_ids):
            sliced_data = dataset.slice_dft(begin, end)
            x, _, x_w2, _, first_indices = compute_features_single_batch(
                self._mlp.params,
                sliced_data,
                verbose=False,
            )
            x = x / self._mlp.scales
            x_w2 = x_w2 / self._mlp.scales
            hat_h = x @ inv_xtx @ x_w2.T
            hat_h_diag = -np.diagonal(hat_h)
            hat_h_diag += 1

            y_pred = x @ self._mlp.coeffs
            ebegin, fbegin, sbegin = first_indices

            strs = sliced_data.structures
            n_total_atoms = [sum(st.n_atoms) for st in strs]
            h_e = hat_h_diag[ebegin : ebegin + len(sliced_data.energies)]
            sl_pred_e = (
                y_pred[ebegin : ebegin + len(sliced_data.energies)] / n_total_atoms
            )
            sl_true_e = sliced_data.energies / n_total_atoms
            true_e.extend(sl_true_e.tolist())
            e_errors.extend(((sl_true_e - sl_pred_e) / h_e).tolist())

            if sliced_data.exist_force:
                h_f = hat_h_diag[fbegin : fbegin + len(sliced_data.forces)]
                sl_pred_f = y_pred[fbegin : fbegin + len(sliced_data.forces)]
                sl_true_f = sliced_data.forces
                true_f.extend(sl_true_f.tolist())
                f_errors.extend(((sl_true_f - sl_pred_f) / h_f).tolist())

            if sliced_data.exist_stress:
                if stress_unit == "eV":
                    normalize = np.repeat(n_total_atoms, 6)
                elif stress_unit == "GPa":
                    eV_to_GPa = 160.21766208
                    volumes = [st.volume for st in strs]
                    normalize = np.repeat(volumes, 6) / eV_to_GPa
                h_s = hat_h_diag[sbegin : sbegin + len(sliced_data.stresses)]
                sl_pred_s = (
                    y_pred[sbegin : sbegin + len(sliced_data.stresses)] / normalize
                )
                sl_true_s = sliced_data.stresses / normalize
                true_s.extend(sl_true_s.tolist())
                s_errors.extend(((sl_true_s - sl_pred_s) / h_s).tolist())

        rmse_e = np.sqrt(np.mean(np.power(e_errors, 2)))
        mae_e = np.mean(np.abs(e_errors))
        if len(f_errors) > 0:
            rmse_f = np.sqrt(np.mean(np.power(f_errors, 2)))
            mae_f = np.mean(np.abs(f_errors))
        if len(s_errors) > 0:
            rmse_s = np.sqrt(np.mean(np.power(s_errors, 2)))
            mae_s = np.mean(np.abs(s_errors))

        error_dict = {
            "energy": rmse_e,
            "force": rmse_f,
            "stress": rmse_s,
            "energy_mae": mae_e,
            "force_mae": mae_f,
            "stress_mae": mae_s,
            "percent_force_norm": None,
            "force_direction": None,
        }
        if self._verbose:
            self.print_error(error_dict, key=output_key)

        if log_energy or log_force or log_stress:
            os.makedirs(path_output + "/predictions", exist_ok=True)
            if log_energy:
                true_e = np.array(true_e)
                pred_e = true_e - np.array(e_errors)
                self._write_energies(dataset, true_e, pred_e, path_output, output_key)
            if log_force:
                true_f = np.array(true_f)
                pred_f = true_f - np.array(f_errors)
                self._write_forces(true_f, pred_f, path_output, output_key)
            if log_stress:
                true_s = np.array(true_s)
                pred_s = true_s - np.array(s_errors)
                self._write_stresses(true_s, pred_s, path_output, output_key)

        return error_dict


class PolymlpErrorUseXLOOCV(PolymlpErrorBase):
    """Class for computing LOOCV prediction errors."""

    def __init__(self, mlp: PolymlpDataMLP, verbose: bool = False):
        """Init method."""
        super().__init__(mlp, verbose=verbose)

    def compute_error(
        self,
        datasets: DatasetList,
        data_xy: PolymlpDataXY,
        stress_unit: Literal["eV", "GPa"] = "eV",
        log_energy: bool = True,
        log_force: bool = False,
        log_stress: bool = False,
        path_output: bool = "./",
        tag: str = "train",
    ):
        """Compute cross-validation errors and predicted values for all datasets."""
        if data_xy.hat_ii is None:
            raise RuntimeError("Hat matrix not found.")

        self._errors = dict()
        for data, indices in zip(datasets, data_xy.first_indices, strict=True):
            output_key = self._generate_output_key(data.name, tag=tag)
            self._errors[f"LOOCV:{data.name}"] = self.compute_error_single(
                data,
                indices,
                data_xy.hat_ii,
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
        indices: tuple,
        hat_ii: np.ndarray,
        output_key: str = "train",
        stress_unit: Literal["eV", "GPa"] = "eV",
        log_energy: bool = True,
        log_force: bool = False,
        log_stress: bool = False,
        path_output: bool = "./",
    ):
        """Compute cross-validation errors and predicted values for single dataset."""
        strs = dataset.structures
        n_total_atoms = [sum(st.n_atoms) for st in strs]
        pred_e, pred_f, pred_s = self._eval_properties(strs)

        ebegin, fbegin, sbegin = indices
        eend = ebegin + dataset.energies.shape[0]
        e1, e2 = self._apply_hat(dataset.energies, pred_e, hat_ii, ebegin, eend)
        rmse_e, _, _ = self._compute_rmse(e1, e2, normalize=n_total_atoms)
        mae_e, _, _ = self._compute_mae(e1, e2, normalize=n_total_atoms)

        rmse_f = None
        mae_f = None
        if dataset.exist_force:
            fend = fbegin + dataset.forces.shape[0]
            f1, f2 = self._apply_hat(dataset.forces, pred_f, hat_ii, fbegin, fend)
            rmse_f, _, _ = self._compute_rmse(f1, f2)
            mae_f, _, _ = self._compute_mae(f1, f2)

        rmse_s = None
        mae_s = None
        if dataset.exist_stress:
            send = sbegin + dataset.stresses.shape[0]
            normalize = self._stress_normalize_coeffs(strs, stress_unit)
            s1, s2 = self._apply_hat(dataset.stresses, pred_s, hat_ii, sbegin, send)
            rmse_s, _, _ = self._compute_rmse(s1, s2, normalize=normalize)
            mae_s, _, _ = self._compute_mae(s1, s2, normalize=normalize)

        error_dict = {
            "energy": rmse_e,
            "force": rmse_f,
            "stress": rmse_s,
            "energy_mae": mae_e,
            "force_mae": mae_f,
            "stress_mae": mae_s,
            "percent_force_norm": None,
            "force_direction": None,
        }
        if self._verbose:
            self.print_error(error_dict, key=output_key)

        if log_energy or log_force or log_stress:
            os.makedirs(path_output + "/predictions", exist_ok=True)
            if log_energy:
                pred_e = pred_e / n_total_atoms
                true_e = dataset.energies / n_total_atoms
                self._write_energies(dataset, true_e, pred_e, path_output, output_key)
            # if log_force:
            #     self._write_forces(true_f, pred_f, path_output, output_key)
            # if log_stress:
            #     self._write_stresses(true_s, pred_s, path_output, output_key)

        return error_dict

    def _apply_hat(
        self,
        true: np.ndarray,
        pred: np.ndarray,
        hat_ii: np.ndarray,
        begin: int,
        end: int,
    ):
        """Apply 1.0/(1-hat_ii)."""
        hat = hat_ii[begin:end]
        denom = 1 - hat
        val1 = true / denom
        val2 = pred / denom
        return val1, val2
