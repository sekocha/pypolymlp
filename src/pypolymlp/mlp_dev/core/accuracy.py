"""Class for computing prediction errors."""

import itertools
import math
import os

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpParams
from pypolymlp.core.utils import rmse
from pypolymlp.mlp_dev.core.regression_base import RegressionBase


class PolymlpDevAccuracy:

    def __init__(self, reg: RegressionBase):

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

    def print_error(self, error, key="train"):

        print("prediction:", key)
        print(
            "  rmse_energy:",
            "{0:13.5f}".format(error["energy"] * 1000),
            "(meV/atom)",
            flush=True,
        )
        print("  rmse_force: ", "{0:13.5f}".format(error["force"]), "(eV/ang)")
        print(
            "  rmse_stress:",
            "{0:13.5f}".format(error["stress"] * 1000),
            "(meV/atom)",
            flush=True,
        )
        return self

    def write_error_yaml(self, filename="polymlp_error.yaml"):

        self._write_error_yaml(self._error_train, tag="train", filename=filename)
        self._write_error_yaml(
            self._error_test,
            tag="test",
            filename=filename,
            initialize=False,
        )

    def _write_error_yaml(
        self,
        error,
        tag="train",
        filename="polymlp_error.yaml",
        initialize=True,
    ):

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

    def _compute_rmse(
        self, true_values, pred_values, normalize=None, return_values=False
    ):

        if normalize is None:
            true = true_values
            pred = pred_values
        else:
            true = true_values / np.array(normalize)
            pred = pred_values / np.array(normalize)

        if return_values:
            return rmse(true, pred), true, pred
        return rmse(true, pred)

    def compute_error(
        self,
        stress_unit="eV",
        log_energy=True,
        log_force=False,
        log_stress=False,
        path_output="./",
        verbose=True,
    ):

        if self._multiple_datasets:
            for dft in self._train:
                if "*" in dft.name:
                    output_key = ".".join(
                        dft.name.split("*")[0].split("/")[:-1]
                    ).replace("..", "")
                else:
                    output_key = "Train-" + dft.name

                self._error_train[dft.name] = self.compute_error_single(
                    dft,
                    output_key=output_key,
                    path_output=path_output,
                    log_energy=log_energy,
                    log_force=log_force,
                    log_stress=log_stress,
                    verbose=verbose,
                )
            for dft in self._test:
                if "*" in dft.name:
                    output_key = ".".join(
                        dft.name.split("*")[0].split("/")[:-1]
                    ).replace("..", "")
                else:
                    output_key = "Test-" + dft.name

                self._error_test[dft.name] = self.compute_error_single(
                    dft,
                    output_key=output_key,
                    path_output=path_output,
                    log_energy=log_energy,
                    log_force=log_force,
                    log_stress=log_stress,
                    verbose=verbose,
                )
        else:
            self._error_train[self._train.name] = self.compute_error_single(
                self._train,
                output_key="train",
                path_output=path_output,
                log_energy=log_energy,
                log_force=log_force,
                log_stress=log_stress,
                verbose=verbose,
            )
            self._error_test[self._test.name] = self.compute_error_single(
                self._test,
                output_key="test",
                path_output=path_output,
                log_energy=log_energy,
                log_force=log_force,
                log_stress=log_stress,
                verbose=verbose,
            )

    def compute_error_single(
        self,
        dft,
        output_key="train",
        stress_unit="eV",
        log_energy=True,
        log_force=False,
        log_stress=False,
        path_output="./",
        verbose=True,
    ):

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

        if log_force == False:
            rmse_f = self._compute_rmse(dft.forces, forces)
        else:
            rmse_f, true_f, pred_f = self._compute_rmse(
                dft.forces, forces, return_values=True
            )

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

        # for i, (t, p) in enumerate(zip(true_f1, pred_f1)):
        #    ist = i // 96 + 1
        #    iatom = i % 96 + 1
        #    print(ist, iatom, np.linalg.norm(t-p), t, p)

        if stress_unit == "eV":
            normalize = np.repeat(n_total_atoms, 6)
        elif stress_unit == "GPa":
            eV_to_GPa = 160.21766208
            volumes = [st.volume for st in strs]
            normalize = np.repeat(volumes, 6) / eV_to_GPa

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
        if verbose:
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
        return self._params

    @property
    def is_multiple_datasets(self) -> bool:
        return self._multiple_datasets

    @property
    def is_hybrid(self) -> bool:
        return self._hybrid

    @property
    def error_train_dict(self) -> dict:
        return self._error_train

    @property
    def error_test_dict(self) -> dict:
        return self._error_test


# def __compute_rmse(true_values,
#                   pred_values_all,
#                   weight_all,
#                   begin_id,
#                   end_id,
#                   normalize=None,
#                   return_values=False):
#
#    pred_values = pred_values_all[begin_id:end_id]
#    weight = weight_all[begin_id:end_id]
#
#    pred = pred_values / weight
#
#    if normalize is None:
#        true = true_values
#    else:
#        true = true_values / np.array(normalize)
#        pred /= np.array(normalize)
#    if return_values:
#        return rmse(true, pred), true, pred
#    return rmse(true, pred)


# def compute_error_single_dataset_from_features(
#    dft_dict, params_dict,
#                  predictions_all,
#                  weights_all,
#                  first_indices,
#                  output_key='train',
#                  log_force=False,
#                  path_output='./'):
#
#    if 'include_force' in dft_dict:
#        include_force = dft_dict['include_force']
#    else:
#        include_force = params_dict['include_force']
#
#    if include_force == False:
#        include_stress = False
#    else:
#        include_stress = params_dict['include_stress']
#
#    n_data = len(predictions_all)
#    ebegin, fbegin, sbegin = first_indices
#    eend = ebegin + len(dft_dict['energy'])
#    if include_force:
#        fend = fbegin + len(dft_dict['force'])
#        send = sbegin + len(dft_dict['stress'])
#
#    n_total_atoms = [sum(st['n_atoms'])
#                     for st in dft_dict['structures']]
#    rmse_e, true_e, pred_e = __compute_rmse(dft_dict['energy'],
#                                            predictions_all,
#                                            weights_all,
#                                            ebegin, eend,
#                                            normalize=n_total_atoms,
#                                            return_values=True)
#
#    rmse_f, rmse_s = None, None
#    if include_force:
#        rmse_f = __compute_rmse(dft_dict['force'],
#                                predictions_all,
#                                weights_all,
#                                fbegin, fend)
#
#    if include_stress:
#        stress_unit = 'eV'
#        if stress_unit == 'eV':
#            normalize = np.repeat(n_total_atoms, 6)
#        elif stress_unit == 'GPa':
#            eV_to_GPa = 160.21766208
#            volumes = [st['volume'] for st in dft_dict['structures']]
#            normalize = np.repeat(volumes, 6)/eV_to_GPa
#        rmse_s = __compute_rmse(dft_dict['stress'],
#                                predictions_all,
#                                weights_all,
#                                sbegin, send,
#                                normalize=normalize)
#
#    error_dict = dict()
#    error_dict['energy'] = rmse_e
#    error_dict['force'] = rmse_f
#    error_dict['stress'] = rmse_s
#    print_error(error_dict, key=output_key)
#
#    filenames = dft_dict['filenames']
#    outdata = np.array([true_e, pred_e, (true_e - pred_e) * 1000]).T
#
#    os.makedirs(path_output + '/predictions', exist_ok=True)
#    f = open(path_output + '/predictions/energy.' + output_key + '.dat', 'w')
#    print('# DFT(eV/atom), MLP(eV/atom), DFT-MLP(meV/atom)', file=f)
#    for d, name in zip(outdata, filenames):
#        print(d[0], d[1], d[2], name, file=f)
#    f.close()
#
#    if log_force:
#        _, true_f, pred_f = __compute_rmse(dft_dict['force'],
#                                            predictions_all,
#                                            weights_all,
#                                            fbegin, fend,
#                                            return_values=True)
#        outdata = np.array([true_f, pred_f, (true_f - pred_f)]).T
#
#        f = open(path_output + '/predictions/force.' + output_key + '.dat', 'w')
#        print('# DFT, MLP, DFT-MLP', file=f)
#        for d in outdata:
#            print(d[0], d[1], d[2], file=f)
#        f.close()
#
#    return error_dict
