#!/usr/bin/env python
import itertools
import os

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.utils import rmse
from pypolymlp.mlp_dev.core.regression_base import RegressionBase


class PolymlpDevAccuracy:

    def __init__(self, reg: RegressionBase):

        if reg.is_hybrid:
            coeffs_rescale = [c / s for c, s in zip(reg.coeffs, reg.scales)]
        else:
            coeffs_rescale = reg.coeffs / reg.scales

        self.__prop = Properties(params_dict=reg.params_dict, coeffs=coeffs_rescale)

        self.__params_dict = reg.params_dict
        self.__train_dict = reg.train_dict
        self.__test_dict = reg.test_dict
        self.__multiple_datasets = reg.is_multiple_datasets
        self.__hybrid = reg.is_hybrid
        self.__error_train = dict()
        self.__error_test = dict()

    def print_error(self, error, key="train"):

        print("prediction:", key)
        print(
            "  rmse_energy:",
            "{0:13.5f}".format(error["energy"] * 1000),
            "(meV/atom)",
        )
        print("  rmse_force: ", "{0:13.5f}".format(error["force"]), "(eV/ang)")
        print(
            "  rmse_stress:",
            "{0:13.5f}".format(error["stress"] * 1000),
            "(meV/atom)",
        )
        return self

    def write_error_yaml(self, filename="polymlp_error.yaml"):

        self.__write_error_yaml(self.__error_train, tag="train", filename=filename)
        self.__write_error_yaml(
            self.__error_test,
            tag="test",
            filename=filename,
            initialize=False,
        )

    def __write_error_yaml(
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

    def __compute_rmse(
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

        if self.__multiple_datasets:
            for set_id, dft_dict in self.__train_dict.items():
                if "*" in set_id:
                    output_key = ".".join(set_id.split("*")[0].split("/")[:-1]).replace(
                        "..", ""
                    )
                else:
                    output_key = "Train-" + set_id

                self.__error_train[set_id] = self.compute_error_single(
                    dft_dict,
                    output_key=output_key,
                    path_output=path_output,
                    log_energy=log_energy,
                    log_force=log_force,
                    log_stress=log_stress,
                    verbose=verbose,
                )
            for set_id, dft_dict in self.__test_dict.items():
                if "*" in set_id:
                    output_key = ".".join(set_id.split("*")[0].split("/")[:-1]).replace(
                        "..", ""
                    )
                else:
                    output_key = "Test-" + set_id

                self.__error_test[set_id] = self.compute_error_single(
                    dft_dict,
                    output_key=output_key,
                    path_output=path_output,
                    log_energy=log_energy,
                    log_force=log_force,
                    log_stress=log_stress,
                    verbose=verbose,
                )
        else:
            self.__error_train["1"] = self.compute_error_single(
                self.__train_dict,
                output_key="train",
                path_output=path_output,
                log_energy=log_energy,
                log_force=log_force,
                log_stress=log_stress,
                verbose=verbose,
            )
            self.__error_test["1"] = self.compute_error_single(
                self.__test_dict,
                output_key="test",
                path_output=path_output,
                log_energy=log_energy,
                log_force=log_force,
                log_stress=log_stress,
                verbose=verbose,
            )

    def compute_error_single(
        self,
        dft_dict,
        output_key="train",
        stress_unit="eV",
        log_energy=True,
        log_force=False,
        log_stress=False,
        path_output="./",
        verbose=True,
    ):

        strs = dft_dict["structures"]
        energies, forces, stresses = self.__prop.eval_multiple(strs)
        forces = np.array(
            list(itertools.chain.from_iterable([f.T.reshape(-1) for f in forces]))
        )
        stresses = stresses.reshape(-1)

        n_total_atoms = [sum(st["n_atoms"]) for st in strs]
        if log_energy is False:
            rmse_e = self.__compute_rmse(
                dft_dict["energy"],
                energies,
                normalize=n_total_atoms,
            )
        else:
            rmse_e, true_e, pred_e = self.__compute_rmse(
                dft_dict["energy"],
                energies,
                normalize=n_total_atoms,
                return_values=True,
            )

        if log_force is False:
            rmse_f = self.__compute_rmse(dft_dict["force"], forces)
        else:
            rmse_f, true_f, pred_f = self.__compute_rmse(
                dft_dict["force"], forces, return_values=True
            )

        if stress_unit == "eV":
            normalize = np.repeat(n_total_atoms, 6)
        elif stress_unit == "GPa":
            eV_to_GPa = 160.21766208
            volumes = [st["volume"] for st in strs]
            normalize = np.repeat(volumes, 6) / eV_to_GPa

        if log_stress is False:
            rmse_s = self.__compute_rmse(
                dft_dict["stress"], stresses, normalize=normalize
            )
        else:
            rmse_s, true_s, pred_s = self.__compute_rmse(
                dft_dict["stress"],
                stresses,
                normalize=normalize,
                return_values=True,
            )

        error_dict = {"energy": rmse_e, "force": rmse_f, "stress": rmse_s}
        if verbose:
            self.print_error(error_dict, key=output_key)

        if log_energy or log_force or log_stress:
            os.makedirs(path_output + "/predictions", exist_ok=True)
            filenames = dft_dict["filenames"]

            outdata = np.array([true_e, pred_e, (true_e - pred_e) * 1000]).T
            f = open(path_output + "/predictions/energy." + output_key + ".dat", "w")
            print("# DFT(eV/atom), MLP(eV/atom), DFT-MLP(meV/atom)", file=f)
            for d, name in zip(outdata, filenames):
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
    def params_dict(self):
        return self.__params_dict

    @property
    def is_multiple_datasets(self):
        return self.__multiple_datasets

    @property
    def is_hybrid(self):
        return self.__hybrid

    @property
    def error_train_dict(self):
        return self.__error_train

    @property
    def error_test_dict(self):
        return self.__error_test


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
