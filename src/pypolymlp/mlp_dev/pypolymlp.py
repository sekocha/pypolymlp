"""Pypolymlp API."""

import itertools
from typing import Literal, Optional

import numpy as np

from pypolymlp.core.data_format import (
    PolymlpDataMLP,
    PolymlpGtinvParams,
    PolymlpModelParams,
    PolymlpParams,
    PolymlpStructure,
)
from pypolymlp.core.displacements import convert_disps_to_positions, set_dft_data
from pypolymlp.core.interface_vasp import (
    parse_structures_from_poscars,
    set_data_from_structures,
)
from pypolymlp.core.io_polymlp import load_mlp_lammps
from pypolymlp.mlp_dev.core.accuracy import PolymlpDevAccuracy
from pypolymlp.mlp_dev.core.mlpdev_data import PolymlpDevData
from pypolymlp.mlp_dev.standard.mlpdev_dataxy import (
    PolymlpDevDataXY,
    PolymlpDevDataXYSequential,
)
from pypolymlp.mlp_dev.standard.regression import Regression


class Pypolymlp:
    """Pypolymlp API."""

    def __init__(self):
        """Init method."""
        self._params = None
        self._train = None
        self._test = None
        self._reg = None
        self._mlp_model = None
        self._multiple_datasets = False

        """Hybrid models are not available at this time."""
        # self.__hybrid = None

    def set_params(
        self,
        params: Optional[PolymlpParams] = None,
        elements: tuple[str] = None,
        include_force: bool = True,
        include_stress: bool = False,
        cutoff: float = 6.0,
        model_type: Literal[1, 2, 3, 4] = 4,
        max_p: Literal[1, 2, 3] = 2,
        feature_type: Literal["pair", "gtinv"] = "gtinv",
        gaussian_params1: tuple[float, float, int] = (1.0, 1.0, 1),
        gaussian_params2: tuple[float, float, int] = (0.0, 5.0, 7),
        distance: Optional[dict] = None,
        reg_alpha_params: tuple[float, float, int] = (-3.0, 1.0, 5),
        gtinv_order: int = 3,
        gtinv_maxl: tuple[int] = (4, 4, 2, 1, 1),
        gtinv_version: Literal[1, 2] = 1,
        atomic_energy: tuple[float] = None,
        rearrange_by_elements: bool = True,
    ):
        """Assign input parameters.

        Parameters
        ----------
        elements: Element species, (e.g., ['Mg','O'])
        include_force: Considering force entries
        include_stress: Considering stress entries
        cutoff: Cutoff radius
        model_type: Polynomial function type
            model_type = 1: Linear polynomial of polynomial invariants
            model_type = 2: Polynomial of polynomial invariants
            model_type = 3: Polynomial of pair invariants
                            + linear polynomial of polynomial invariants
            model_type = 4: Polynomial of pair and second-order invariants
                            + linear polynomial of polynomial invariants
        max_p: Order of polynomial function
        feature_type: 'gtinv' or 'pair'
        gaussian_params: Parameters for exp[- param1 * (r - param2)**2]
            Parameters are given as np.linspace(p[0], p[1], p[2]),
            where p[0], p[1], and p[2] are given by gaussian_params1
            and gaussian_params2.
        distance: Interatomic distances for element pairs.
            (e.g.) distance = {(Sr, Sr): [3.5, 4.8], (Ti, Ti): [2.5, 5.5]}
        reg_alpha_params: Parameters for penalty term in
            linear ridge regression. Parameters are given as
            np.linspace(p[0], p[1], p[2]).
        gtinv_order: Maximum order of polynomial invariants.
        gtinv_maxl: Maximum angular numbers of polynomial invariants.
            [maxl for order=2, maxl for order=3, ...]
        atomic_energy: Atomic energies.
        rearrange_by_elements: Set True if not developing special MLPs.
        """

        if params is not None:
            self._params = params
        else:
            n_type = len(elements)

            assert len(gaussian_params1) == len(gaussian_params2) == 3
            params1 = self._sequence(gaussian_params1)
            params2 = self._sequence(gaussian_params2)
            pair_params = list(itertools.product(params1, params2))
            pair_params.append([0.0, 0.0])

            if atomic_energy is None:
                atomic_energy = tuple([0.0 for i in range(n_type)])
            else:
                assert len(atomic_energy) == n_type

            element_order = elements if rearrange_by_elements else None

            if feature_type == "gtinv":
                gtinv = PolymlpGtinvParams(
                    order=gtinv_order,
                    max_l=gtinv_maxl,
                    n_type=n_type,
                    version=gtinv_version,
                )
                max_l = max(gtinv_maxl)
            else:
                gtinv = PolymlpGtinvParams(
                    order=0,
                    max_l=[],
                    n_type=n_type,
                )
                max_l = 0

            pair_params_cond, pair_cond = self._set_pair_params_conditional(
                pair_params, elements, distance
            )

            model = PolymlpModelParams(
                cutoff=cutoff,
                model_type=model_type,
                max_p=max_p,
                max_l=max_l,
                feature_type=feature_type,
                gtinv=gtinv,
                pair_type="gaussian",
                pair_conditional=pair_cond,
                pair_params=pair_params,
                pair_params_conditional=pair_params_cond,
            )

            self._params = PolymlpParams(
                n_type=n_type,
                elements=elements,
                model=model,
                atomic_energy=atomic_energy,
                regression_alpha=np.linspace(
                    reg_alpha_params[0], reg_alpha_params[1], reg_alpha_params[2]
                ),
                include_force=include_force,
                include_stress=include_stress,
                element_order=element_order,
            )

        return self

    def _set_pair_params_conditional(
        self,
        pair_params: np.ndarray,
        elements: list,
        distance: dict,
    ):
        """Set active parameter indices for element pairs."""
        if distance is None:
            cond = False
            distance = dict()
        else:
            cond = True
            for k in distance.keys():
                k = sorted(k, key=lambda x: elements.index(x))

        atomtypes = dict()
        for i, ele in enumerate(elements):
            atomtypes[ele] = i

        element_pairs = itertools.combinations_with_replacement(elements, 2)
        pair_params_indices = dict()
        for ele_pair in element_pairs:
            key = (atomtypes[ele_pair[0]], atomtypes[ele_pair[1]])
            if ele_pair not in distance:
                pair_params_indices[key] = list(range(len(pair_params)))
            else:
                match = [len(pair_params) - 1]
                for dis in distance[ele_pair]:
                    for i, p in enumerate(pair_params[:-1]):
                        if dis < p[1] + 1 / p[0] and dis > p[1] - 1 / p[0]:
                            match.append(i)
                pair_params_indices[key] = sorted(set(match))
        return pair_params_indices, cond

    def set_datasets_vasp(self, train_vaspruns: list[str], test_vaspruns: list[str]):
        """Set single DFT dataset in vasp format.

        Parameters
        ----------
        train_vaspruns: vasprun files for training dataset (list)
        test_vaspruns: vasprun files for test dataset (list)
        """
        if self._params is None:
            raise KeyError(
                "Set parameters using set_params() " "before using set_datasets."
            )

        self._params.dft_train = dict()
        self._params.dft_test = dict()
        self._params.dft_train["train_single"] = {
            "vaspruns": sorted(train_vaspruns),
            "include_force": self._params.include_force,
            "weight": 1.0,
        }
        self._params.dft_test["test_single"] = {
            "vaspruns": sorted(test_vaspruns),
            "include_force": self._params.include_force,
            "weight": 1.0,
        }
        self._multiple_datasets = True
        return self

    def set_multiple_datasets_vasp(
        self,
        train_vaspruns: list[list[str]],
        test_vaspruns: list[list[str]],
    ):
        """Set multiple DFT datasets in vasp format.

        Parameters
        ----------
        train_vaspruns: list of list containing vasprun files (training)
        test_vaspruns: list of list containing vasprun files (test)
        """
        if self._params is None:
            raise KeyError(
                "Set parameters using set_params() " "before using set_datasets."
            )

        self._params.dataset_type = "vasp"
        self._params.dft_train = dict()
        self._params.dft_test = dict()
        for i, vaspruns in enumerate(train_vaspruns):
            self._params.dft_train["dataset" + str(i + 1)] = {
                "vaspruns": sorted(vaspruns),
                "include_force": self._params.include_force,
                "weight": 1.0,
            }
        for i, vaspruns in enumerate(test_vaspruns):
            self._params.dft_test["dataset" + str(i + 1)] = {
                "vaspruns": sorted(vaspruns),
                "include_force": self._params.include_force,
                "weight": 1.0,
            }
        self._multiple_datasets = True
        return self

    def set_datasets_phono3py(
        self,
        train_yaml: str,
        test_yaml: str,
        train_energy_dat: str = None,
        test_energy_dat: str = None,
        train_ids: tuple[int] = None,
        test_ids: tuple[int] = None,
    ):
        """Set single DFT dataset in phono3py format."""
        if self._params is None:
            raise KeyError(
                "Set parameters using set_params() " "before using set_datasets."
            )

        self._params.dataset_type = "phono3py"
        self._params.dft_train = {
            "phono3py_yaml": train_yaml,
            "energy": train_energy_dat,
            "indices": train_ids,
        }
        self._params.dft_test = {
            "phono3py_yaml": test_yaml,
            "energy": test_energy_dat,
            "indices": test_ids,
        }
        return self

    def set_datasets_displacements(
        self,
        train_disps: np.ndarray,
        train_forces: np.ndarray,
        train_energies: np.ndarray,
        test_disps: np.ndarray,
        test_forces: np.ndarray,
        test_energies: np.ndarray,
        structure_without_disp: PolymlpStructure,
    ):
        """Set datasets from displacements-(energies, forces) sets.

        Parameters
        ----------
        train_disps: Displacements (training), shape=(n_train, 3, n_atoms).
        train_forces: Forces (training), shape=(n_train, 3, n_atoms).
        train_energies: Energies (training), shape=(n_train).
        test_disps: Displacements (test), shape=(n_test, 3, n_atom).
        test_forces: Forces (test data), shape=(n_test, 3, n_atom).
        test_energies: Energies (test data), shape=(n_test).
        structure_without_disp: Structure without displacements, PolymlpStructure
        """
        if self._params is None:
            raise KeyError(
                "Set parameters using set_params() "
                "before using set_datasets_displacements."
            )
        assert train_disps.shape[1] == 3
        assert test_disps.shape[1] == 3
        assert train_disps.shape[0] == train_energies.shape[0]
        assert test_disps.shape[0] == test_energies.shape[0]
        assert train_disps.shape == train_forces.shape
        assert test_disps.shape == test_forces.shape

        self._train = self._set_dft_data_from_displacements(
            train_disps,
            train_forces,
            train_energies,
            structure_without_disp,
            element_order=self._params.element_order,
        )
        self._test = self._set_dft_data_from_displacements(
            test_disps,
            test_forces,
            test_energies,
            structure_without_disp,
            element_order=self._params.element_order,
        )
        self._train.name = "train_single"
        self._test.name = "test_single"
        self._train = [self._train]
        self._test = [self._test]
        self._multiple_datasets = True
        return self

    def set_datasets_structures(
        self,
        train_structures: Optional[list[PolymlpStructure]] = None,
        test_structures: Optional[list[PolymlpStructure]] = None,
        train_energies: Optional[np.ndarray] = None,
        test_energies: Optional[np.ndarray] = None,
        train_forces: Optional[list[np.ndarray]] = None,
        test_forces: Optional[list[np.ndarray]] = None,
        train_stresses: Optional[np.ndarray] = None,
        test_stresses: Optional[np.ndarray] = None,
    ):
        """Set datasets from structures-(energies, forces, stresses) sets.

        Parameters
        ----------
        train_structures: Structures in PolymlpStructure format (training).
        test_structures: Structures in PolymlpStructure format (test).
        train_energies: Energies (training), shape=(n_train) in eV/cell.
        test_energies: Energies (test data), shape=(n_test) in eV/cell.
        train_forces: Forces (training), shape = n_train x (3, n_atoms_i) in eV/ang.
        test_forces: Forces (test), shape= n_test x (3, n_atoms_i) in eV/ang.
        train_stresses: Stress tensors (training), shape=(n_train, 3, 3), in eV/cell.
        test_stresses: Stress tensors (test data), shape=(n_test, 3, 3) in eV/cell.
        """
        if self._params is None:
            raise KeyError(
                "Set parameters using set_params() "
                "before using set_datasets_displacements."
            )
        assert train_structures is not None
        assert test_structures is not None
        assert train_energies is not None
        assert test_energies is not None
        assert len(train_structures) == len(train_energies)
        assert len(test_structures) == len(test_energies)
        if train_forces is not None:
            assert len(train_structures) == len(train_forces)
            assert train_forces[0].shape[0] == 3
        if test_forces is not None:
            assert len(test_structures) == len(test_forces)
            assert test_forces[0].shape[0] == 3
        if train_stresses is not None:
            assert len(train_structures) == len(train_stresses)
            assert train_stresses[0].shape[0] == 3
            assert train_stresses[0].shape[1] == 3
        if test_stresses is not None:
            assert len(test_structures) == len(test_stresses)
            assert test_stresses[0].shape[0] == 3
            assert test_stresses[0].shape[1] == 3

        self._train = set_data_from_structures(
            train_structures,
            train_energies,
            train_forces,
            train_stresses,
            element_order=self._params.element_order,
        )
        self._test = set_data_from_structures(
            test_structures,
            test_energies,
            test_forces,
            test_stresses,
            element_order=self._params.element_order,
        )
        self._train.name = "train_single"
        self._test.name = "test_single"
        self._train = [self._train]
        self._test = [self._test]
        self._multiple_datasets = True
        return self

    def _set_dft_data_from_displacements(
        self,
        disps: np.ndarray,
        forces: np.ndarray,
        energies: np.ndarray,
        structure_without_disp: PolymlpStructure,
        element_order=None,
    ):

        positions_all = convert_disps_to_positions(
            disps,
            structure_without_disp.axis,
            structure_without_disp.positions,
        )
        dft = set_dft_data(
            forces,
            energies,
            positions_all,
            structure_without_disp,
            element_order=element_order,
        )
        return dft

    def _sequence(self, params):
        return np.linspace(float(params[0]), float(params[1]), int(params[2]))

    def run(
        self,
        file_params=None,
        sequential=None,
        batch_size=None,
        path_output="./",
        verbose=False,
        output_files=False,
    ):
        """Run linear ridge regression to estimate MLP coefficients."""

        polymlp_in = PolymlpDevData()
        if file_params is not None:
            polymlp_in.parse_infiles(file_params, verbose=True)
            self._params = polymlp_in.params
        else:
            polymlp_in.params = self._params

        if self._train is None:
            polymlp_in.parse_datasets()
        else:
            polymlp_in.train = self._train
            polymlp_in.test = self._test

        if output_files:
            polymlp_in.write_polymlp_params_yaml(
                filename=path_output + "/polymlp_params.yaml"
            )
        n_features = polymlp_in.n_features

        if sequential is None:
            sequential = True if polymlp_in.is_multiple_datasets else False

        if not sequential:
            polymlp = PolymlpDevDataXY(polymlp_in, verbose=verbose).run()
            if verbose:
                polymlp.print_data_shape()
        else:
            if batch_size is None:
                batch_size = max((10000000 // n_features), 128)
            if verbose:
                print("Batch size:", batch_size, flush=True)
            polymlp = PolymlpDevDataXYSequential(polymlp_in, verbose=verbose).run_train(
                batch_size=batch_size
            )

        self._reg = Regression(polymlp).fit(
            seq=sequential,
            clear_data=True,
            batch_size=batch_size,
        )
        self._mlp_model = self._reg.best_model
        if output_files:
            self._reg.save_mlp_lammps(filename=path_output + "/polymlp.lammps")

        acc = PolymlpDevAccuracy(self._reg)
        acc.compute_error(
            path_output=path_output,
            verbose=verbose,
            log_energy=output_files,
        )
        if output_files:
            acc.write_error_yaml(filename=path_output + "/polymlp_error.yaml")

        self._mlp_model.error_train = acc.error_train_dict
        self._mlp_model.error_test = acc.error_test_dict
        return self

    def get_structures_from_poscars(self, poscars: list[str]) -> list[PolymlpStructure]:
        return parse_structures_from_poscars(poscars)

    @property
    def parameters(self) -> PolymlpParams:
        """Return parameters of developed polymlp."""
        return self._params

    @property
    def summary(self):
        """Return summary of developed polymlp.

        Attributes
        ----------
        coeffs: MLP coefficients.
        scales: Scales of features. scaled_coeffs (= coeffs / scales)
                must be used for calculating properties.
        rmse: Root-mean-square error for test data.
        alpha, beta: Optimal regurlarization parameters.
        predictions_train, predictions_test: Predicted energy values.
        error_train, error_test: Root-mean square error for each dataset.
        """
        return self._mlp_model

    @property
    def coeffs(self):
        """Return scaled coefficients.

        It is appropriate to use the scaled coefficient to calculate properties.
        """
        return self._mlp_model.coeffs / self._mlp_model.scales

    def save_mlp(self, filename="polymlp.lammps"):
        """Save polynomial MLP as file."""
        self._reg.save_mlp_lammps(filename=filename)

    def load_mlp(self, filename="polymlp.lammps"):
        """Load polynomial MLP from file."""
        self._params, mlp_dict = load_mlp_lammps(filename)
        self._mlp_model = PolymlpDataMLP(
            coeffs=mlp_dict["coeffs"],
            scales=mlp_dict["scales"],
        )
