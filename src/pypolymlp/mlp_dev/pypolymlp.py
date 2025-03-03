"""Pypolymlp API."""

from typing import Literal, Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpDataMLP, PolymlpParams, PolymlpStructure
from pypolymlp.core.displacements import get_structures_from_displacements
from pypolymlp.core.interface_datasets import set_dataset_from_structures
from pypolymlp.core.interface_vasp import parse_structures_from_poscars
from pypolymlp.core.io_polymlp import convert_to_yaml, load_mlp
from pypolymlp.core.polymlp_params import set_all_params
from pypolymlp.core.utils import check_memory_size_in_regression, split_train_test
from pypolymlp.mlp_dev.core.accuracy import PolymlpDevAccuracy
from pypolymlp.mlp_dev.core.mlpdev_data import PolymlpDevData
from pypolymlp.mlp_dev.core.utils_sequential import get_auto_batch_size
from pypolymlp.mlp_dev.standard.learning_curve import LearningCurve
from pypolymlp.mlp_dev.standard.mlpdev_dataxy import (
    PolymlpDevDataXY,
    PolymlpDevDataXYSequential,
)
from pypolymlp.mlp_dev.standard.regression import Regression


class Pypolymlp:
    """Pypolymlp API."""

    def __init__(self):
        """Init method."""
        self._polymlp_in = PolymlpDevData()
        self._params = None

        self._train = None
        self._test = None
        self._multiple_datasets = False
        # TODO: set_params is not available for hybrid models at this time.
        self._hybrid = False

        self._reg = None
        self._mlp_model = None
        self._acc = None

        np.set_printoptions(legacy="1.21")

    #    def __version__(self):
    #        """Version."""
    #        return __version__

    def load_parameter_file(self, file_params: Union[str, list[str]]):
        """Load input parameter file and set parameters."""
        self._polymlp_in.parse_infiles(file_params)
        self._params = self._polymlp_in.params
        self._hybrid = self._polymlp_in.is_hybrid
        return self

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
            self._params = self._polymlp_in.params = params
            return self

        self._params = set_all_params(
            elements=elements,
            include_force=include_force,
            include_stress=include_stress,
            cutoff=cutoff,
            model_type=model_type,
            max_p=max_p,
            feature_type=feature_type,
            gaussian_params1=gaussian_params1,
            gaussian_params2=gaussian_params2,
            distance=distance,
            reg_alpha_params=reg_alpha_params,
            gtinv_order=gtinv_order,
            gtinv_maxl=gtinv_maxl,
            gtinv_version=gtinv_version,
            atomic_energy=atomic_energy,
            rearrange_by_elements=rearrange_by_elements,
        )
        self._polymlp_in.params = self._params
        return self

    def print_params(self):
        """Print input parameters."""
        self._polymlp_in.print_params()
        return self

    def _is_params_none(self):
        """Check whether params instance exists."""
        if self._params is None:
            raise RuntimeError(
                "Set parameters using set_params() or load_parameter_file()",
                "before using set_datasets.",
            )
        return self

    def load_datasets(self):
        """Load datasets provided in params instance."""
        self._polymlp_in.parse_datasets()
        self._train = self._polymlp_in.train
        self._test = self._polymlp_in.test
        return self

    def _split_dataset_auto(self, files: list[str], train_ratio: float = 0.9):
        """Split dataset into training and test datasets automatically."""
        train_files, test_files = split_train_test(files, train_ratio=train_ratio)
        self._params.dft_train = sorted(train_files)
        self._params.dft_test = sorted(test_files)
        self._multiple_datasets = False
        return self

    def set_datasets_electron(
        self,
        yamlfiles: list[str],
        temperature: float = 300,
        target: Literal[
            "free_energy",
            "energy",
            "entropy",
            "specific_heat",
        ] = "free_energy",
        train_ratio: float = 0.9,
    ):
        """Set single electron dataset.

        Parameters
        ----------
        yamlfiles: electron.yaml files (list)
        """
        self._is_params_none()
        self._params.dataset_type = "electron"
        self._params.include_force = False
        self._params.temperature = temperature
        self._params.electron_property = target

        self._split_dataset_auto(yamlfiles, train_ratio=train_ratio)
        self.load_datasets()
        return self

    def set_datasets_sscha(self, yamlfiles: list[str], train_ratio: float = 0.9):
        """Set single sscha dataset.

        Parameters
        ----------
        yamlfiles: sscha_results.yaml files (list)
        """
        self._is_params_none()
        self._params.dataset_type = "sscha"
        self._params.include_force = True

        self._split_dataset_auto(yamlfiles, train_ratio=train_ratio)
        self.load_datasets()
        return self

    def set_datasets_vasp(self, train_vaspruns: list[str], test_vaspruns: list[str]):
        """Set single DFT dataset in vasp format.

        Parameters
        ----------
        train_vaspruns: vasprun files for training dataset (list)
        test_vaspruns: vasprun files for test dataset (list)
        """
        self._is_params_none()
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
        self.load_datasets()
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
        self._is_params_none()
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
        self.load_datasets()
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
        self._is_params_none()
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
        self.load_datasets()
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
        self._is_params_none()
        assert train_disps.shape[1] == 3
        assert test_disps.shape[1] == 3
        assert train_disps.shape[0] == train_energies.shape[0]
        assert test_disps.shape[0] == test_energies.shape[0]
        assert train_disps.shape == train_forces.shape
        assert test_disps.shape == test_forces.shape

        train_strs = get_structures_from_displacements(
            train_disps, structure_without_disp
        )
        test_strs = get_structures_from_displacements(
            test_disps, structure_without_disp
        )
        self.set_datasets_structures(
            train_strs,
            test_strs,
            train_energies,
            test_energies,
            train_forces,
            test_forces,
            train_stresses=None,
            test_stresses=None,
        )
        return self

    def set_datasets_structures(
        self,
        train_structures: list[PolymlpStructure],
        test_structures: list[PolymlpStructure],
        train_energies: np.ndarray,
        test_energies: np.ndarray,
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
        self._is_params_none()
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

        self._train = set_dataset_from_structures(
            train_structures,
            train_energies,
            train_forces,
            train_stresses,
            element_order=self._params.element_order,
        )
        self._test = set_dataset_from_structures(
            test_structures,
            test_energies,
            test_forces,
            test_stresses,
            element_order=self._params.element_order,
        )
        self._post_datasets_from_api()
        return self

    def _post_datasets_from_api(self):
        """Set datasets as a post process."""
        self._train.name = "train_single"
        self._test.name = "test_single"
        self._train = [self._train]
        self._test = [self._test]
        self._polymlp_in.train = self._train
        self._polymlp_in.test = self._test
        self._multiple_datasets = True
        return self

    def fit(
        self,
        sequential: bool = True,
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ):
        """Estimate MLP coefficients, compute features, and compute X.T @ X.

        Parameters
        ----------
        sequential: Use sequential regression to save memory allocation.
                    Default is True.
        batch_size: Batch size for sequential regression.
        """
        if self._train is None or self._test is None:
            raise RuntimeError("Set input parameters and datasets.")

        mem_req = check_memory_size_in_regression(self._polymlp_in.n_features)
        if verbose:
            print("Minimum memory required for solver in GB:", mem_req, flush=True)
            print("Memory required for allocating X additionally.", flush=True)

        if not sequential:
            polymlp = PolymlpDevDataXY(self._polymlp_in, verbose=verbose).run()
            if verbose:
                polymlp.print_data_shape()
        else:
            if batch_size is None:
                batch_size = get_auto_batch_size(
                    self._polymlp_in.n_features,
                    verbose=verbose,
                )
            if verbose:
                print("Batch size for computing X:", batch_size, flush=True)
            polymlp = PolymlpDevDataXYSequential(
                self._polymlp_in,
                verbose=verbose,
            ).run_train(batch_size=batch_size)

        self._reg = Regression(polymlp).fit(
            seq=sequential,
            clear_data=True,
            batch_size=batch_size,
        )
        self._mlp_model = self._reg.best_model
        return self

    def estimate_error(
        self,
        log_energy: bool = False,
        file_path: str = "./",
        verbose: bool = False,
    ):
        """Estimate prediction errors."""
        if self._reg is None:
            raise RuntimeError("Regression must be performed before estimating errors.")

        self._acc = PolymlpDevAccuracy(self._reg)
        self._acc.compute_error(
            log_energy=log_energy,
            path_output=file_path,
            verbose=verbose,
        )
        self._mlp_model.error_train = self._acc.error_train_dict
        self._mlp_model.error_test = self._acc.error_test_dict
        return self

    def run(
        self,
        sequential: bool = True,
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ):
        """Estimate MLP coefficients and prediction errors.

        Parameters
        ----------
        sequential: Use sequential regression to save memory allocation.
                    Default is True.
        batch_size: Batch size for sequential regression.
        """
        self._polymlp = self.fit(
            sequential=sequential,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.estimate_error(verbose=verbose)
        return self

    def fit_learning_curve(self, verbose: bool = False):
        """Compute learing curve."""
        if self._train is None or self._test is None:
            raise RuntimeError("Set input parameters and datasets.")

        if len(self._train) > 1:
            raise RuntimeError("Use single dataset for learning curve calculation")

        polymlp = PolymlpDevDataXY(self._polymlp_in, verbose=verbose).run()
        total_n_atoms = self._train[0].total_n_atoms

        self._learning = LearningCurve(polymlp, total_n_atoms, verbose=verbose)
        self._learning.run()
        return self

    def save_learning_curve(self, filename="polymlp_learning_curve.dat"):
        """Save learing curve."""
        self._learning.save_log(filename=filename)
        return self

    def get_structures_from_poscars(self, poscars: list[str]) -> list[PolymlpStructure]:
        """Load poscar files and convert them to structure instances."""
        return parse_structures_from_poscars(poscars)

    def save_mlp(self, filename: str = "polymlp.yaml", yaml: bool = True):
        """Save polynomial MLP as file.

        When hybrid models are used, mlp files will be generated as
        filename.1, filename.2, ...
        """
        if yaml:
            self._reg.save_mlp(filename=filename)
        else:
            self._reg.save_mlp_lammps(filename=filename)
        return self

    def load_mlp(self, filename: str = "polymlp.yaml"):
        """Load polynomial MLP from file."""
        # TODO: hybrid is not available.
        self._params, coeffs = load_mlp(filename)
        scales = np.ones(len(coeffs))
        self._mlp_model = PolymlpDataMLP(coeffs=coeffs, scales=scales)
        return self

    def convert_to_yaml(
        self,
        filename_txt: str = "polymlp.lammps",
        filename_yaml: str = "polymlp.yaml",
    ):
        """Convert polymlp.lammps to polymlp.yaml."""
        convert_to_yaml(filename_txt, filename_yaml)
        return self

    def save_params(self, filename="polymlp_params.yaml"):
        """Save MLP parameters as file."""
        self._polymlp_in.write_polymlp_params_yaml(filename=filename)
        return self

    def save_errors(self, filename="polymlp_error.yaml"):
        """Save prediction errors as file."""
        self._acc.write_error_yaml(filename=filename)
        return self

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
    def parameters(self) -> PolymlpParams:
        """Return parameters of developed polymlp."""
        return self._params

    @property
    def coeffs(self):
        """Return scaled coefficients.

        Use this scaled coefficients to calculate properties.
        """
        if self._hybrid:
            return [c / s for c, s in zip(self._reg.coeffs, self._reg.scales)]
        return self._mlp_model.coeffs / self._mlp_model.scales
