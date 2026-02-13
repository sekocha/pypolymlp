"""Pypolymlp API."""

import io
from typing import Literal, Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure
from pypolymlp.core.dataset import (
    set_datasets_from_multiple_filesets,
    set_datasets_from_single_fileset,
    set_datasets_from_structures,
)
from pypolymlp.core.displacements import get_structures_from_displacements
from pypolymlp.core.interface_vasp import parse_structures_from_poscars
from pypolymlp.core.io_polymlp import convert_to_yaml, load_mlp
from pypolymlp.core.parser_polymlp_params import ParamsParser
from pypolymlp.core.polymlp_params import print_params, set_all_params
from pypolymlp.core.utils import split_train_test
from pypolymlp.mlp_dev.core.dataclass import PolymlpDataMLP
from pypolymlp.mlp_dev.core.eval_accuracy import PolymlpEvalAccuracy, write_error_yaml
from pypolymlp.mlp_dev.core.features_attr import (
    get_num_features,
    write_polymlp_params_yaml,
)
from pypolymlp.mlp_dev.gradient.fit_cg import fit_cg
from pypolymlp.mlp_dev.gradient.fit_sgd import fit_sgd
from pypolymlp.mlp_dev.standard.fit import fit, fit_learning_curve, fit_standard
from pypolymlp.mlp_dev.standard.utils_learning_curve import save_learning_curve_log


class Pypolymlp:
    """Pypolymlp API."""

    def __init__(self, verbose: bool = False):
        """Init method."""
        self._verbose = verbose

        self._params = None
        self._common_params = None
        self._train = None
        self._test = None

        # TODO: For electrons.
        # self._train_yml = None
        # self._test_yml = None

        # TODO: set_params is not available for hybrid models at this time.
        self._hybrid = False

        self._mlp_model = None
        self._learning_log = None

        np.set_printoptions(legacy="1.21")

    def load_parameter_file(
        self,
        file_params: Union[str, list[str]],
        train_ratio: float = 0.9,
        prefix: Optional[str] = None,
    ):
        """Load input parameter file and set parameters."""
        parser = ParamsParser(
            file_params,
            train_ratio=train_ratio,
            prefix_data_location=prefix,
        )
        self._train, self._test = parser.train, parser.test
        self._params = parser.params
        self._common_params = parser.common_params
        if not isinstance(self._params, PolymlpParams):
            self._hybrid = True

        return self

    def set_params(
        self,
        params: Optional[PolymlpParams] = None,
        elements: tuple[str] = None,
        include_force: bool = True,
        include_stress: bool = True,
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
            self._params = self._common_params = params
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
        self._common_params = self._params
        return self

    def print_params(self):
        """Print input parameters."""
        print_params(self._params, self._common_params)
        if self._train is not None:
            print("datasets:", flush=True)
            print("  train_data:", flush=True)
            for dataset in self._train:
                print("  -", dataset.name, flush=True)
            print("  test_data:", flush=True)
            for dataset in self._test:
                print("  -", dataset.name, flush=True)
        return self

    def _is_params_none(self):
        """Check whether params instance exists."""
        if self._params is None:
            raise RuntimeError("Set parameters before using the function.")
        return self

    def _is_data_none(self):
        """Check whether train and test data exist."""
        if self._train is None or self._test is None:
            raise RuntimeError("Set parameters and datasets before using fit.")
        return self

    def _turn_off_derivative_flags(self, forces: list, stresses: np.ndarray):
        """ """
        if forces is None:
            self._params.include_force = False
            self._params.include_stress = False
        if stresses is None:
            self._params.include_stress = False
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
        temperature: Temperature (K).
        target: Target electronic property.
        train_ratio: Ratio between training and entire data sizes.
        """
        self._is_params_none()
        self._params.dataset_type = "electron"
        self._params.include_force = False
        self._params.include_stress = False
        self._params.temperature = temperature
        self._params.electron_property = target

        self._train, self._test = set_datasets_from_single_fileset(
            self._params,
            files=yamlfiles,
            train_ratio=train_ratio,
        )
        return self

    def set_datasets_sscha(self, yamlfiles: list[str], train_ratio: float = 0.9):
        """Set single sscha dataset.

        Parameters
        ----------
        yamlfiles: sscha_results.yaml files (list).
        train_ratio: Ratio between training and entire data sizes.
        """
        self._is_params_none()
        self._params.dataset_type = "sscha"
        self._params.include_force = True
        self._params.include_stress = False

        self._train, self._test = set_datasets_from_single_fileset(
            self._params,
            files=yamlfiles,
            train_ratio=train_ratio,
        )
        return self

    def set_datasets_vasp(
        self,
        train_vaspruns: Optional[list[str]] = None,
        test_vaspruns: Optional[list[str]] = None,
        vaspruns: Optional[list[str]] = None,
        train_ratio: float = 0.9,
    ):
        """Set single DFT dataset in vasp format.

        Parameters
        ----------
        train_vaspruns: vasprun files for training dataset (list)
        test_vaspruns: vasprun files for test dataset (list)
        vaspruns: vasprun files.
                  They are automatically divided into training and test datasets.
        train_ratio: Ratio between training and entire data sizes.
        """
        self._is_params_none()
        self._params.dataset_type = "vasp"
        if train_vaspruns is not None and test_vaspruns is not None:
            self._train, self._test = set_datasets_from_single_fileset(
                self._params,
                train_files=train_vaspruns,
                test_files=test_vaspruns,
            )
        else:
            self._train, self._test = set_datasets_from_single_fileset(
                self._params,
                files=vaspruns,
                train_ratio=train_ratio,
            )
        return self

    def set_multiple_datasets_vasp(
        self,
        train_vaspruns: list[list[str]],
        test_vaspruns: list[list[str]],
    ):
        """Set multiple DFT datasets in vasp format.

        Parameters
        ----------
        train_vaspruns: List of list containing vasprun files (training)
        test_vaspruns: List of list containing vasprun files (test)
        """
        self._is_params_none()
        self._params.dataset_type = "vasp"
        self._train, self._test = set_datasets_from_multiple_filesets(
            self._params,
            train_files=train_vaspruns,
            test_files=test_vaspruns,
        )
        return self

    def set_datasets_phono3py(self, yaml: str, train_ratio: float = 0.9):
        """Set single DFT dataset in phono3py format.

        Parameters
        ----------
        yaml: Phono3py yaml file.
        train_ratio: Ratio between training and entire data sizes.
        """
        self._is_params_none()
        self._params.dataset_type = "phono3py"
        self._params.include_stress = False
        self._train, self._test = set_datasets_from_single_fileset(
            self._params,
            files=yaml,
            train_ratio=train_ratio,
        )
        return self

    def set_datasets_structures_autodiv(
        self,
        structures: list[PolymlpStructure],
        energies: np.ndarray,
        forces: Optional[list[np.ndarray]] = None,
        stresses: Optional[np.ndarray] = None,
        train_ratio: float = 0.9,
    ):
        """Set datasets from structures-(energies, forces, stresses) sets.

        Given dataset is automatically divided into training and test datasets.

        Parameters
        ----------
        structures: Structures in PolymlpStructure format (training and test).
        energies: Energies (training and test), shape=(n_data) in eV/cell.
        forces: Forces (training and test), shape = n_data x (3, n_atoms_i) in eV/ang.
        stresses: Stress tensors (training and test), shape=(n_data, 3, 3), in eV/cell.
        train_ratio: Ratio between training and entire data sizes.
        """
        self._is_params_none()
        self._turn_off_derivative_flags(forces, stresses)

        self._train, self._test = set_datasets_from_structures(
            self._params,
            structures=structures,
            energies=energies,
            forces=forces,
            stresses=stresses,
            train_ratio=train_ratio,
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
        self._turn_off_derivative_flags(train_forces, train_stresses)
        self._turn_off_derivative_flags(test_forces, test_stresses)

        self._train, self._test = set_datasets_from_structures(
            self._params,
            train_structures=train_structures,
            test_structures=test_structures,
            train_energies=train_energies,
            test_energies=test_energies,
            train_forces=train_forces,
            test_forces=test_forces,
            train_stresses=train_stresses,
            test_stresses=test_stresses,
        )
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
        self._params.include_stress = False
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

    def fit(self, batch_size: Optional[int] = None, verbose: Optional[bool] = None):
        """Estimate MLP coefficients without computing entire X.

        Parameters
        ----------
        batch_size: Batch size for sequential regression.
                    If None, the batch size is automatically determined
                    depending on the memory size and number of features.
        """
        if verbose is not None:
            self._verbose = verbose

        self._is_data_none()
        self._mlp_model = fit(
            self._params,
            self._train,
            self._test,
            batch_size=batch_size,
            verbose=self._verbose,
        )
        return self

    def fit_standard(self, verbose: Optional[bool] = None):
        """Estimate MLP coefficients with direct evaluation of X."""
        if verbose is not None:
            self._verbose = verbose

        self._is_data_none()
        self._mlp_model = fit_standard(
            self._params,
            self._train,
            self._test,
            verbose=self._verbose,
        )
        return self

    def fit_cg(
        self,
        gtol: float = 1e-2,
        max_iter: Optional[int] = None,
        verbose: Optional[bool] = None,
    ):
        """Estimate MLP coefficients using conjugate gradient.

        Parameters
        ----------
        gtol: Gradient tolerance for CG.
        max_iter: Number of maximum iterations in CG.
        """
        if verbose is not None:
            self._verbose = verbose

        self._is_data_none()
        self._mlp_model = fit_cg(
            self._params,
            self._train,
            self._test,
            gtol=gtol,
            max_iter=max_iter,
            verbose=self._verbose,
        )
        return self

    def fit_sgd(self, verbose: Optional[bool] = None):
        """Estimate MLP coefficients using stochastic gradient descent."""
        raise NotImplementedError("SGD not available.")

        if verbose is not None:
            self._verbose = verbose

        self._is_data_none()
        self._mlp_model = fit_sgd(
            self._params,
            self._train,
            self._test,
            verbose=self._verbose,
        )
        return self

    def estimate_error(
        self,
        log_energy: bool = False,
        file_path: str = "./",
        verbose: Optional[bool] = None,
    ):
        """Estimate prediction errors."""
        if verbose is not None:
            self._verbose = verbose

        if self._mlp_model is None:
            raise RuntimeError("Regression must be performed before estimating errors.")

        acc = PolymlpEvalAccuracy(self._mlp_model, verbose=self._verbose)
        self._mlp_model.error_train = acc.compute_error(
            self._train,
            log_energy=log_energy,
            path_output=file_path,
            tag="train",
        )
        self._mlp_model.error_test = acc.compute_error(
            self._test,
            log_energy=log_energy,
            path_output=file_path,
            tag="test",
        )

    def run(
        self,
        batch_size: Optional[int] = None,
        use_cg: bool = False,
        gtol: float = 1e-2,
        max_iter: Optional[int] = None,
        verbose: Optional[bool] = None,
    ):
        """Estimate MLP coefficients and prediction errors.

        Parameters
        ----------
        batch_size: Batch size for sequential regression.
                    If None, the batch size is automatically determined
                    depending on the memory size and number of features.
        use_cg: CG algorithm is used or not.
        gtol: Gradient tolerance for CG.
        max_iter: Number of maximum iterations in CG.
        """
        if verbose is not None:
            self._verbose = verbose
        if not use_cg:
            self.fit(batch_size=batch_size)
        else:
            # TODO: batch size must be active.
            self.fit_cg(gtol=gtol, max_iter=max_iter)

        self.estimate_error()
        return self

    def fit_learning_curve(self, verbose: Optional[bool] = None):
        """Compute learing curve."""
        if verbose is not None:
            self._verbose = verbose

        self._is_data_none()
        self._learning_log = fit_learning_curve(
            self._params,
            self._train,
            self._test,
            verbose=self._verbose,
        )
        return self

    def save_learning_curve(self, filename: str = "polymlp_learning_curve.dat"):
        """Save learing curve."""
        save_learning_curve_log(self._learning_log, filename=filename)
        return self

    def save_mlp(self, filename: str = "polymlp.yaml"):
        """Save polynomial MLP as file.

        When hybrid models are used, mlp files will be generated as
        filename.1, filename.2, ...
        """
        if self._mlp_model is None:
            raise RuntimeError("No polymlp has been developed.")
        self._mlp_model.save_mlp(filename=filename)
        return self

    def load_mlp(self, filename: Union[str, io.IOBase] = "polymlp.yaml"):
        """Load polynomial MLP from file."""
        # TODO: hybrid is not available.
        self._params, coeffs = load_mlp(filename)
        scales = np.ones(len(coeffs))
        self._mlp_model = PolymlpDataMLP(coeffs=coeffs, scales=scales)
        return self

    def save_params(self, filename: str = "polymlp_params.yaml"):
        """Save MLP parameters as file.

        Parameters
        ----------
        filename: File name for saving parameter attributes.
        """
        self._is_params_none()
        write_polymlp_params_yaml(self._params, filename=filename)
        return self

    def save_errors(self, filename: str = "polymlp_error.yaml"):
        """Save prediction errors as file.

        Parameters
        ----------
        filename: File name for saving errors.
        """
        if self._mlp_model.error_train is None:
            raise RuntimeError("estimate_error must be performed before save_errors.")
        write_error_yaml(self._mlp_model.error_train, filename=filename, mode="w")
        write_error_yaml(self._mlp_model.error_test, filename=filename, mode="a")
        return self

    def split_train_test(self, list_obj: list, train_ratio: float = 0.9):
        """Split list into training and test datasets."""
        return split_train_test(list_obj, train_ratio=train_ratio)

    def convert_to_yaml(
        self,
        filename_txt: str = "polymlp.lammps",
        filename_yaml: str = "polymlp.yaml",
    ):
        """Convert polymlp.lammps to polymlp.yaml.

        Parameters
        ----------
        filename_txt: File name of legacy polymlp file.
        filename_yaml: File name of output polymlp file in yaml format.
        """
        convert_to_yaml(filename_txt, filename_yaml)
        return self

    def get_structures_from_poscars(
        self, poscars: Union[str, list[str]]
    ) -> list[PolymlpStructure]:
        """Load poscar files and convert them to structure instances."""
        return parse_structures_from_poscars(poscars)

    @property
    def summary(self):
        """Return summary of developed polymlp.

        Attributes
        ----------
        coeffs: MLP coefficients.
        scales: Scales of features. scaled_coeffs (= coeffs / scales)
                must be used for calculating properties.
        rmse_train: Root-mean-square error for training data.
        rmse_test: Root-mean-square error for test data.
        alpha, beta: Optimal regurlarization parameters.
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
        return self._mlp_model.scaled_coeffs

    @property
    def learning_curve(self):
        """Return instance of LearningCurve."""
        return self._learning_log

    @property
    def n_features(self):
        """Return number of features."""
        if self._mlp_model is None:
            return get_num_features(self._params)
        return self._mlp_model.coeffs.shape[0]

    @property
    def train(self):
        """Return training datasets."""
        return self._train

    @property
    def test(self):
        """Return test datasets."""
        return self._test

    @property
    def datasets(self):
        """Return training and test datasets."""
        return (self._train, self._test)
