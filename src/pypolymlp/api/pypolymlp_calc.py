"""API Class for calculating properties."""

from typing import Literal, Optional, Union

import numpy as np

from pypolymlp.calculator.compute_features import (
    compute_from_infile,
    compute_from_polymlp_lammps,
)
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure
from pypolymlp.core.interface_vasp import (
    parse_structures_from_poscars,
    parse_structures_from_vaspruns,
)
from pypolymlp.utils.vasp_utils import write_poscar_file


class PypolymlpCalc:
    """API Class for calculating properties."""

    def __init__(
        self,
        pot: Union[str, list[str]] = None,
        params: Union[PolymlpParams, list[PolymlpParams]] = None,
        coeffs: Union[np.ndarray, list[np.ndarray]] = None,
        properties: Optional[Properties] = None,
        verbose: bool = True,
        require_mlp: bool = True,
    ):
        """Init method.

        Parameters
        ----------
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties instance.

        Any one of pot, (params, coeffs), and properties is needed.
        """
        if require_mlp:
            if pot is None and params is None and properties is None:
                raise RuntimeError("polymlp not defined.")

            if properties is None:
                self._prop = Properties(pot=pot, params=params, coeffs=coeffs)
            else:
                self._prop = properties

        self._verbose = verbose
        self._structures = None

        self._unitcell = None
        self._poscar = None

        self._elastic = None
        self._eos = None
        self._phonon = None
        self._qha = None
        self._go = None
        self._fc = None

    def load_poscars(self, poscars: Union[str, list[str]]) -> list[PolymlpStructure]:
        """Parse POSCAR files.

        Returns
        -------
        structures: list[PolymlpStructure], Structures.
        """
        if isinstance(poscars, str):
            self._poscar = poscars
            poscars = [poscars]
        self.structures = parse_structures_from_poscars(poscars)
        return self.structures

    def load_vaspruns(self, vaspruns: Union[str, list[str]]) -> list[PolymlpStructure]:
        """Parse vasprun files.

        Returns
        -------
        structures: list[PolymlpStructure], Structures.
        """
        if isinstance(vaspruns, str):
            vaspruns = [vaspruns]
        self.structures = parse_structures_from_vaspruns(vaspruns)
        return self.structures

    def load_structures_from_files(
        self,
        poscars: Optional[Union[str, list[str]]] = None,
        vaspruns: Optional[Union[str, list[str]]] = None,
    ):
        """Parse structure files.

        Only POSCAR and vasprun.xml files are available.

        Returns
        -------
        structures: list[PolymlpStructure], Structures.
        """
        if poscars is None and vaspruns is None:
            raise RuntimeError("Structure files not found.")

        if poscars is not None:
            self.structures = self.load_poscars(poscars)
        elif vaspruns is not None:
            self.structures = self.load_vaspruns(vaspruns)
        return self.structures

    def load_phonopy_structures(self, structures_ph):
        """Load structures in phonopy format."""
        from pypolymlp.utils.phonopy_utils import phonopy_cell_to_structure

        self.structures = [phonopy_cell_to_structure(s) for s in structures_ph]

    def save_poscars(self, filename="POSCAR_pypolymlp", prefix="POSCAR"):
        """Save structures to POSCAR files."""
        if len(self.structures) == 1:
            write_poscar_file(self.first_structure, filename=filename)
        else:
            len_zfill = max(np.ceil(np.log10(len(self.structures))).astype(int) + 1, 3)
            for i, st in enumerate(self.structures):
                write_poscar_file(st, filename=prefix + str(i).zfill(len_zfill))
        return self

    def eval(
        self,
        structures: Optional[Union[PolymlpStructure, list[PolymlpStructure]]] = None,
    ):
        """Evaluate properties for a single structure.

        Returns
        -------
        e: Energy. shape=(n_str,), unit: eV/supercell
        f: Forces. shape=(n_str, 3, natom), unit: eV/angstrom.
        s: Stress tensors. shape=(n_str, 6),
            unit: eV/supercell in the order of xx, yy, zz, xy, yz, zx.
        """
        if structures is not None:
            self.structures = structures
        return self._prop.eval_multiple(self.structures)

    def save_properties(self):
        """Save properties.

        Numpy files of polymlp_energies.npy, polymlp_forces.npy,
        and polymlp_stress_tensors.npy are generated.
        They contain the energy values, forces, and stress tensors
        for structures used for the latest run of self.eval.
        """
        self._prop.save(verbose=self._verbose)
        return self

    def print_properties(self):
        """Print properties for a single structure."""
        self._prop.print_single()
        return self

    def run_features(
        self,
        structures: Optional[Union[PolymlpStructure, list[PolymlpStructure]]] = None,
        develop_infile: Optional[str] = None,
        features_force: bool = False,
        features_stress: bool = False,
    ):
        """Compute features.

        Parameters
        ----------
        structures: Structures for computing features.
        develop_infile: A pypolymlp input file for developing MLP.

        Return
        ------
        features: Structural features. shape=(n_str, n_features)
            if features_force == False and features_stress == False.
        """
        if structures is not None:
            self.structures = structures

        if develop_infile is None:
            self._features = compute_from_polymlp_lammps(
                self.structures,
                params=self.params,
                force=features_force,
                stress=features_stress,
                return_mlp_dict=False,
            )
        else:
            self._features = compute_from_infile(
                develop_infile,
                self.structures,
                force=features_force,
                stress=features_stress,
            )

        return self._features

    def save_features(self):
        """Save features."""
        np.save("features.npy", self._features)

    def run_elastic_constants(self, poscar: Optional[str] = None):
        """Run elastic constant calculations.

        pymatgen is required.

        Returns
        -------
        elastic_constants: Elastic constants in GPa. shape=(6,6).
        """
        from pypolymlp.calculator.compute_elastic import PolymlpElastic

        if poscar is not None:
            self.load_poscars(poscar)
            self._poscar = poscar

        self.unitcell = self.first_structure

        self._elastic = PolymlpElastic(
            unitcell=self.unitcell,
            unitcell_poscar=self._poscar,
            properties=self._prop,
            verbose=self._verbose,
        )
        self._elastic.run()
        return self._elastic.elastic_constants

    def write_elastic_constants(self, filename="polymlp_elastic.yaml"):
        """Save elastic constants to a file."""
        self._elastic.write_elastic_constants(filename=filename)

    def run_eos(
        self,
        structure: Optional[PolymlpStructure] = None,
        eps_min: float = 0.7,
        eps_max: float = 2.0,
        eps_step: float = 0.03,
        fine_grid: bool = True,
        eos_fit: bool = False,
    ):
        """Run EOS calculations.

        pymatgen is required if eos_fit = True.

        Parameters
        ----------
        structure: Equilibrium structure.
        eps_min: Lower bound of volume change.
        eps_max: Upper bound of volume change.
        eps_step: Interval of volume change.
        fine_grid: Use a fine grid around equilibrium structure.
        eos_fit: Fit vinet EOS curve using volume-energy data.

        volumes = np.arange(eps_min, eps_max, eps_step) * eq_volume

        Returns
        -------
        self: PolymlpCalc
        """
        from pypolymlp.calculator.compute_eos import PolymlpEOS

        if structure is not None:
            self.structures = structure
        self.unitcell = self.first_structure

        self._eos = PolymlpEOS(
            unitcell=self.unitcell,
            properties=self._prop,
            verbose=self._verbose,
        )
        self._eos.run(
            eps_min=eps_min,
            eps_max=eps_max,
            eps_int=eps_step,
            fine_grid=fine_grid,
            eos_fit=eos_fit,
        )
        return self

    def write_eos(self, filename="polymlp_eos.yaml"):
        """Save EOS to a file."""
        self._eos.write_eos_yaml(filename=filename)

    def init_phonon(
        self,
        unitcell: [PolymlpStructure] = None,
        supercell_matrix: np.ndarray = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ):
        """Initialize phonon calculations.

        phonopy is required.

        Parameters
        ----------
        unitcell: Unit cell of equilibrium structure.
        supercell_matrix: Supercell matrix.

        """
        from pypolymlp.calculator.compute_phonon import PolymlpPhonon

        if unitcell is not None:
            self.structures = unitcell
        self.unitcell = self.first_structure

        self._phonon = PolymlpPhonon(
            unitcell=self.unitcell,
            supercell_matrix=supercell_matrix,
            properties=self._prop,
        )
        return self

    def run_phonon(
        self,
        distance: float = 0.001,
        mesh: np.ndarray = (10, 10, 10),
        t_min: float = 0,
        t_max: float = 1000,
        t_step: float = 10,
        with_eigenvectors: bool = False,
        is_mesh_symmetry: bool = True,
        with_pdos: bool = False,
    ):
        """Run phonon calculations.

        phonopy is required.

        Parameters
        ----------
        distance: Magnitude of displacements in ang.
        mesh: k-mesh grid.
        t_min: Minimum temperature.
        t_max: Maximum temperature.
        t_step: Temperature interval.
        with_eigenvectors: Compute eigenvectors.
        is_mesh_symmetry: Consider symmetry.
        with_pdos: Compute PDOS.

        """
        self._phonon.produce_force_constants(distance=distance)
        self._phonon.compute_properties(
            mesh=mesh,
            t_min=t_min,
            t_max=t_max,
            t_step=t_step,
            with_eigenvectors=with_eigenvectors,
            is_mesh_symmetry=is_mesh_symmetry,
            with_pdos=with_pdos,
        )
        return self

    def write_phonon(self, path="./"):
        """Save results from phonon calculations."""
        self._phonon.write_properties(path_output=path)
        return self

    def run_qha(
        self,
        unitcell: [PolymlpStructure] = None,
        supercell_matrix: np.ndarray = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        distance: float = 0.001,
        mesh: np.ndarray = (10, 10, 10),
        t_min: float = 0,
        t_max: float = 1000,
        t_step: float = 10,
        eps_min: float = 0.8,
        eps_max: float = 1.2,
        eps_step: float = 0.02,
    ):
        """Initialize and run QHA phonon calculations.

        phonopy is required.

        Parameters
        ----------
        unitcell: Unit cell of equilibrium structure.
        supercell_matrix: Supercell matrix.
        distance: Magnitude of displacements in ang.
        mesh: k-mesh grid.
        t_min: Minimum temperature.
        t_max: Maximum temperature.
        t_step: Temperature interval.
        eps_min: Minimum volume change.
        eps_max: Maximum volume change.
        eps_step: Volume change interval.
            volumes = np.arange(eps_min, eps_max + 0.001, eps_step) * vol_eq

        """
        from pypolymlp.calculator.compute_phonon import PolymlpPhononQHA

        if unitcell is not None:
            self.structures = unitcell
        self.unitcell = self.first_structure

        self._qha = PolymlpPhononQHA(
            unitcell=self.unitcell,
            supercell_matrix=supercell_matrix,
            properties=self._prop,
        )

        self._qha.run(
            distance=distance,
            mesh=mesh,
            t_min=t_min,
            t_max=t_max,
            t_step=t_step,
            eps_min=eps_min,
            eps_max=eps_max,
            eps_step=eps_step,
        )
        return self

    def write_qha(self, path="./"):
        """Save results from QHA phonon calculations."""
        self._qha.write_qha(path_output=path)
        return self

    def init_geometry_optimization(
        self,
        init_str: Optional[PolymlpStructure] = None,
        with_sym: bool = True,
        relax_cell: bool = False,
        relax_volume: bool = False,
        relax_positions: bool = True,
        pressure: float = 0.0,
    ):
        """Initialize geometry optimization.

        symfc is required if with_sym = True.

        Parameters
        ----------
        init_str: Initial structure.
        with_sym: Consider symmetry.
        relax_cell: Relax cell.
        relax_volume: Relax volume.
        relax_positions: Relax atomic positions.
        pressure: Pressure in GPa.
        """
        from pypolymlp.calculator.str_opt.optimization import GeometryOptimization

        if init_str is not None:
            self.structures = init_str
        init_str = self.first_structure

        try:
            self._go = GeometryOptimization(
                init_str,
                properties=self._prop,
                relax_cell=relax_cell,
                relax_volume=relax_volume,
                relax_positions=relax_positions,
                with_sym=with_sym,
                pressure=pressure,
                verbose=self._verbose,
            )
        except ValueError:
            self._go = None
            if self._verbose:
                print("Warning: No degrees of freedom in structure.", flush=True)
        return self

    def run_geometry_optimization(
        self,
        method: Literal["BFGS", "CG", "L-BFGS-B", "SLSQP"] = "BFGS",
        gtol: float = 1e-4,
        maxiter: int = 1000,
        c1: Optional[float] = None,
        c2: Optional[float] = None,
    ):
        """Run geometry optimization.

        Parameters
        ----------
        method: Optimization method, CG, BFGS, L-BFGS-B, or SLSQP.
                If relax_volume = False, SLSQP is automatically used.
        gtol: Tolerance for gradients.
        maxiter: Maximum iteration in scipy optimization.
        c1: c1 parameter in scipy optimization.
        c2: c2 parameter in scipy optimization.

        Returns
        -------
        energy: Energy at the final iteration.
        n_iter: Number of iterations required for convergence.
        success: Return True if optimization finished successfully.
        """
        if self._go is None:
            return (None, None, None)

        if self._verbose:
            print("Initial structure", flush=True)
            self._go.print_structure()

        self._go.run(method=method, gtol=gtol, maxiter=maxiter, c1=c1, c2=c2)
        self.structures = self._go.structure
        if self._verbose:
            if not self._go._relax_cell:
                print("Residuals (force):", flush=True)
                print(self._go.residual_forces.T, flush=True)
            else:
                res_f, res_s = self._go.residual_forces
                print("Residuals (force):", flush=True)
                print(res_f.T, flush=True)
                print("Residuals (stress):", flush=True)
                print(res_s, flush=True)
            print("Final structure", flush=True)
            self._go.print_structure()

        return (self._go.energy, self._go.n_iter, self._go.success)

    def init_fc(
        self,
        unitcell: [PolymlpStructure] = None,
        supercell_matrix: np.ndarray = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        cutoff: float = None,
    ):
        """Initialize force constant calculations.

        symfc and phonopy is required.

        Parameters
        ----------
        unitcell: Unit cell of equilibrium structure.
        supercell_matrix: Supercell matrix. (Only diagonal elements are valid.)
        cutoff: Cutoff distance for force constant calculation.

        """
        from pypolymlp.calculator.fc import PolymlpFC
        from pypolymlp.utils.phonopy_utils import phonopy_supercell

        if unitcell is not None:
            self.structures = unitcell
        self.unitcell = self.first_structure

        supercell_matrix_diag = np.diag(supercell_matrix)
        supercell = phonopy_supercell(self.unitcell, supercell_matrix_diag)

        self._fc = PolymlpFC(
            supercell=supercell,
            properties=self._prop,
            cutoff=cutoff,
            verbose=self._verbose,
        )
        return self

    def run_fc(
        self,
        disps: Optional[np.ndarray] = None,
        forces: Optional[np.ndarray] = None,
        n_samples: int = 100,
        distance: float = 0.001,
        is_plusminus: bool = False,
        orders: list = (2, 3),
        batch_size: int = 100,
        is_compact_fc: bool = True,
        use_mkl: bool = True,
    ):
        """Run force constant calculations.

        symfc and phonopy is required. If displacements and forces are not given,
        n_samples supercells are sampled and forces are calculated using polymlp.

        Parameters
        ----------
        disps: Displacements. shape=(n_str, 3, n_atom).
        forces: Forces. shape=(n_str, 3, n_atom).
        n_samples: Number of supercells sampled.
        distance: Displacement magnitude in angstroms.
        is_plusminus: Consider plus and minus displacements.
        orders: Force constant orders.
        batch_size: Batch size for force constant regression.
        is_compact_fc: Generate compact forms of force constants.
        use_mkl: Use MKL in symfc.

        """
        if disps is None or forces is None:
            self._fc.sample(
                n_samples=n_samples,
                displacements=distance,
                is_plusminus=is_plusminus,
            )

        self._fc.run(
            orders=orders,
            batch_size=batch_size,
            write_fc=False,
            is_compact_fc=is_compact_fc,
            use_mkl=use_mkl,
        )

    def save_fc(self):
        """Save force constants."""
        self._fc.save_fc()

    @property
    def instance_properties(self) -> Properties:
        """Return Properties instance."""
        return self._prop

    @property
    def params(self) -> PolymlpParams:
        """Return parameters."""
        return self._prop.params

    @property
    def energies(self) -> np.ndarray:
        """Return energies from the final calculation."""
        return self._prop.energies

    @property
    def forces(self) -> list:
        """Return forces from the final calculation."""
        return self._prop.forces

    @property
    def stresses(self) -> np.ndarray:
        """Return stress tensors from the final calculation."""
        return self._prop.stresses

    @property
    def stresses_gpa(self) -> np.ndarray:
        """Return stress tensors in GPa from the final calculation."""
        return self._prop.stresses_gpa

    @property
    def structures(self) -> list[PolymlpStructure]:
        """Return structures for the final calculation."""
        return self._structures

    @property
    def first_structure(self) -> PolymlpStructure:
        """Return the first structure for the final calculation."""
        return self._structures[0]

    @structures.setter
    def structures(
        self, structures: Union[PolymlpStructure, list[PolymlpStructure]]
    ) -> list[PolymlpStructure]:
        """Set structures."""
        if isinstance(structures, PolymlpStructure):
            self._structures = [structures]
        elif isinstance(structures, list):
            self._structures = structures
        else:
            raise RuntimeError("Invalid structure type.")

    @property
    def unitcell(self) -> PolymlpStructure:
        """Return unit cell."""
        return self._unitcell

    @unitcell.setter
    def unitcell(self, cell):
        """Set unit cell."""
        self._unitcell = cell

    @property
    def features(self) -> PolymlpStructure:
        """Return features."""
        return self._features

    @property
    def elastic_constants(self) -> np.ndarray:
        """Return elastic constants."""
        return self._elastic.elastic_constants

    @property
    def eos_fit_data(self):
        """Return EOS fit parameters.

        Returns
        -------
        equilibrium energy, equilibrium volume, bulk modulus
        """
        return (self._eos._e0, self._eos._v0, self._eos._b0)

    @property
    def go_data(self):
        """Return results of geometry optimization.

        Returns
        -------
        energy: Energy at the final iteration.
        n_iter: Number of iterations required for convergence.
        success: Return True if optimization finished successfully.
        """
        return (self._go.energy, self._go.n_iter, self._go.success)

    @property
    def instance_phonopy(self):
        return self._phonon.phonopy


# def set_structures(args):
#
#     if args.phono3py_yaml is not None:
#         from pypolymlp.core.interface_phono3py import (
#             parse_structures_from_phono3py_yaml,
#         )
#
#         print("Loading", args.phono3py_yaml)
#         if args.phono3py_yaml_structure_ids is not None:
#             r1, r2 = args.phono3py_yaml_structure_ids
#             select_ids = np.arange(r1, r2)
#         else:
#             select_ids = None
#
#         structures = parse_structures_from_phono3py_yaml(
#             args.phono3py_yaml, select_ids=select_ids
#         )
#
#     return structures
#
