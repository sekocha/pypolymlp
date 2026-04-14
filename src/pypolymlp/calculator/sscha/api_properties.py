"""Class for calculating SSCHA properties."""

from typing import Literal, Optional, Union

import numpy as np

from pypolymlp.api.pypolymlp_sscha import PypolymlpSSCHA
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.units import EVtoKJmol


class PropertiesSSCHA:

    def __init__(
        self,
        pot: Union[str, list],
        properties: Properties,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        pot: Polymlp file name.
        properties: Properties instance.
        """
        self._sscha = PypolymlpSSCHA(verbose=verbose)
        self._sscha._prop = properties
        self._sscha._pot = pot
        self._verbose = verbose

    def eval(
        self,
        structure: PolymlpStructure,
        temp: Optional[float] = None,
        temp_min: float = 0,
        temp_max: float = 2000,
        temp_step: float = 50,
        n_temp: Optional[int] = None,
        ascending_temp: bool = False,
        n_samples_init: Optional[int] = None,
        n_samples_final: Optional[int] = None,
        tol: float = 0.005,
        max_iter: int = 50,
        mixing: float = 0.5,
        mesh: tuple = (10, 10, 10),
        init_fc_algorithm: Literal["harmonic", "const", "random", "file"] = "harmonic",
        init_fc_file: Optional[str] = None,
        fc2: Optional[np.ndarray] = None,
        precondition: bool = True,
        cutoff_radius: Optional[float] = None,
        use_temporal_cutoff: bool = False,
        path: str = "./sscha",
        write_pdos: bool = False,
        use_mkl: bool = True,
    ):
        """Evaluate free energy, forces, and virial stress tensor.

        Return
        ------
        free_energy: SSCHA free energy in eV/cell.
        force: Forces including static forces in eV/angstrom, shape=(3, n_atom).
        stress: Virial stress tensor in eV/cell, shape=(6) for xx, yy, zz, xy, yz, zx.
        """
        self._sscha.unitcell = structure
        self._sscha.supercell_matrix = np.eye(3, dtype=int)

        self._sscha.run(
            temp=temp,
            temp_min=temp_min,
            temp_max=temp_max,
            temp_step=temp_step,
            n_temp=n_temp,
            ascending_temp=ascending_temp,
            n_samples_init=n_samples_init,
            n_samples_final=n_samples_final,
            tol=tol,
            max_iter=max_iter,
            mixing=mixing,
            mesh=mesh,
            init_fc_algorithm=init_fc_algorithm,
            init_fc_file=init_fc_file,
            fc2=fc2,
            precondition=precondition,
            cutoff_radius=cutoff_radius,
            use_temporal_cutoff=use_temporal_cutoff,
            path=path,
            write_pdos=write_pdos,
            use_mkl=use_mkl,
        )
        free_energy = self._sscha.properties.free_energy / EVtoKJmol

        static_forces = self._sscha.properties.static_forces
        average_forces = self._sscha.properties.average_forces
        forces = static_forces + average_forces

        static_stress = self._sscha.properties.static_stress_tensor
        average_stress = self._sscha.properties.average_stress_tensor
        stress = static_stress + average_stress
        return free_energy, forces, stress
