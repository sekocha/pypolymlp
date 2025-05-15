# Python API (Property calculations)

If version 0.8.0 or earlier is used, polymlp files are generated in a text format as `polymlp.lammps`.
If a newer version (0.9.0 or later) is used, polymlp files are generated in YAML format  as `polymlp.yaml`, which can be utilized by replacing `polymlp.lammps` with `polymlp.yaml` in the following documentation.

## Calculations of energy, forces, and stress tensor

- Single structure
```python
import numpy as np
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.calculator.properties import Properties

"""
structure: PolymlpStructure format.
Attributes:
- axis: (3,3), [a, b, c]
- positions: (3, n_atom) [x1, x2, ...]
- n_atoms: [4, 4]
- elements: Element list (e.g.) ['Mg','Mg','Mg','Mg','O','O','O','O']
- types: Atomic type integers (e.g.) [0, 0, 0, 0, 1, 1, 1, 1]
"""
structure = Poscar('POSCAR').structure

"""
energy: unit: eV/supercell
forces: unit: eV/angstrom (3, n_atom)
stress: unit: eV/supercell: (6) in the order of xx, yy, zz, xy, yz, zx
"""
prop = Properties(pot='polymlp.lammps')
energy, forces, stress = prop.eval(structure)
```
or
```python
import numpy as np
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

polymlp = PypolymlpCalc(pot="polymlp.lammps")
polymlp.load_structures_from_files(poscars='POSCAR')
energies, forces, stresses = polymlp.eval()
```

- Multiple structures (Compatible with OPENMP support)
```python
import numpy as np
from pypolymlp.calculator.properties import Properties

"""
energy_all: unit: eV/supercell (n_str)
forces_all: unit: eV/angstrom (n_str, 3, n_atom)
stress_all: unit: eV/supercell: (n_str, 6)
                                in the order of xx, yy, zz, xy, yz, zx
"""
prop = Properties(pot='polymlp.lammps')
energy_all, forces_all, stress_all = prop.eval_multiple(
                                        [structure1, structure2, structure3]
                                    )
```
or
```python
import numpy as np
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

polymlp = PypolymlpCalc(pot="polymlp.lammps")
polymlp.load_structures_from_files(poscars=["POSCAR1", "POSCAR2", "POSCAR3"])
energies, forces, stresses = polymlp.eval()
```

- When using a hybrid polynomial MLP, multiple MLP files should be given as a list.
```python
polymlps = ['polymlp.lammps.1', 'polymlp.lammps.2']
prop = Properties(pot=polymlps)
```
or
```python
polymlps = ['polymlp.lammps.1', 'polymlp.lammps.2']
polymlp_calc = PypolymlpCalc(pot=polymlps)
```


- Property calculations from phonopy structure instances
```python
import numpy as np
from pypolymlp.calculator.properties import Properties

"""
energy_all: unit: eV/supercell (n_str)
forces_all: unit: eV/angstrom (n_str, 3, n_atom)
stress_all: unit: eV/supercell: (n_str, 6)
                                in the order of xx, yy, zz, xy, yz, zx
"""
#phonopy.generate_displacements(distance=0.01)
#supercells = ph.supercells_with_displacements

prop = Properties(pot='polymlp.lammps')
energy_all, forces_all, stress_all = prop.eval_multiple_phonopy(supercells)
```

- Conversion of a phonopy cell instance into a structure dictionary
```python
from pypolymlp.utils.phonopy_utils import phonopy_cell_to_structure
structure = phonopy_cell_to_structure(cell_phonopy)
```

## Local geometry optimizations
```python
import numpy as np
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

polymlp = PypolymlpCalc(pot="polymlp.lammps")
polymlp.load_poscars("POSCAR")

polymlp.init_geometry_optimization(
    with_sym=True,
    relax_cell=True,
    relax_positions=True,
)
e0, n_iter, success = polymlp.run_geometry_optimization()
if success:
    polymlp.save_poscars(filename="POSCAR_CONVERGE")
```

## Structural feature calculations
- Feature calculation using polymlp.in
```python
import numpy as np
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

polymlp = PypolymlpCalc(require_mlp=False)
polymlp.load_structures_from_files(poscars=["POSCAR1", "POSCAR2", "POSCAR3"])
polymlp.run_features(
    develop_infile="polymlp.in",
    features_force=False,
    features_stress=False,
)
polymlp.save_features()
```

- Feature calculation using polymlp.lammps
```python
import numpy as np
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

polymlp = PypolymlpCalc(pot="polymlp.lammps")
polymlp.load_structures_from_files(poscars=["POSCAR1", "POSCAR2", "POSCAR3"])
polymlp.run_features(
    features_force=False,
    features_stress=False,
)
polymlp.save_features()
```


## Force constant calculations
(Requirement: symfc)
- Force constant calculations using a POSCAR file
```python
import numpy as np
from pypolymlp.calculator.fc import PolymlpFC
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.phonopy_utils import phonopy_supercell

unitcell = Poscar('POSCAR').structure
supercell = phonopy_supercell(unitcell, np.diag([3,3,2]))

"""supercell: phonopy structure class"""
polyfc = PolymlpFC(supercell=supercell, pot='polymlp.lammps')

"""optional"""
polyfc.run_geometry_optimization()

polyfc.sample(n_samples=100, displacements=0.001, is_plusminus=False)

"""fc2.hdf5 and fc3.hdf5 will be generated."""
polyfc.run(batch_size=100)
```
or
```python
import numpy as np
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

polymlp = PypolymlpCalc(pot="polymlp.lammps")
polymlp.load_poscars("POSCAR")

"""optional"""
polymlp.init_geometry_optimization(
    with_sym=True,
    relax_cell=False,
    relax_positions=True,
)
polymlp.run_geometry_optimization()

polymlp.init_fc(supercell_matrix=np.diag([3,3,2]), cutoff=None)
polymlp.run_fc(
    n_samples=100,
    distance=0.001,
    is_plusminus=False,
    orders=(2, 3),
    batch_size=100,
    is_compact_fc=True,
    use_mkl=True,
)
polymlp.save_fc()
```

- Force constant calculations using phono3py.yaml.xz
```python
from pypolymlp.calculator.fc import PolymlpFC

polyfc = PolymlpFC(
    phono3py_yaml='phono3py_params_wurtzite_AgI.yaml.xz',
    use_phonon_dataset=False,
    pot='polymlp.lammps',
)

"""optional"""
polyfc.run_geometry_optimization()

"""If not using sample(), displacements are read from phono3py.yaml.xz"""
polyfc.sample(n_samples=100, displacements=0.001, is_plusminus=False)

"""fc2.hdf5 and fc3.hdf5 will be generated."""
polyfc.run(batch_size=100)
```

## Phonon calculations
(Requirement: phonopy)
```python
from pypolymlp.calculator.compute_phonon import (
    PolymlpPhonon, PolymlpPhononQHA,
)

unitcell = Poscar('POSCAR').structure
supercell_matrix = np.diag([3,3,3])
ph = PolymlpPhonon(unitcell, supercell_matrix, pot='polymlp.lammps')
ph.produce_force_constants(displacements=0.01)
ph.compute_properties(
    mesh=[10,10,10],
    t_min=0,
    t_max=1000,
    t_step=10,
    pdos=False
)

qha = PolymlpPhononQHA(unitcell, supercell_matrix, pot='polymlp.lammps')
qha.run()
qha.write_qha()
```
or
```python
import numpy as np
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

polymlp = PypolymlpCalc(pot="polymlp.lammps")
polymlp.load_poscars("POSCAR")
polymlp.init_phonon(supercell_matrix=np.diag([3, 3, 2]))
polymlp.run_phonon(
    distance=0.001,
    mesh=[20, 20, 20],
    t_min=0,
    t_max=1000,
    t_step=10,
    with_eigenvectors=False,
    is_mesh_symmetry=True,
    with_pdos=False,
)
polymlp.write_phonon()

polymlp.run_qha(
    supercell_matrix=np.diag([3, 3, 2]),
    distance=0.001,
    mesh=[20, 20, 20],
    t_min=0,
    t_max=1000,
    t_step=10,
    eps_min=0.8,
    eps_max=1.2,
    eps_step=0.02,
)
polymlp.write_qha()
```

To use phonopy API after producing force constants using polynomial MLPs, phonopy object can be obtained as follows.
```python
unitcell = Poscar('POSCAR').structure
supercell_matrix = np.diag([3,3,3])
ph = PolymlpPhonon(unitcell, supercell_matrix, pot='polymlp.lammps')
ph.produce_force_constants(displacements=0.01)
phonopy = ph.phonopy
```

## Elastic constant calculations
(Requirement: pymatgen)
```python
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.calculator.compute_elastic import PolymlpElastic

unitcell = Poscar('POSCAR').structure
el = PolymlpElastic(unitcell, 'POSCAR', pot='polymlp.lammps')
el.run()
el.write_elastic_constants()
elastic_constants = el.elastic_constants
```
or
```python
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

polymlp = PypolymlpCalc(pot="polymlp.lammps")
polymlp.load_poscars("POSCAR")
polymlp.run_elastic_constants()
polymlp.write_elastic_constants(filename="polymlp_elastic.yaml")
elastic_constants = polymlp.elastic_constants
```

## Equation of states calculation
(Requirement: phonopy)
```python
from pypolymlp.calculator.compute_eos import PolymlpEOS

unitcell = Poscar('POSCAR').structure
eos = PolymlpEOS(unitcell, pot='polymlp.lammps')
eos.run(
    eps_min=0.7, eps_max=2.0, eps_int=0.03, fine_grid=True, eos_fit=True
)
eos.write_eos_yaml(filename='polymlp_eos.yaml')
```
or
```python
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

polymlp = PypolymlpCalc(pot="polymlp.lammps")
polymlp.load_structures_from_files(poscars='POSCAR')
polymlp.run_eos(
    eps_min=0.7,
    eps_max=2.0,
    eps_step=0.03,
    fine_grid=True,
    eos_fit=True,
)
polymlp.write_eos()
energy0, volume0, bulk_modulus = polymlp.eos_fit_data
```

## Molecular dynamics calculations
(Requirement: ASE, phonopy)
```python
from pypolymlp.api.pypolymlp_md import PypolymlpMD

"""
Parameters
----------
temperature : int
    Target temperature (K).
time_step : float
    Time step for MD (fs).
friction : float
    Friction coefficient for Langevin thermostat (1/fs).
n_eq : int
    Number of equilibration steps.
n_steps : int
    Number of production steps.
"""

md = PypolymlpMD(verbose=True)
md.load_poscar("POSCAR")
md.set_supercell([4, 4, 3])

md.set_ase_calculator(pot="polymlp.yaml")
md.run_Langevin(
    temperature=300,
    time_step=1.0,
    friction=0.01,
    n_eq=5000,
    n_steps=20000,
)
md.save_yaml(filename="polymlp_md.yaml")
```

## Thermodynamic integration using MD
(Requirement: ASE, phonopy)
```python
from pypolymlp.api.pypolymlp_md import PypolymlpMD

"""
Parameters
----------
thermostat: Thermostat.
n_alphas: Number of sample points for thermodynamic integration
          using Gaussian quadrature.
temperature : int
    Target temperature (K).
time_step : float
    Time step for MD (fs).
ttime : float
    Timescale of the Nose-Hoover thermostat (fs).
friction : float
    Friction coefficient for Langevin thermostat (1/fs).
n_eq : int
    Number of equilibration steps.
n_steps : int
    Number of production steps.
"""

md = PypolymlpMD(verbose=True)
md.load_poscar("POSCAR")
md.set_supercell([5, 5, 3])

md.set_ase_calculator_with_fc2(pot="polymlp.yaml", fc2hdf5="fc2.hdf5")
md.run_thermodynamic_integration(
    thermostat="Langevin",
    temperature=300.0,
    time_step=1.0,
    friction=0.01,
    n_eq=5000,
    n_steps=20000,
    heat_capacity=False,
)
md.save_thermodynamic_integration_yaml(filename="polymlp_ti.yaml")
```
or
```python
from pypolymlp.api.pypolymlp_md import run_thermodynamic_integration

md = run_thermodynamic_integration(
    pot="polymlp.lammps",
    poscar="POSCAR",
    supercell_size=(5, 5, 3),
    fc2hdf5="fc2.hdf5",
    temperature=300.0,
    n_alphas=15,
    n_eq=5000,
    n_steps=20000,
    heat_capacity=False,
    filename="polymlp_ti.yaml",
)
```

## SSCHA calculations
(Requirement: symfc, phonopy)
```python
import numpy as np
from pypolymlp.api.pypolymlp_sscha import PypolymlpSSCHA

sscha = PypolymlpSSCHA(verbose=True)
supercell_size = [3, 3, 3]
sscha.load_poscar("POSCAR", np.diag(supercell_size))

sscha.set_polymlp("polymlp.yaml")

# Optional if NAC parameters are needed.
sscha.set_nac_params("vasprun.xml")

sscha.run(
    temp_min=100,
    temp_max=1000,
    temp_step=1000,
    ascending_temp=False,
    n_samples_init=None,
    n_samples_final=None,
    tol=0.005,
    max_iter=30,
    mixing=0.5,
    mesh=(10, 10, 10),
    init_fc_algorithm="harmonic",
)
```
