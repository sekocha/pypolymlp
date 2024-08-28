# Python API (Property calculations)
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

- When using a hybrid polynomial MLP, multiple MLP files should be given as a list.
```python
polymlps = ['polymlp.lammps.1', 'polymlp.lammps.2']
prop = Properties(pot=polymlps)
```

- Property calculations from phonopy structure objects
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

- Conversion of a phonopy cell class object into a structure dictionary
```python
from pypolymlp.utils.phonopy_utils import phonopy_cell_to_structure
structure = phonopy_cell_to_structure(cell_phonopy)
```


## Force constant calculations
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

## Phonon calculations
(Required: phonopy)
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

To use phonopy API after producing force constants using polynomial MLPs, phonopy object can be obtained as follows.
```python
unitcell = Poscar('POSCAR').structure
supercell_matrix = np.diag([3,3,3])
ph = PolymlpPhonon(unitcell, supercell_matrix, pot='polymlp.lammps')
ph.produce_force_constants(displacements=0.01)
phonopy = ph.phonopy
```

## Elastic constant calculations
(Required: pymatgen)
```python
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.calculator.compute_elastic import PolymlpElastic

unitcell = Poscar('POSCAR').structure
el = PolymlpElastic(unitcell, 'POSCAR', pot='polymlp.lammps')
el.run()
el.write_elastic_constants()
elastic_constants = el.elastic_constants
```

## Equation of states calculation
(Required: pymatgen)
```python
from pypolymlp.calculator.compute_eos import PolymlpEOS

unitcell = Poscar('POSCAR').structure
eos = PolymlpEOS(unitcell, pot='polymlp.lammps')
eos.run(
    eps_min=0.7, eps_max=2.0, eps_int=0.03, fine_grid=True, eos_fit=True
)
eos.write_eos_yaml(filename='polymlp_eos.yaml')
```

## SSCHA calculations
(Required: phonopy)
Coming soon.
