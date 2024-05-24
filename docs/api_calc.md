# Python API (Property calculations)
## Calculations of energy, forces, and stress tensor

- Single structure
```
import numpy as np
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.calculator.properties import Properties

'''
str_dict: dictionary
- 'axis': (3,3), [a, b, c] 
- 'positions': (3, n_atom) [x1, x2, ...]
- 'n_atoms': [4, 4]
- 'elements': Element list (e.g.) ['Mg','Mg','Mg','Mg','O','O','O','O']
- 'types': Atomic type integers (e.g.) [0, 0, 0, 0, 1, 1, 1, 1]
- 'volume': 64.0 (ang.^3)
'''
str_dict = Poscar('POSCAR').get_structure()

prop = Properties(pot='polymlp.lammps')

'’'
energy: unit: eV/supercell
forces: unit: eV/angstrom (3, n_atom)
stress: unit: eV/supercell: (6) in the order of xx, yy, zz, xy, yz, zx
'’’
energy, forces, stress = prop.eval(str_dict)
```

- Multiple structures (Compatible with OPENMP support)
```
'’'
energy_all: unit: eV/supercell (n_str)
forces_all: unit: eV/angstrom (n_str, 3, n_atom)
stress_all: unit: eV/supercell: (n_str, 6) 
                                in the order of xx, yy, zz, xy, yz, zx
'’’
import numpy as np
from pypolymlp.calculator.properties import Properties

prop = Properties(pot='polymlp.lammps')
energy_all, forces_all, stress_all = prop.eval_multiple(
                                        [str_dict1, str_dict2, str_dict3]
                                    )
```

- Conversion of a phonopy cell class object into a structure dictionary 
```
from pypolymlp.utils.phonopy_utils import phonopy_cell_to_st_dict
st_dict = phonopy_cell_to_st_dict(cell_phonopy)
```

- When using a hybrid polynomial MLP, multiple MLP files should be given as a list.
```
polymlps = ['polymlp.lammps.1', 'polymlp.lammps.2']
prop = Properties(pot=polymlps)
```

## Force constant calculations
- Force constant calculations using phono3py.yaml.xz
```  
from pypolymlp.api.pypolymlp_fc import PypolymlpFC

polymlp_fc = PypolymlpFC('polymlp.lammps')
yaml = 'phono3py_params_wurtzite_AgI.yaml.xz'
polymlp_fc.compute_fcs_phono3py_yaml(yaml)
```  

- Force constant calculations using a structure
```  
import numpy as np
from pypolymlp.api.pypolymlp_fc import PypolymlpFC

polymlp_fc = PypolymlpFC('polymlp.lammps')
unitcell_dict = polymlp_fc.parse_poscar('POSCAR-unitcell')
supercell_matrix = np.diag([3,3,2])
polymlp_fc.compute_fcs(unitcell_dict=unitcell_dict,
                       supercell_matrix=supercell_matrix,
                       n_samples=1000,
                       displacements=0.03)
```  

