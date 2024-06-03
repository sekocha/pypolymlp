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
from pypolymlp.symfc.dev.compute_fcs_class_dev import PolymlpFC

polyfc = PolymlpFC(
    phono3py_yaml='phono3py_params_wurtzite_AgI.yaml.xz',
    use_phonon_dataset=False,
    pot='polymlp.lammps',
)

'''optional'''
polyfc.run_geometry_optimization()

'''If not using sample(), displacements are read from phono3py.yaml.xz'''
polyfc.sample(n_samples=100, displacements=0.001, is_plusminus=False)

'''fc2.hdf5 and fc3.hdf5 will be generated.'''
polyfc.run(batch_size=100)
```  

- Force constant calculations using a POSCAR file
```  
import numpy as np
from pypolymlp.symfc.dev.compute_fcs_class_dev import PolymlpFC
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.phonopy_utils import phonopy_supercell

unitcell_dict = Poscar('POSCAR').get_structure()
supercell = phonopy_supercell(unitcell_dict, np.diag([3,3,2]))

polyfc = PolymlpFC(supercell=supercell, pot='polymlp.lammps')

'''optional'''
polyfc.run_geometry_optimization()

polyfc.sample(n_samples=100, displacements=0.001, is_plusminus=False)

'''fc2.hdf5 and fc3.hdf5 will be generated.'''
polyfc.run(batch_size=100)

```  

