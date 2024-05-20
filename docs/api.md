## Python API
### MLP development

- MLP development using polymlp.in
```
import numpy as np
from pypolymlp.api.pypolymlp import Pypolymlp

polymlp = Pypolymlp()
polymlp.run(file_params='polymlp.in', log=True)

params_dict = polymlp.parameters
mlp_dict = polymlp.summary
```

- MLP development from vasprun.xml files without using polymlp.in
```
import numpy as np
import glob
from pypolymlp.api.pypolymlp import Pypolymlp

'''
Parameters
----------
elements: Element species, (e.g., ['Mg','O'])
include_force: Considering force entries
include_stress: Considering stress entries
cutoff: Cutoff radius (Angstrom)
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
reg_alpha_params: Parameters for penalty term in
    linear ridge regression. Parameters are given as
    np.linspace(p[0], p[1], p[2]).
gtinv_order: Maximum order of polynomial invariants.
gtinv_maxl: Maximum angular numbers of polynomial invariants.
    [maxl for order=2, maxl for order=3, ...]
atomic_energy: Atomic energies (in eV).
rearrange_by_elements: Set True if not developing special MLPs.
'''

polymlp = Pypolymlp()
polymlp.set_params(
    ['Mg','O'],
    cutoff=8.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=[4,4],
    gaussian_params2=[0.0,7.0,8],
    atomic_energy=[-0.00040000,-1.85321219],
)
train_vaspruns = glob.glob('vaspruns/train/vasprun-*.xml.polymlp')
test_vaspruns = glob.glob('vaspruns/test/vasprun-*.xml.polymlp')
polymlp.set_datasets_vasp(train_vaspruns, test_vaspruns)
polymlp.run(log=True)
```

- MLP development from phono3py.yaml.xz without using polymlp.in
```
from pypolymlp.api.pypolymlp import Pypolymlp

polymlp = Pypolymlp()
polymlp.set_params(
    ['Ag','I'],
    cutoff=8.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=[4,4],
    gaussian_params2=[0.0,7.0,10],
    atomic_energy=[-0.19820116,-0.21203241],
)
train_yaml = 'phono3py_params_wurtzite_AgI.yaml.xz'
test_yaml = 'phono3py_params_wurtzite_AgI.yaml.xz'
train_energy_dat = 'energies_ltc_wurtzite_AgI_fc3-forces.dat'
test_energy_dat = 'energies_ltc_wurtzite_AgI_fc3-forces.dat'
train_ids = np.arange(20)
test_ids = np.arange(380,400)

polymlp.set_datasets_phono3py(
    train_yaml,
    train_energy_dat,
    test_yaml,
    test_energy_dat,
    train_ids=train_ids,
    test_ids=test_ids,
)
polymlp.run(log=True)
```

- MLP development using displacements and forces
```
from pypolymlp.api.pypolymlp import Pypolymlp

polymlp = Pypolymlp()
polymlp.set_params(
    ['Ag','I'],
    cutoff=8.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=[4,4],
    gaussian_params2=[0.0,7.0,10],
    atomic_energy=[-0.19820116,-0.21203241],
)

'''
Parameters in polymlp.set_datasets_displacements
-------------------------------------------------
train_disps: (n_train, 3, n_atoms)
train_forces: (n_train, 3, n_atoms)
train_energies: (n_train)
test_disps: (n_test, 3, n_atom)
test_forces: (n_test, 3, n_atom)
test_energies: (n_test)
'''

polymlp.set_datasets_displacements(
    train_disps,
    train_forces,
    train_energies,
    test_disps,
    test_forces,
    test_energies,
    st_dict,
)
polymlp.run(log=True)
```

### Calculations of energy, forces, and stress tensor

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

### Force constant calculations
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

