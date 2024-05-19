# A generator of polynomial machine learning potentials

## Polynomial machine learning potentials

- [Polynomial machine learning potential repository](http://cms.mtl.kyoto-u.ac.jp/seko/mlp-repository/index.html)
- [A. Seko, J. Appl. Phys. 133, 011101 (2023)](https://doi.org/10.1063/5.0129045)

## Required libraries and python modules

- numpy
- scipy
- Eigen3 (if building pypolymlp)
- pybind11 (if building pypolymlp)
- phonopy (if using phonon datasets and/or computing force constants)
- phono3py (if using phonon datasets and/or computing force constants)
- symfc (if computing force constants)
- spglib (optional)
- joblib (optional)

## Installing pypolymlp using pip package

- Intel Linux
- Compatible with python 3.9, 3.10, and 3.11

```
> pip install $(pypolymlp)/dist/pypolymlp-0.1-py3-none-any.whl
```

## Manually installing pypolymlp

1. Conda package management system

```
> conda create -n pypolymlp 
> conda activate pypolymlp

> conda install numpy scipy pybind11 eigen cmake
(optional)
> conda install spglib
> conda install phono3py
> conda install joblib
```

2. Building a shared library (libmlpcpp)

The process of building libmlpcpp may take approximately five minutes to one hour. 
```
> cd $(pypolymlp)/src/pypolymlp/cxx
> cmake -S . -B build
> cmake --build build
> cmake --install build
```
or
```
> cd $(pypolymlp)/src/pypolymlp/cxx
> make
```

If necessary, the stack size may need to be set to unlimited.
```
ulimit -s unlimited
```

3. Install pypolymlp using pip

```
> cd $(pypolymlp)
> pip install .
```

## MLP development 

```
> pypolymlp -i polymlp.in
> cat polymlp.in

    n_type 2
    elements Mg O
    
    feature_type gtinv
    cutoff 8.0
    model_type 3
    max_p 2
    
    gtinv_order 3
    gtinv_maxl 4 4
    
    gaussian_params1 1.0 1.0 1
    gaussian_params2 0.0 7.0 8
    
    reg_alpha_params -3 1 5
    
    atomic_energy -0.00040000 -1.85321219
    
    train_data vaspruns/train/vasprun-*.xml.polymlp
    test_data vaspruns/test/vasprun-*.xml.polymlp

    (if using multiple datasets)
    train_data vaspruns/train1/vasprun-*.xml.polymlp
    train_data vaspruns/train2/vasprun-*.xml.polymlp
    test_data vaspruns/test1/vasprun-*.xml.polymlp
    test_data vaspruns/test2/vasprun-*.xml.polymlp

    include_force True
    include_stress True
```

### MLP development using a memory-efficient sequential implementation

```
> pypolymlp -i polymlp.in --sequential
```

### MLP development using hybrid models

```
> pypolymlp -i polymlp1.in polymlp2.in (--sequential)
> cat polymlp1.in

    n_type 2
    elements Mg O
    
    feature_type gtinv
    cutoff 8.0
    model_type 3
    max_p 2
    
    gtinv_order 3
    gtinv_maxl 4 4
    
    gaussian_params1 1.0 1.0 1
    gaussian_params2 0.0 7.0 8
    
    reg_alpha_params -3 1 5
    
    atomic_energy -0.00040000 -1.85321219

    train_data vaspruns/train1/vasprun-*.xml.polymlp
    train_data vaspruns/train2/vasprun-*.xml.polymlp
    test_data vaspruns/test1/vasprun-*.xml.polymlp
    test_data vaspruns/test2/vasprun-*.xml.polymlp
    
    include_force True
    include_stress True

> cat polymlp2.in

    n_type 2
    elements Mg O
    
    feature_type gtinv
    cutoff 4.0
    model_type 3
    max_p 2
    
    gtinv_order 3
    gtinv_maxl 4 4
    
    gaussian_params1 1.0 1.0 1
    gaussian_params2 0.0 3.0 4
```

## Calculators

### Properties (energies, forces, and stress tensors)

```
> pypolymlp --properties --pot polymlp.lammps --poscars */POSCAR
> pypolymlp --properties --pot polymlp.lammps --vaspruns vaspruns/vasprun.xml.polymlp.*
> pypolymlp --properties --pot polymlp.lammps --phono3py_yaml phono3py_params_wurtzite_AgI.yaml.xz
```

### Polynomial structural features

```
> pypolymlp --features --pot polymlp.lammps --poscars */POSCAR
> pypolymlp --features -i polymlp.in --poscars */POSCAR
```

### Force constants 

(phonopy, phono3py, and symfc are required.)
```
> pypolymlp --force_constants --pot polymlp.lammps --phono3py_yaml phono3py_params_wurtzite_AlN.yaml.xz
> pypolymlp --force_constants --pot polymlp.lammps --str_yaml polymlp_str.yaml --fc_n_samples 1000
> pypolymlp --force_constants --pot polymlp.lammps --poscar POSCAR-unitcell --supercell 3 3 2 --fc_n_samples 1000
```

### Phonon calculations

(phonopy is required.)
```
> pypolymlp --phonon --pot polymlp.lammps --poscar POSCAR-unitcell --supercell 3 3 2
```

## Utilities

### Structure generation for DFT calculations

1. Prototype structure selection

<!--
```
> pypolymlp-structure --prototype --n_types 1
> pypolymlp-structure --prototype --n_types 2 --comp 0.25 0.75
> pypolymlp-structure --prototype --n_types 2 --comp 1 3
> pypolymlp-structure --prototype --n_types 3 
```
Only alloy structure types are available. 
Selected prototypes are listed in polymlp_prototypes.yaml.
-->
Prepare prototype structures in POSCAR format.

2. Random structure generation
- Generation from prototype structures
```
> pypolymlp-structure --random --poscars prototypes/* --n_str 10 --low_density 2 --high_density 2
```
Structures for are listed in polymlp_str_samples.yaml. 
Structures are generated in "poscar" directory.

- Generation from a given structure
```
> pypolymlp-structure --random --poscars POSCAR --n_str 10 --low_density 2 --high_density 2
```

- Random displacements for phonon calculations
(phonopy is required.)
```
> pypolymlp-structure --random_phonon --supercell 3 3 2 --n_str 20 --disp 0.03 -p POSCAR
```
Structures are generated in "poscar_phonon" directory.

3. DFT calculations for structures 

### Compression of vasprun.xml files

```
> pypolymlp-utils --vasprun_compress vaspruns/vasprun-*.xml
```
Compressed vasprun.xml is generated as vasprun.xml.polymlp.

### Automatic division of DFT dataset

```
> pypolymlp-utils --auto_dataset dataset1/*/vasprun.xml dataset2/*/vasprun.xml
> cat polymlp.in.append >> polymlp.in
```
A given DFT dataset is automatically divided into some sets, depending on the values of the energy, the forces acting on atoms, and the volume.
A generated file "polymlp.in.append" can be appended in your polymlp.in, which will be used for developing MLPs. 
Datasets identified with "train1" and "test1" are composed of structures with low energy and small force values.
The predictive power for them is more important than the other structures for the successive calculations using polynomial MLPs, so the prediction errors for "train1" and "test1" datasets should be accuracy measures for polynomial MLPs.

### Atomic energies
(Experimental: Only for VASP calculations using PBE and PBEsol functionals)

```
> pypolymlp-utils --atomic_energy_elements Mg O --atomic_energy_functional PBE
> pypolymlp-utils --atomic_energy_formula MgO --atomic_energy_functional PBE
> pypolymlp-utils --atomic_energy_formula Al2O3 --atomic_energy_functional PBEsol
```

A standard output can be appended in your polymlp.in, which will be used for developing MLPs. 
The polynomial MLP has no constant term, which means that the energy for isolated atoms is set to zero.
The energy values for the isolated atoms must be subtracted from the energy values for structures.

### Estimation of computational costs

calc_cost option generates a file 'polymlp_cost.yaml', which is required for finding optimal MLPs.

1. Single polynomial MLP

```
> pypolymlp-utils --calc_cost --pot polymlp.lammps

# hybrid polynomial MLP
> pypolymlp-utils --calc_cost --pot polymlp.lammps*
```

2. Multiple polynomial MLPs

```
> pypolymlp-utils --calc_cost -d $(path_mlps)/polymlp-00*
```

### Enumeration of optimal MLPs on convex hull

```
> pypolymlp-utils --find_optimal Ti-Pb/* --key test-disp1
```

Files 'polymlp_error.yaml' and 'polymlp_cost.yaml' are needed for each MLP.

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

