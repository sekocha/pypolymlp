# Python API (MLP development)

## MLP development using polymlp.in
```python
import numpy as np
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

polymlp = Pypolymlp()
polymlp.load_parameter_file("polymlp.in")
polymlp.load_datasets()
polymlp.run(verbose=True)

polymlp.save_mlp(filename="polymlp.yaml")

params = polymlp.parameters
mlp_info = polymlp.summary
```

## MLP development from vasprun.xml files without using polymlp.in
```python
import numpy as np
import glob
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

"""
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
"""

polymlp = Pypolymlp()
polymlp.set_params(
    elements=['Mg','O'],
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
polymlp.run(verbose=True)
polymlp.save_mlp(filename="polymlp.yaml")
```

## MLP development from phono3py.yaml.xz without using polymlp.in

```python
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

polymlp = Pypolymlp()
polymlp.set_params(
    elements=['Ag','I'],
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
    test_yaml,
    train_energy_dat=train_energy_dat,
    test_energy_dat=test_energy_dat,
    train_ids=train_ids,
    test_ids=test_ids,
)
polymlp.run(verbose=True)
polymlp.save_mlp(filename="polymlp.yaml")
```

When energy values are read from phono3py.yaml.xz, train_energy_dat and test_energy dat are not required as follows.
```python
train_yaml = 'phono3py_params_wurtzite_AgI.yaml.xz'
test_yaml = 'phono3py_params_wurtzite_AgI.yaml.xz'
train_ids = np.arange(20)
test_ids = np.arange(380,400)

polymlp.set_datasets_phono3py(
    train_yaml,
    test_yaml,
    train_ids=train_ids,
    test_ids=test_ids,
)
polymlp.run(verbose=True)
polymlp.save_mlp(filename="polymlp.yaml")
```

## MLP development using POSCAR files

```python
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

polymlp = Pypolymlp()
polymlp.set_params(
    elements=['Ag','I'],
    cutoff=8.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=[4,4],
    gaussian_params2=[0.0,7.0,10],
    atomic_energy=[-0.19820116,-0.21203241],
)

train_poscars = glob.glob('poscars/train/POSCAR-*')
train_structures = polymlp.get_structures_from_poscars(train_poscars)

test_poscars = glob.glob('poscars/test/POSCAR-*')
test_structures = polymlp.get_structures_from_poscars(test_poscars)

"""
DFT values must be prepared by the following settings.

train_energies: shape=(n_train), unit: eV/cell.
test_energies: shape=(n_test), unit: eV/cell.
train_forces: shape=(n_train, (3, n_atom)), unit: eV/ang.
test_forces: shape=(n_test, (3, n_atom)), unit: eV/ang.
train_stresses: shape=(n_train, 3, 3), unit: eV/cell.
test_stresses: shape=(n_test, 3, 3), unit: eV/cell.
"""

polymlp.set_datasets_structures(
    train_structures = train_structures,
    test_structures = test_structures,
    train_energies = train_energies,
    test_energies = test_energies,
    train_forces = train_forces,
    test_forces = test_forces,
    train_stresses = train_stresses,
    test_stresses = test_stresses,
)
polymlp.run(verbose=True)
polymlp.save_mlp(filename="polymlp.yaml")
```


## MLP development using displacements and forces

```python
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp
from pypolymlp.core.data_format import PolymlpStructure

polymlp = Pypolymlp()
polymlp.set_params(
    elements=['Ag','I'],
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

structure_without_disp: supercell structure without displacements, PolymlpStructure format.
(attributes)
- axis: (3,3), [a, b, c]
- positions: (3, n_atom) [x1, x2, ...]
- n_atoms: [4, 4]
- elements: Element list (e.g.) ['Mg','Mg','Mg','Mg','O','O','O','O']
- types: Atomic type integers (e.g.) [0, 0, 0, 0, 1, 1, 1, 1]
- volume: 64.0 (ang.^3)

'''
structure_without_disp = PolymlpStructure(
    axis = axis,
    positions = positions,
    n_atoms = n_atoms,
    elements = elements,
    types = types,
)

'''
Structure can also be generated from POSCAR as follows.
'''
from pypolymlp.core.interface_vasp import Poscar
structure_without_disp = Poscar('POSCAR').structure

polymlp.set_datasets_displacements(
    train_disps,
    train_forces,
    train_energies,
    test_disps,
    test_forces,
    test_energies,
    structure_without_disp,
)
polymlp.run(verbose=True)
polymlp.save_mlp(filename="polymlp.yaml")
```

## MLP development using a dataset of structure and properties.

```python
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

polymlp = Pypolymlp()
polymlp.set_params(
    elements=['Ag','I'],
    cutoff=8.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=[4,4],
    gaussian_params2=[0.0,7.0,10],
    atomic_energy=[-0.19820116,-0.21203241],
)

'''
Parameters in polymlp.set_datasets_structures
-------------------------------------------------
train_structures: shape=(n_train), list of PolymlpStructure.
train_energies: shape=(n_test), list of PolymlpStructure.
test_energies: shape=(n_test), unit: eV/cell.
train_forces: shape=(n_train, (3, n_atom)), unit: eV/ang.
test_forces: shape=(n_test, (3, n_atom)), unit: eV/ang.
train_stresses: shape=(n_train, 3, 3), unit: eV/cell.
test_stresses: shape=(n_test, 3, 3), unit: eV/cell.

Each structure must be provided in the `PolymlpStructure` format as

structure = PolymlpStructure(
    axis = axis,
    positions = positions,
    n_atoms = n_atoms,
    elements = elements,
    types = types,
)

(attributes)
- axis: shape=(3, 3), [a, b, c]
- positions: shape=(3, n_atom) [x1, x2, ...]
- n_atoms: [4, 4]
- elements: Element list (e.g.) ['Mg','Mg','Mg','Mg','O','O','O','O']
- types: Atomic type integers (e.g.) [0, 0, 0, 0, 1, 1, 1, 1]
- volume: 64.0 (ang.^3)
'''

polymlp.set_datasets_structures(
    train_structures = train_structures,
    test_structures = test_structures,
    train_energies = train_energies,
    test_energies = test_energies,
    train_forces = train_forces,
    test_forces = test_forces,
    train_stresses = train_stresses,
    test_stresses = test_stresses,
)
polymlp.run(verbose=True)
polymlp.save_mlp(filename="polymlp.yaml")
```

If force or stress tensor data are not available, the corresponding input lines may be omitted.
```python
polymlp.set_datasets_structures(
    train_structures = train_structures,
    test_structures = test_structures,
    train_energies = train_energies,
    test_energies = test_energies,
    train_forces = train_forces,
    test_forces = test_forces,
)
polymlp.run(verbose=True)
```

## From multiple sets of vasprun.xml files

```python
train_vaspruns1 = glob.glob('vaspruns/train1/vasprun-*.xml.polymlp')
train_vaspruns2 = glob.glob('vaspruns/train2/vasprun-*.xml.polymlp')
test_vaspruns1 = glob.glob('vaspruns/test1/vasprun-*.xml.polymlp')
test_vaspruns2 = glob.glob('vaspruns/test2/vasprun-*.xml.polymlp')
polymlp.set_multiple_datasets_vasp(
    [train_vaspruns1, train_vaspruns2],
    [test_vaspruns1, test_vaspruns2]
)

polymlp.run(verbose=True)
```

## File IO
- Save polynomial MLP to a file.
```python
polymlp.save_mlp(filename="polymlp.yaml")
```

- Save energy predictions for training and test datasets.
When using `polymlp.run`, energy prediction files are not generated by default.
Instead, use the `fit` and `estimate_error` commands to generate energy predictions for the training and test datasets, as shown below.
```python
polymlp.fit(verbose=True)
polymlp.estimate_error(log_energy=True, verbose=True)
```

- Save RMS errors for training and test datasets.
```python
polymlp.save_errors(filename="polymlp_error.yaml")
```

- Save parameters used for developing polymlp.
```python
polymlp.save_params(filename="polymlp_params.yaml")
```
