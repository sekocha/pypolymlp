# Phonon calculations
`phonopy` is required.

## Using command line interface
```shell
> pypolymlp-calc --phonon --pot polymlp.yaml --poscar POSCAR --supercell 3 3 2 --ph_mesh 20 20 20
```

## Using Python API
```python
import numpy as np
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

polymlp = PypolymlpCalc(pot="polymlp.yaml")
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

To use phonopy API after producing force constants using polynomial MLPs, phonopy instance can be obtained as follows.
```python
unitcell = Poscar('POSCAR').structure
supercell_matrix = np.diag([3,3,3])
ph = PolymlpPhonon(unitcell, supercell_matrix, pot='polymlp.yaml')
ph.produce_force_constants(displacements=0.01)
phonopy = ph.phonopy
```
