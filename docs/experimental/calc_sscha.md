# SSCHA calculations
`symfc` and `phonopy` are required for performing SSCHA calculations.
Pypolymlp adopts an iterative procedure for SSCHA calculations, in which property calculations using polynomial MLP and force constant esimation using linear regression are repeated until the force constants converge.

## Using command line interface
### Single SSCHA calculation

SSCHA calculations for a structure specified by `POSCAR` using polynomial MLP `polymlp.yaml` can be performed as follows. If `--n_samples` option is not provided, the number of sample structures is automatically determined.

```shell
> pypolymlp-sscha --poscar POSCAR --pot polymlp.yaml --supercell 3 3 2 --temp_min 100 --temp_max 700 --temp_step 100 --mixing 0.5 --ascending_temp --n_samples 3000 6000

# Number of sample structures is automatically determined.
> pypolymlp-sscha --poscar POSCAR --pot polymlp.yaml --supercell 3 3 2 --temp_min 100 --temp_max 700 --temp_step 100 --mixing 0.5
```
If SSCHA calculations are successfully finished, `sscha_results.yaml` and effective force constants `fc2.hdf5` are generated for each temperature.

### Generation of random structures from SSCHA force constants
Random structures are generated based on the density matrix determined by the given effective force constants. The energy and force values for these structures are then calculated using the provided MLP.
```shell
pypolymlp-sscha-post --distribution --yaml sscha_results.yaml --fc2 fc2.hdf5 --n_samples 20 --pot polymlp.yaml
```

## Using Python API
### Single SSCHA calculation
```python
import numpy as np
from pypolymlp.api.pypolymlp_sscha import PypolymlpSSCHA

sscha = PypolymlpSSCHA(verbose=True)
supercell_size = [3, 3, 3]
sscha.load_poscar("POSCAR", np.diag(supercell_size))

sscha.set_polymlp("polymlp.yaml")

# Optional function if NAC parameters are needed.
sscha.set_nac_params("vasprun.xml")

"""
Parameters
----------
temp: Single simulation temperature.
temp_min: Minimum temperature.
temp_max: Maximum temperature.
temp_step: Temperature interval.
ascending_temp: Set simulation temperatures in ascending order.
n_samples_init: Number of samples in first loop of SSCHA iterations.
                If None, the number of samples is automatically determined.
n_samples_final: Number of samples in second loop of SSCHA iterations.
                If None, the number of samples is automatically determined.
tol: Convergence tolerance for FCs.
max_iter: Maximum number of iterations.
mixing: Mixing parameter.
        FCs are updated by FC2 = FC2(new) * mixing + FC2(old) * (1-mixing).
mesh: q-point mesh for computing harmonic properties using effective FC2.
init_fc_algorithm: Algorithm for generating initial FCs.
init_fc_file: If algorithm = "file", coefficients are read from init_fc_file.
fc2: FC2 used for initial force constants if it is not None.
cutoff_radius: Cutoff radius used for estimating FC2.
"""
sscha.run(
    temp_min=0,
    temp_max=1000,
    temp_step=100,
    ascending_temp=False,
    n_samples_init=None,
    n_samples_final=None,
    tol=0.005,
    max_iter=30,
    mixing=0.5,
    mesh=(10, 10, 10),
    init_fc_algorithm="harmonic",
    fc2=None,
    cutoff_radius=None,
)

"""
Attributes
----------
force_constants: FC2 at the final temperature.
                shape=(n_atom, n_atom, 3, 3).
"""
fc2 = sscha.force_constants
```

### Generation of random structures from SSCHA force constants

Random stuctures can be sampled from a converged force constants as follows.
The energy and force values of these random structures are also evaluated using a given polynomial MLP.

```python
from pypolymlp.api.pypolymlp_sscha_post import PypolymlpSSCHAPost

sscha = PypolymlpSSCHAPost(verbose=True)
sscha.init_structure_distribution(
    yamlfile="./sscha/1000/sscha.yaml",
    fc2file="./sscha/1000/fc2.hdf5",
    pot="polymlp.yaml",
)
sscha.run_structure_distribution(n_samples=1000)
sscha.save_structure_distribution(path=".", save_poscars=False)

"""
Attributes
----------
displacements: Displacements in structures sampled from density matrix.
               shape = (n_supercell, 3, n_atom), in Angstroms.
forces: Forces of structures sampled from density matrix.
        shape = (n_supercell, 3, n_atom), in eV/Angstroms.
energies: Energies of structures sampled from density matrix.
        shape = (n_supercell), in eV/supercell.
static_potential: Static potential of equilibrium supercell structure in eV/supercell.

supercells: Supercell structures sampled from density matrix.
unitcell: Unitcell structure.
"""
energies = sscha.energies
supercells = sscha.supercells
```
