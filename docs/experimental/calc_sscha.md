# Self-Consistent Phonon Calculations

> **Note**: Requires `symfc` and `phonopy`.

`pypolymlp` supports stochastic self-consistent harmonic approximation (SSCHA) calculations to evaluate lattice vibrational properties, including anharmonic contributions effectively.
`pypolymlp` employs an iterative procedure for SSCHA calculations, in which property evaluations (energy, forces, and stress tensors) for sampled structures using a polynomial MLP and the estimation of effective force constants via linear regression are performed repeatedly.
This procedure is repeated until the effective force constants converge.
Once the effective force constants are obtained, vibrational free energy and related properties can be evaluated within the SSCHA framework.


## MLP for SSCHA Calculations

To perform accurate SSCHA calculations for target compounds and structures across a range of volumes and temperatures, it is necessary to use either general-purpose polynomial MLPs or on-the-fly polynomial MLPs.
These models should enable accurate property evaluations for structures with large atomic displacements generated from the effective harmonic Hamiltonian in SSCHA calculations.

When SSCHA calculations are performed for many target structures using a single MLP, it is preferable to use a general-purpose MLP with high predictive power across a wide range of structures.
In contrast, when SSCHA calculations are carried out for a single compound under a specific condition or across a range of volumes and temperatures, developing an on-the-fly MLP can be a suitable option, as it can provide higher accuracy for the target system than a general-purpose MLP.

See [Development of On-the-fly MLP](../tutorial_onthefly.md) for more details.


## Single SSCHA calculations Using the Command-Line Interface

SSCHA calculations can be performed using the `pypolymlp-sscha` command.
Given a structure specified by the `--poscar` option, a supercell size specified by the `--supercell` option, and a polynomial MLP specified by the `--pot` option, a single SSCHA calculation can be performed as follows.

```shell
> pypolymlp-sscha --poscar POSCAR --pot polymlp.yaml --supercell 3 3 2 --temp 300 --mixing 0.5 --tol 0.01
```

In this example, the temperature is set to 300 K.
Also, the tolerance and mixing parameters for the force constants are set to 0.01 and 0.5, respectively.
If the SSCHA calculations complete successfully, `sscha_results.yaml` and the effective force constants file `fc2.hdf5` are generated for each temperature.

### Temperature Settings

When a single temperature is used for the SSCHA calculation, the `--temp` option can be used to specify the temperature.
When SSCHA calculations are performed sequentially at multiple temperatures, the `--temp_min`, `--temp_max`, and `--temp_step` options can be used to define the temperature sequence:

```shell
--temp_min 100 --temp_max 500 --temp_step 100
```

Alternatively, the `--n_temp` option can be used instead of `--temp_step` to specify the number of temperature points:

```shell
--temp_min 100 --temp_max 500 --n_temp 10
```
When SSCHA calculations are performed at multiple temperatures sequentially, the calculation at each temperature starts from the converged force constants obtained at the previous temperature, which helps accelerate convergence to the desired force-constant states.

If the `--ascending_temp` option is enabled, the temperature sequence is generated in ascending order:

```shell
--temp_min 100 --temp_max 500 --temp_step 100 --ascending_temp
```

In this case, the temperature sequence is [100, 200, 300, 400, 500].
Otherwise, the temperature sequence is generated in descending order.

### Initial Force-Constant Settings

The choice of initial force constants may lead to different converged solutions.
In `pypolymlp`, the harmonic force constants for a given structure, calculated using the specified polynomial MLP, are used as the default initial force constants.

```shell
--init harmonic
```

Alternatively, random force constants, constant force constants, or force constants read from a file can also be used:

```shell
--init random
--init const
--init file --init_file fc2.hdf5
```

### Tolerance and Mixing Parameters

The tolerance parameter is used as a convergence criterion for the SSCHA iterations.
The iterations stop when the change in the force constants satisfies the following condition defined by the tolerance parameter $\theta$:

$$
\frac{|\Phi_{\rm new}' - \Phi_{\rm old}|}{|\Phi_{\rm old}|} \leq \theta,
$$

where $\Phi_{\rm new}'$ and $\Phi_{\rm old}$ denote the updated and previous force constants, respectively.


A mixing scheme is applied to the updated force constants to improve the stability of convergence.
The mixing parameter controls the contribution of the force constants from the previous iteration and is defined as:

$$
\Phi_{\rm new}' = \alpha \Phi_{\rm new} + (1 - \alpha) \Phi_{\rm old},
$$

where $\alpha$ denotes the mixing parameter.

Here, $\Phi_{\rm new}$ represents the force constants estimated from a force–displacement dataset in the least-squares sense, where the dataset is sampled from the density matrix associated with the force constants from the previous iteration.
The mixed force constants, $\Phi_{\rm new}'$, are then used in the next iteration.


### Setting of the Number of Sample Structures

Within the SSCHA framework, the anharmonic contribution to the free energy is evaluated as an ensemble average over structures with atomic displacements sampled according to the effective force constants.
The forces acting on atoms and the stress tensor at a given temperature are also computed from these sampled structures.
Therefore, the number of samples required to accurately evaluate these properties and achieve convergence within a given tolerance strongly depends on the number of independent force-constant components and the chosen tolerance parameter.

In general, more samples are required for low-symmetry compounds with a larger number of independent force constants.
In `pypolymlp`, the number of sample structures is automatically determined based on the number of independent force constants and the tolerance parameter, if it is not explicitly specified.


When the number of sample structures is explicitly specified, the `--n_samples` option can be used.
The first value is used to obtain the converged effective force constants, while the second value is used to evaluate the SSCHA free energy and related properties.
To accurately evaluate these properties, it is recommended to set a larger value for the second number than for the first:

```shell
--n_samples 3000 6000
```

In this example, 3000 and 6000 structures are used to obtain the converged effective force constants and to evaluate the SSCHA free energy, respectively.

Other main parameters that can be specified via options are as follows.

```shell
 --mesh MESH MESH MESH  q-mesh for reciprocal space calculation
 --born_vasprun BORN_VASPRUN  vasprun.xml file for parsing born effective charges
 --cutoff_fc2 CUTOFF_FC2  Cutoff radius for effective force constants.
 --write_pdos           Save projected DOS.
```

## Generation of Random Structures from Converged Effective Force Constants
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
force_constants: FC2 at the final temperature, shape=(n_atom, n_atom, 3, 3).
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
