# SSCHA calculations
`symfc` and `phonopy` are required for performing SSCHA calculations.

## Using command line interface
### Single SSCHA calculation
```shell
> pypolymlp-sscha --poscar POSCAR --pot polymlp.yaml --supercell 3 3 2 --temp_min 100 --temp_max 700 --temp_step 100 --mixing 0.5 --ascending_temp --n_samples 3000 6000

# Number of sample structures are automatically determined.
> pypolymlp-sscha --poscar POSCAR --pot polymlp.yaml --supercell 3 3 2 --temp_min 100 --temp_max 700 --temp_step 100 --mixing 0.5 --ascending_temp
```

### Generation of random structures from SSCHA force constants
Random structures are generated based on the density matrix determined by the given effective force constants. The energy and force values for these structures are then calculated using the provided MLP.
```shell
pypolymlp-sscha-post --distribution --yaml sscha_results.yaml --fc2 fc2.hdf5 --n_samples 20 --pot polymlp.yaml
```

### Thermodynamic properties from SSCHA results at various volumes and temperatures

Thermodynamic properties can be calculated from SSCHA results on a volume-temperature grid.
```shell
> pypolymlp-sscha-post --properties --yaml ./*/sscha/*/sscha_results.yaml
```
<!--
Phase boundary determination from SSCHA thermodynamic properties for two phases
```shell
# Transition temperature
> pypolymlp-sscha-post --transition hcp/sscha_properties.yaml bcc/sscha_properties.yaml

# Pressure-temperature phase boundary
> pypolymlp-sscha-post --boundary hcp/sscha_properties.yaml bcc/sscha_properties.yaml
```
-->

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
)
```
