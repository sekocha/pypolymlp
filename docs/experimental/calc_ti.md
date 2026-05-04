# Thermodynamic Integration Using Molecular Dynamics

> **Note**: Requires `ASE` and `phonopy`.

Thermodynamic integration can be used to calculate the free energy difference between a reference state defined by harmonic force constants and a target state described by a polynomial MLP.
To perform thermodynamic integration, molecular dynamics simulations are carried out for multiple intermediate states between the reference and target systems.
The number of intermediate states is specified by the user.
The free energy difference is numerically calculated as
$$
\Delta F = F_{\rm MLP} - F_{\rm FC2} = \int_0^1 \langle U_{\rm MLP} - U_{\rm FC2} \rangle_\alpha \, d\alpha.
$$

## Using the Command-Line Interface

Thermodynamic integration can be performed using the `pypolymlp-md` command with the `--ti` option.
Thermodynamic integration for structure `POSCAR` using polymomial MLP `polymlp.yaml` are performed as follows.

```shell
pypolymlp-md --poscar POSCAR --pot polymlp.yamls --supercell_size 3 3 3 --temp 500 --n_eq 5000 --n_steps 20000 --ti --n_samples 20 --fc2_path ./sscha --max_alpha 0.98
```

The values of the free energy, energy, entropy, and other related properties are stored in the `polymlp_ti.yaml` file.

In this example, the number of intermediate states is set to 20 using the `--n_samples` option.

If the `--fc2_path` option is provided, a reference state is automatically selected from the `--fc2_path` directory.
`pypolymlp` assumes that the effective force constants at the lowest temperature in the `--fc2_path` directory are used as the reference state.
If a reference state is specified using force constants, use the `--fc2` option.


The `--max_alpha` option can be used to set an upper bound for thermodynamic integration.
In some cases, molecular dynamics simulations at high temperatures for the target state ($\alpha = 1$) may lead to melting, and such configurations should not be included in the thermodynamic integration starting from harmonic vibrational states.
The `--max_alpha` option is useful for excluding such problematic molecular dynamics points from the integration.


## Using the Python API

Thermodynamic integration can also be performed using the Python API.
An example of how to perform thermodynamic integration using the Python API is shown below.

```python
from pypolymlp.api.pypolymlp_md import PypolymlpMD

"""
Parameters
----------
thermostat: Thermostat.
n_alphas: Number of sample points for thermodynamic integration
          using Gaussian quadrature.
temperature : int
    Target temperature (K).
n_eq : int
    Number of equilibration steps.
n_steps : int
    Number of production steps.
"""

md = PypolymlpMD(verbose=True)
md.load_poscar("POSCAR")
md.set_supercell([3, 3, 3])
md.set_ase_calculator_with_fc2(pot="polymlp.yaml", fc2hdf5="fc2.hdf5")
md.run_thermodynamic_integration(
    thermostat="Langevin",
    n_alphas=10,
    max_alpha=1.0,
    temperature=300.0,
    time_step=1.0,
    friction=0.01,
    n_eq=5000,
    n_steps=20000,
)
md.save_ti_yaml(filename="polymlp_ti.yaml")
```
