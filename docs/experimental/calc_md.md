# Molecular Dynamics Calculations

> **Note**: Requires `ASE` and `phonopy`.

When using the command-line interface, the `pypolymlp-md` command performs molecular dynamics simulations in the NVT ensemble as in the following example.

Molecular dynamics simulations can be performed using algorithms implemented in ASE.
ASE calculators for the polynomial MLP are provided in `pypolymlp` and is used to compute the properties required for molecular dynamics simulations.

## Standard MD Using the Command-Line Interface

When using the command-line interface, the `pypolymlp-md` command performs molecular dynamics simulations in the NVT ensemble, as shown in the example below.

```shell
> pypolymlp-md --poscar POSCAR --pot polymlp.yaml --supercell_size 3 3 3 --temp 300 --n_eq 5000 --n_steps 20000
```

The available options are as follows:
```
  --pot [POT ...]       polymlp file.
  -p, --poscar POSCAR   Initial structure.
  --supercell_size SUPERCELL_SIZE SUPERCELL_SIZE SUPERCELL_SIZE
                        Diagonal supercell size.
  --thermostat {Langevin,Nose-Hoover}
                        Thermostat.
  --temp TEMP           Temperature.
  --time_step TIME_STEP
                        Time step (fs).
  --friction FRICTION   Friction in Langevin thermostat (1/fs).
  --ttime TTIME         Time step interact with thermostat in Langevin thermostat
                        (fs).
  --n_eq N_EQ           Number of equilibration steps.
  --n_steps N_STEPS     Number of steps.
```

## Free Energy Perturbation Using the Command-Line Interface

> **Note**: Requires version 0.19.10 or later, as well as `ASE` and `phonopy`.

When a reference state is specified using either the `--fc2` option (for a harmonic vibrational state defined by second-order force constants) or the `--pot_ref` option (for a polynomial MLP), the free energy difference is calculated via free energy perturbation from the reference state to the target state specified by the polynomial MLP given with the `--pot` option.

The free energy difference in first-order perturbation theory is given by

$$
\Delta F = \langle U_{\mathrm{MLP}} - U_{\mathrm{ref}} \rangle_{\mathrm{ref}},
$$

which is evaluated as an ensemble average over the reference state.


To perform free energy perturbation between a harmonic vibrational state and a state described by a polynomial MLP, use the `--fc2` option as shown below:

```shell
pypolymlp-md --poscar POSCAR --pot polymlp.yaml --supercell_size 3 3 3 --temp 300 --n_eq 5000 --n_steps 20000 --fc2 fc2.hdf5
```

To perform free energy perturbation between two states described by polynomial MLPs, use the `--pot` and `--pot_ref` options as follows:

```shell
pypolymlp-md --poscar POSCAR --pot_ref polymlp.yaml.fast --pot polymlp.yaml.expensive --supercell_size 3 3 3 --temp 300 --n_eq 5000 --n_steps 20000
```

The calculated free energy values are stored in the `polymlp_md.yaml` file.

If an intermediate state between two states is used as the reference state, you can control the degree of mixing with the `--alpha` option, as shown below:

```shell
pypolymlp-md --poscar POSCAR --pot_ref polymlp.yaml.fast --pot polymlp.yaml.expensive --supercell_size 3 3 3 --temp 300 --n_eq 5000 --n_steps 20000 --alpha 0.8
```
In this example, a mixture of 0.2 * `polymlp.yaml.fast` + 0.8 * `polymlp.yaml.expensive`
is treated as the reference state, while `polymlp.yaml.expensive` is used as the target system.


## Standard MD Using the Python API

Standard MD simulations can also be performed using the Python API.
An example to use the Python API for MD simulations is shown.

```python
from pypolymlp.api.pypolymlp_md import PypolymlpMD

"""
Parameters
----------
thermostat: str
    Thermostat type, "Langevin" or "Nose-Hoover".
temperature : int
    Target temperature (K).
time_step : float
    Time step for MD (fs).
n_eq : int
    Number of equilibration steps.
n_steps : int
    Number of production steps.
"""
md = PypolymlpMD(verbose=True)
md.load_poscar("POSCAR")
md.set_supercell([3, 3, 3])
md.set_ase_calculator(pot="polymlp.yaml")
md.run_md_nvt(
    thermostat="Nose-Hoover",
    temperature=700,
    n_eq=5000,
    n_steps=20000,
    interval_save_forces=1,
    interval_save_trajectory=1,
    interval_log=1,
    logfile="polymlp_md_log.dat",
)
md.save_yaml(filename="polymlp_md.yaml")
```
