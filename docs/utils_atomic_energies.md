## Atomic Energies

> **Note:** This feature is experimental.
> It is only available for VASP with the PBE and PBEsol functionals.

In the current implementation of polynomial MLP models, the intercept (constant term) is not included. This means that the potential energy of a structure is measured relative to the sum of the energies of the isolated atoms that compose the structure.
Therefore, when using potential energies from training datasets obtained from DFT calculations, the atomic energies must be subtracted from the DFT-computed total energies to serve as reference values.

When using VASP for DFT calculations, atomic energy values for the PBE and PBEsol functionals are available in `pypolymlp-utils`, as shown below.

```shell
> pypolymlp-utils --atomic_energy_elements Mg O --atomic_energy_functional PBE
> pypolymlp-utils --atomic_energy_formula MgO --atomic_energy_functional PBE
> pypolymlp-utils --atomic_energy_formula Al2O3 --atomic_energy_functional PBEsol
```
The standard output from the atomic energy option can be appended to your `polymlp.in`, which will be used for developing MLPs.
