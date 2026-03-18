# Notes on input parameters

## Parameter settings
- `elements`: Strings of element species, (e.g., ['Mg','O']).

- `n_type`: Number of elements.

- `cutoff`: Cutoff radius (in angstroms).

- `model_type`: Polynomial function type with respect to invariants.
  - `model_type = 1`: Use linear polynomial of polynomial invariants
  - `model_type = 2`: Use polynomial of polynomial invariants
  - `model_type = 3`: Use polynomial of pair invariants
                      + linear polynomial of polynomial invariants
  - `model_type = 4`: Use polynomial of pair and second-order invariants
                      + linear polynomial of polynomial invariants

- `max_p`: Order of polynomial function.
           Only a value satisfying `max_p <= 3` is available.

- `feature_type`: Type of structural features. `feature_type = gtinv` or
                  `feature_type = pair` is available.

- `n_gaussians`: `n_gaussians` Gaussian functions are automatically provided.
                 If parameters in Gaussians are manually given, use `gaussian_params1`
                 and `gaussian_params2`.

- `gaussian_params`: Parameters for exp[- param1 * (r - param2)**2]
                     Parameters are given as np.linspace(p[0], p[1], p[2]),
                     where p[0], p[1], and p[2] are given by gaussian_params1
                     and gaussian_params2.

- `gtinv_order`: Maximum order of polynomial invariants.

- `gtinv_maxl`: Maximum angular numbers of polynomial invariants.
                [maxl for order=2, maxl for order=3, ...]

- `atomic_energy_unit`: "eV" or "Hartree" (Default: "eV").

- `atomic_energy`: Atomic energies (in `atomic_energy_unit`).

- `include_force`: Considering force entries (Default True).

- `include_stress`: Considering stress entries (Default True).

- `reg_alpha_params`: Parameters for penalty term in linear ridge regression.
                      Parameters are given as np.linspace(p[0], p[1], p[2]).

- `rearrange_by_elements`: Set True if not developing special MLPs.


## Notes on Atomic Energy
In the current implementation of polynomial MLP models, the intercept (constant term) is not included.
This means that the potential energy of a structure is measured relative to the sum of the energies of the isolated atoms that compose the structure.

Therefore, when using potential energies from training datasets obtained from DFT calculations, the atomic energies must be subtracted from the DFT-computed total energies to serve as reference values.

See the utility [Atomic Energies](utils_atomic_energies.md) when using VASP.


## Dataset settings
When both the training and test datasets are explicitly provided, they can be included in the input file as follows:

```shell
train_data vaspruns/train1/vasprun-*.xml.polymlp
train_data vaspruns/train2/vasprun-*.xml.polymlp
test_data vaspruns/test1/vasprun-*.xml.polymlp
test_data vaspruns/test2/vasprun-*.xml.polymlp
```

In cases where the datasets are automatically divided into training and test sets, they can be included in the input file as follows:

```shell
data vaspruns1/vasprun-*.xml.polymlp
data vaspruns2/vasprun-*.xml.polymlp
```

When a dataset contains property entries for multiple structures, such as those derived from molecular dynamics (MD) simulations, they can be specified in the input file as follows:

```shell
data_md vasprun-md1.xml
data_md vasprun-md2.xml
data_md vasprun-md-*.xml
```

These datasets will be automatically divided into training and test sets.
