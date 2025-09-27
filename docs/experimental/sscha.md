# SSCHA calculation

## Single SSCHA calculation
```shell
> pypolymlp-sscha --poscar POSCAR --pot polymlp.yaml --supercell 3 3 2 --temp_min 100 --temp_max 700 --temp_step 100 --mixing 0.5 --ascending_temp --n_samples 3000 6000
```

## Generation of random structures and their properties from SSCHA force constants
Random structures are generated based on the density matrix determined by the given effective force constants. The energy and force values for these structures are then calculated using the provided MLP.
```shell
pypolymlp-sscha-post --distribution --yaml sscha_results.yaml --fc2 fc2.hdf5 --n_samples 20 --pot polymlp.yaml
```

## SSCHA calculations at multiple volumes

Thermodynamic properties and phase boundary can be calculated as follows.
### Property calculations using SSCHA calculations on a volume-temperature grid
```shell
> pypolymlp-sscha-post --properties --yaml ./*/sscha/*/sscha_results.yaml
```

### Phase boundary determination from SSCHA thermodynamic properties for two phases
```shell
# Transition temperature
> pypolymlp-sscha-post --transition hcp/sscha_properties.yaml bcc/sscha_properties.yaml

# Pressure-temperature phase boundary
> pypolymlp-sscha-post --boundary hcp/sscha_properties.yaml bcc/sscha_properties.yaml
```
