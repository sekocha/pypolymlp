# Command-line interface for pypolymlp developers

## Force constants

### Calculation of force-constant basis set using symfc

```shell
pypolymlp-symfc --poscar POSCAR --supercell 3 3 2 --orders 2 3 --disable_mkl
```

```shell
pypolymlp-symfc --poscar POSCAR --supercell 3 3 3 --orders 2 3 --disable_mkl --cutoff_fc2 8 --cutoff_fc3 8
```

```shell
pypolymlp-symfc --poscar POSCAR --supercell 3 3 3 --orders 2 3 4 --disable_mkl --cutoff_fc2 8 --cutoff_fc3 8 --cutoff_fc4 6
```

### Calculation of force-constants from DFT dataset using symfc

```shell
pypolymlp-symfc --poscar POSCAR --supercell 2 2 2 --orders 2 3 --disable_mkl --vaspruns vaspruns/vasprun-*.xml
```

## OpenKIM

### Generation of OpenKIM portable model
```shell
pypolymlp-kim --pot polymlp.yaml
```
