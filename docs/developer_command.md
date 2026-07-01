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

## Enumeration of polynomial invariants
```shell
pypolymlp-invariant -l 3 3 3 3 3 3
pypolymlp-invariant --orders 2 3 4 --maxl 5
```

## Calculation Using Lammps with Interatomic Potentials

### Using Polynomial MLP
```shell
pypolymlp-lammps --properties --poscar POSCAR --pot polymlp.yaml --style polymlp --elements Ti Al
```

### Using Conventional IPs
```shell
pypolymlp-lammps --properties --poscar POSCAR --pot Ti-Al-2003.eam.alloy --style eam/alloy --elements Ti Al
pypolymlp-lammps --elastic --poscar POSCAR --pot Ti-Al-2003.eam.alloy --style eam/alloy --elements Ti Al
pypolymlp-lammps --eos --poscar POSCAR --pot Ti-Al-2003.eam.alloy --style eam/alloy --elements Ti Al
pypolymlp-lammps --phonon --poscar POSCAR --pot Ti-Al-2003.eam.alloy --style eam/alloy --elements Ti Al --supercell 2 2 2
```

## Automated Calculations Using Lammps
```shell
pypolymlp-lammps-autocalc --pot Ti-Al-2003.eam.alloy --style eam/alloy --elements Ti Al
```
