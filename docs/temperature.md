# MLP development at finite temperatures

## Electronic free energy model from VASP calculations

### Free energy calculation from vasprun.xml files
```
> pypolymlp-utils --electron_vasprun */vasprun.xml
```

### MLP development for electronic free energy
```
> pypolymlp -i polymlp.in

# polymlp.in
dataset_type electron
data ./*/electron.yaml
temperature 500
include_force False
```
or
```python
yamlfiles = sorted(glob.glob("./*/electron.yaml"))
polymlp = Pypolymlp()
polymlp.set_params(
    elements=["Ti"],
    cutoff=8.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=[8, 8],
    gaussian_params2=[0.0, 6.0, 7],
    atomic_energy=[0],
    reg_alpha_params=(-3, 1, 30),
)
polymlp.set_datasets_electron(yamlfiles, temperature=500)
polymlp.run(verbose=True)
polymlp.save_mlp(filename="polymlp.lammps.500")
```


## Vibrational free energy model from SSCHA calculations

### Structure generation for SSCHA calculations
1. Generation of random structures with symmetric constraints
```
> pypolymlp-sscha-structure --poscar POSCAR --n_samples 100 --max_deform 0.3 --max_distance 0.3
```

2. Generation of structures with different volumes
```
> pypolymlp-sscha-structure --volume --poscar POSCAR --n_samples 20 --min_volume 0.8 --max_volume 1.3
```

3. Generation of structures with different volumes and cell shapes
```
> pypolymlp-sscha-structure --cell --poscar POSCAR --n_samples 20 --min_volume 0.8 --max_volume 1.3 --max_deform 0.3
```

### Single SSCHA calculation
```
> pypolymlp-sscha --poscar POSCAR --pot polymlp.lammps --supercell 3 3 2 --temp_min 100 --temp_max 700 --temp_step 100 --mixing 0.5 --ascending_temp --n_samples 3000 6000
```

### MLP development for SSCHA free energy
```
> pypolymlp -i polymlp.in

# polymlp.in
dataset_type sscha
data ./*/sscha/300/sscha_results.yaml
include_force False
```
or
```python
yamlfiles = sorted(glob.glob("./*/sscha/300/sscha_results.yaml"))
polymlp = Pypolymlp()
polymlp.set_params(
    elements=["Zn", "S"],
    cutoff=8.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=[8, 8],
    gaussian_params2=[0.0, 7.0, 8],
    atomic_energy=[0.0, 0.0],
    reg_alpha_params=(-3, 3, 50),
)
polymlp.set_datasets_sscha(yamlfiles)
polymlp.run(verbose=True)
polymlp.save_mlp(filename="polymlp.lammps.300")
```
