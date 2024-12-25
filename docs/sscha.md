# SSCHA calculations

## Single SSCHA calculation
```
> pypolymlp-sscha --poscar POSCAR --pot polymlp.lammps --supercell 3 3 2 --temp_min 100 --temp_max 700 --temp_step 100 --mixing 0.5 --ascending_temp --n_samples 3000 6000
```

## Generation of random structures with symmetric constraints
```
> pypolymlp-sscha-structure --sym --poscar POSCAR --n_samples 100 --max_deform 0.3 --max_distance 0.3
```

## Generation of structures with different volumes
```
> pypolymlp-sscha-structure --volume --poscar POSCAR --n_samples 20 --min_volume 0.8 --max_volume 1.3
```

## Generation of structures with different volumes and cell shapes
```
> pypolymlp-sscha-structure --cell --poscar POSCAR --n_samples 20 --min_volume 0.8 --max_volume 1.3 --max_deform 0.3
```

## MLP development for SSCHA free energy
```
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
