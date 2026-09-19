# MLP development for SSCHA free energy at finite temperatures

## Structure generation for SSCHA calculations
1. Generation of random structures with symmetric constraints
```shell
> pypolymlp-sscha-structure --poscar POSCAR --n_samples 100 --max_deform 0.3 --max_distance 0.3
```

2. Generation of structures with different volumes
```shell
> pypolymlp-sscha-structure --volume --poscar POSCAR --n_samples 20 --min_volume 0.8 --max_volume 1.3
```

3. Generation of structures with different volumes and cell shapes
```shell
> pypolymlp-sscha-structure --cell --poscar POSCAR --n_samples 20 --min_volume 0.8 --max_volume 1.3 --max_deform 0.3
```

4. Run SSCHA calculations for structures

5. MLP development for SSCHA free energy
```shell
> pypolymlp -i polymlp.in

# polymlp.in
dataset_type sscha
data ./*/sscha/300/sscha_results.yaml
include_force True
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
polymlp.save_mlp(filename="polymlp.yaml.300")
```
