#!/bin/bash

python run_NVT.py -p 'POSCAR-Al_fcc_333' -m 'polymlp_190.lammps'
python run_Langevin.py -p 'POSCAR-Al_fcc_333' -m 'polymlp_190.lammps'