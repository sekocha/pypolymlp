#!/usr/bin/env python
import numpy as np
import os, shutil

poscar_dir = '/home/seko/database/icsd/data-20160509/1-original/poscar/'
os.makedirs('poscars', exist_ok=True)

dirs = ['1-all', '2-alloy', '2-ionic', '3-alloy', '3-ionic']
for d in dirs:
    print(d)
    f = open(d + '/summary_nonequiv')
    lines = f.readlines()
    f.close()
    ids = [int(l.split(',')[0]) for l in lines[1:]]
    for i in ids:
        file1 = poscar_dir + 'icsd-' + str(i)
        file2 = 'poscars/' + 'icsd-' + str(i)
        shutil.copy(file1, file2)
        
    
