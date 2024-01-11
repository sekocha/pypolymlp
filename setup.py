"""Setup script."""
import setuptools
from setuptools import setup
import os, glob
import shutil

"""Temporary codes"""

os.makedirs('./build/bdist.linux-x86_64/wheel/pypolymlp/cxx/lib', exist_ok=True)
for file1 in glob.glob('./src/pypolymlp/cxx/lib/*.so'):
    shutil.copy(file1, './build/bdist.linux-x86_64/wheel/pypolymlp/cxx/lib')

#os.makedirs('./build/bdist.linux-x86_64/wheel/pypolymlp/str_gen/prototypes', exist_ok=True)
#shutil.copytree('./src/pypolymlp/str_gen/prototypes/poscars', 
#    './build/bdist.linux-x86_64/wheel/pypolymlp/str_gen/prototypes/poscars',
#    dirs_exist_ok=True)
#shutil.copytree('./src/pypolymlp/str_gen/prototypes/list_icsd', 
#    './build/bdist.linux-x86_64/wheel/pypolymlp/str_gen/prototypes/list_icsd',
#    dirs_exist_ok=True)

"""Temporary codes (End)"""

setup(
    name="pypolymlp",
    version="0.1",
    setup_requires=["numpy", "setuptools"],
    description="This is the pypolymlp module.",
    author="Atsuto Seko",
    author_email="seko@cms.mtl.kyoto-u.ac.jp",
    install_requires=["numpy", "scipy"],
    provides=["pypolymlp"],
    platforms=["all"],
    entry_points={
        'console_scripts': [
            'pypolymlp=pypolymlp.api.run_polymlp:run',
            'pypolymlp-utils=pypolymlp.api.run_polymlp_utils:run',
            'pypolymlp-structure=pypolymlp.api.run_polymlp_str:run',
        ],
    },
    packages=setuptools.find_packages("./src"),
    #cmake_install_dir="cxx/lib",
)
