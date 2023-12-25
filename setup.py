"""Setup script."""
import setuptools
from setuptools import setup
import os
import shutil

''' temporary codes '''
os.makedirs('./build/bdist.linux-x86_64/wheel/pypolymlp/cxx/lib', exist_ok=True)
shutil.copy('./src/pypolymlp/cxx/lib/mlpcpp.so', './build/bdist.linux-x86_64/wheel/pypolymlp/cxx/lib')

setup(
    name="pypolymlp",
    version="0.1",
    setup_requires=["numpy", "setuptools"],
    description="This is the pypolymlp module.",
    author="Atsuto Seko",
    author_email="seko@cms.mtl.kyoto-u.ac.jp",
    install_requires=["numpy", "scipy", "phonopy", "spglib"],
    provides=["pypolymlp"],
    platforms=["all"],
    entry_points={
        'console_scripts': [
            'pypolymlp=pypolymlp.run_polymlp:run',
        ],
    },
    packages=setuptools.find_packages("./src"),
    #cmake_install_dir="cxx/lib",
)


