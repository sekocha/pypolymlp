"""Setup script."""
import setuptools
from setuptools import setup

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
    cmake_install_dir="c++/build",
    entry_points={
        'console_scripts': [
            'pypolymlp=pypolymlp.run_polymlp:run',
        ],
    },
    packages=setuptools.find_packages(),
)
