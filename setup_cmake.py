"""Setup script."""
import setuptools
import pathlib
import subprocess
from setuptools import setup
import os, glob
import shutil

def _run_cmake(cxx_dir):
    os.makedirs(cxx_dir + '/_build', exist_ok=True)
    args = [
        "cmake",
        "-S",
        cxx_dir,
        "-B",
        cxx_dir + '/_build'
    ]
    cmake_output = subprocess.check_output(args)
    print(cmake_output.decode("utf-8"))
    subprocess.check_call(["cmake", "--build", cxx_dir + '/_build', "-v"])
    subprocess.check_call(["cmake", "--install", cxx_dir + '/_build'])
    return cmake_output


cxx_dir = "./src/pypolymlp/cxx"
_run_cmake(cxx_dir)

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
