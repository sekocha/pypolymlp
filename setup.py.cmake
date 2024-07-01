"""Setup script."""

import os
import pathlib
import shutil
import subprocess

import numpy
import setuptools
# from setuptools import setup

# from skbuild import setup

def _run_cmake(build_dir):
    # build_dir.mkdir()
    os.makedirs(build_dir, exist_ok=True)
    args = [
        "cmake",
        "-S",
        #".",
        "./src/pypolymlp/cxx",
        "-B",
        "./src/pypolymlp/cxx/_build",
#         "-DCMAKE_INSTALL_PREFIX=.",
    ]
    cmake_output = subprocess.check_output(args)
    print(cmake_output.decode("utf-8"))
    subprocess.check_call(["cmake", "--build", "src/pypolymlp/cxx/_build", "-v"])
    return cmake_output

"""Temporary codes"""

#os.makedirs("./build/bdist.linux-x86_64/wheel/pypolymlp/cxx/lib", exist_ok=True)
#for file1 in glob.glob("./src/pypolymlp/cxx/lib/*.so"):
#    shutil.copy(file1, "./build/bdist.linux-x86_64/wheel/pypolymlp/cxx/lib")

"""Temporary codes (End)"""


def _get_extensions(build_dir):
    """Return python extension setting.

    User customization by site.cfg file
    -----------------------------------
    See _get_params_from_site_cfg().

    Automatic search using cmake
    ----------------------------
    Invoked by environment variable PHONOPY_USE_OPENMP=true.

    """
    # params = _get_params_from_site_cfg()

    # Libraray search
    found_extra_link_args = []
    found_extra_compile_args = []

    # sources = ["c/_phonopy.c"]
    cmake_output = _run_cmake(build_dir)
    # found_flags = {}
    # found_libs = {}
    # for line in cmake_output.decode("utf-8").split("\n"):
    #     for key in ["OpenMP"]:
    #         if f"{key} libs" in line and len(line.split()) > 3:
    #             found_libs[key] = line.split()[3].split(";")
    #         if f"{key} flags" in line and len(line.split()) > 3:
    #             found_flags[key] = line.split()[3].split(";")
    # for _, value in found_libs.items():
    #     found_extra_link_args += value
    # for _, value in found_flags.items():
    #     found_extra_compile_args += value
    # print("=============================================")
    # print("Parameters found by cmake")
    # print("extra_compile_args: ", found_extra_compile_args)
    # print("extra_link_args: ", found_extra_link_args)
    # print("=============================================")
    # print()

    # # Build ext_modules
    # extensions = []
    # params["extra_link_args"] += found_extra_link_args
    # params["extra_compile_args"] += found_extra_compile_args
    # params["define_macros"].append(("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"))
    # params["include_dirs"] += ["cxx", numpy.get_include()]

    # libmlp = list((pathlib.Path.cwd() / "_build").glob("*mlpcpp.*"))
    # if libmlp:
    #     print("=============================================")
    #     print(f"Pypolymlp library: {libmlp[0]}")
    #     print("=============================================")
    #     params["extra_objects"] += [str(libmlp[0])]

#    extensions.append(
#        setuptools.Extension("phonopy._phonopy", sources=sources, **params)
#    )

    return extensions


def main(build_dir):
    """Run setuptools."""
    #version = _get_version()

    packages = setuptools.find_packages("./src")
    #packages.append("pypolymlp.cxx.lib.libmlpcpp")

    setuptools.setup(
        name="pypolymlp",
        version="0.1",
        description="This is the pypolymlp module.",
        author="Atsuto Seko",
        author_email="seko@cms.mtl.kyoto-u.ac.jp",
        python_requires=">=3.9",
        install_requires=[
            "numpy",
            "scipy",
        ],
        provides=["pypolymlp"],
        platforms=["all"],
        entry_points={
            "console_scripts": [
                "pypolymlp=pypolymlp.api.run_polymlp:run",
                "pypolymlp-calc=pypolymlp.api.run_polymlp_calc:run",
                "pypolymlp-utils=pypolymlp.api.run_polymlp_utils:run",
                "pypolymlp-structure=pypolymlp.api.run_polymlp_str:run",
            ],
        },
        packages=packages,
        ext_modules=_get_extensions(build_dir),
    )
#    _clean_cmake(build_dir)


if __name__ == "__main__":
    #build_dir = pathlib.Path.cwd() / "_build"
    build_dir = pathlib.Path.cwd() / "src/pypolymlp/cxx/_build"
    main(build_dir)
