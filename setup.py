"""Setup script."""

import setuptools
import skbuild


def _get_version() -> str:
    with open("version.py") as f:
        for line in f:
            if "__version__" in line:
                version = line.split()[2].replace('"', "")
                return version


def main():
    """Run skbuild.setup."""
    version = _get_version()
    # print(version)
    # version = "0.1.5"

    packages = setuptools.find_packages("./src")
    with open("README.md") as f:
        long_description = f.read()

    skbuild.setup(
        name="pypolymlp",
        version=version,
        author="Atsuto Seko",
        author_email="seko@cms.mtl.kyoto-u.ac.jp",
        maintainer="Atsuto Seko",
        maintainer_email="seko@cms.mtl.kyoto-u.ac.jp",
        description="Generator of polynomial machine learning potentials.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        license="BSD-3-Clause",
        python_requires=">=3.9",
        install_requires=[
            "setuptools >= 61.0",
            "scikit-build",
            "wheel",
            "cmake",
            "intel-openmp",
            "pybind11",
            "pybind11-global",
            "eigen",
            "numpy",
            "scipy",
            "pyyaml",
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
        cmake_source_dir="./src/pypolymlp/cxx",
        cmake_install_dir="./src/pypolymlp/cxx/lib",
        packages=packages,
        package_dir={"": "src"},
    )


if __name__ == "__main__":
    main()
