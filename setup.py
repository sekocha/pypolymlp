"""Setup script."""

import setuptools
from skbuild import setup


def _get_version() -> str:
    with open("version.py") as f:
        for line in f:
            if "__version__" in line:
                version = line.split()[2].replace('"', "")
                return version


def main():
    """Run skbuild.setup."""
    version = _get_version()

    packages = setuptools.find_packages("./src")
    packages.append("pypolymlp.cxx.lib")

    with open("README.md") as f:
        long_description = f.read()

    setup(
        name="pypolymlp",
        version=version,
        author="Atsuto Seko",
        author_email="seko@cms.mtl.kyoto-u.ac.jp",
        maintainer="Atsuto Seko",
        maintainer_email="seko@cms.mtl.kyoto-u.ac.jp",
        description="Generator of polynomial machine learning potentials.",
        url="https://github.com/sekocha/pypolymlp",
        long_description=long_description,
        long_description_content_type="text/markdown",
        license="BSD-3-Clause",
        python_requires=">=3.8",
        install_requires=[
            "setuptools >= 61.0",
            "scikit-build",
            "ninja",
            "wheel",
            "cmake",
            "make",
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
        packages=packages,
        package_dir={"": "src"},
        include_package_data=True,
        cmake_source_dir="./src/pypolymlp/cxx",
        cmake_install_dir="./src/pypolymlp/cxx/lib",
        # cmake_args=["-DSKBUILD=ON", "-GUnix Makefiles"],
        cmake_args=["-DSKBUILD=ON"],
    )


if __name__ == "__main__":
    main()
