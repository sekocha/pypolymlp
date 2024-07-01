"""Setup script."""

import setuptools
import skbuild


def main():
    """Run setuptools."""
    # version = _get_version()

    packages = setuptools.find_packages("./src")

    skbuild.setup(
        name="pypolymlp",
        version="0.1",
        author="Atsuto Seko",
        author_email="seko@cms.mtl.kyoto-u.ac.jp",
        maintainer="Atsuto Seko",
        maintainer_email="seko@cms.mtl.kyoto-u.ac.jp",
        description="This is the pypolymlp module.",
        python_requires=">=3.9",
        install_requires=[
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
        # ext_modules=_get_extensions(build_dir),
    )


if __name__ == "__main__":
    main()
