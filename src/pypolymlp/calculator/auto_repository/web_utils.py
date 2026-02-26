"""Utility functions for generating web contents."""

import glob
import os

import numpy as np
import yaml

from pypolymlp.calculator.auto.web_sphinx_utils import array_to_csv_table, include_image


def generate_summary_txt(
    path_web: str,
    path_prediction: str,
    polymlps_id: str,
    polymlps: list,
):
    """Generate text for summary."""
    f = open(path_web + "/info.rst", "w")
    print(":orphan:", file=f)
    print(file=f)
    print(polymlps_id, file=f)
    print("====================================================", file=f)
    print(file=f)

    include_image("summary/polymlp_convex.png", title=None, height=350, file=f)

    id1 = polymlps[0]["id"]
    path_parse = path_prediction + "/predictions/" + id1 + "/"
    f1 = open(path_parse + "polymlp_size.dat")
    n_st = int(str(f1.readline()))
    f1.close()

    print(
        "The current structure dataset comprises " + str(n_st) + " structures."
        " Procedures to generate structures and estimate MLPs are found in"
        " `A. Seko, J. Appl. Phys. 133, 011101 (2023)"
        " <https://doi.org/10.1063/5.0129045>`_.",
        file=f,
    )
    print(file=f)

    include_image(
        "summary/polymlp_eqm_properties.png",
        title="Predictions using convex hull MLPs",
        height=350,
        file=f,
    )

    print(
        "These properties are calculated for MLP equilibrium structures"
        " obtained by performing local structure optimizations from"
        " the DFT equilibrium structures."
        " These DFT equilibrium structures are obtained by optimizing"
        " prototype structures that are included in ICSD."
        " As a result, the structure type of the converged structure may"
        " sometimes differ from the one shown in the legend.",
        file=f,
    )
    print(file=f)
    print(
        "The other properties predicted using each Pareto optimal MLP"
        " are available from column **Predictions** "
        " in the following table.",
        file=f,
    )
    print(file=f)

    min_rmse_f = min([float(d["rmse_force"]) for d in polymlps])
    if min_rmse_f > 0.05:
        print(
            polymlps_id,
            "shows large prediction errors."
            " Distributed MLPs should be carefully used."
            " MLPs are often accurate for reasonable structures,"
            " but it is sometimes inaccurate for unrealistic"
            " structures.",
            file=f,
        )
        print(file=f)

    print(file=f)

    print(".", end="", file=f)
    print(".", end="", file=f)
    print(" csv-table:: Pareto optimals (on convex hull)", file=f)
    print("  :header: Name, Time, RMSE, Predictions, Files", file=f)
    print("  :widths: 15,8,8,6,15", file=f)
    print(file=f)

    for d in polymlps:
        id1 = d["id"]
        txt_cost1 = "{:.3f}".format(float(d["cost_single"]))
        txt_cost2 = "{:.3f}".format(float(d["cost_openmp"]))
        txt_rmse1 = "{:.3f}".format(float(d["rmse_energy"]))
        txt_rmse2 = "{:.4f}".format(float(d["rmse_force"]))
        txt_header = "  " + id1
        txt_cost = txt_cost1 + " / " + txt_cost2
        txt_rmse = txt_rmse1 + " / " + txt_rmse2

        if d["active"]:
            txt_pred = ":doc:`predictions <predictions/" + id1 + "/prediction>`"
            txt_mlp = ":download:`polymlp <polymlps/polymlp-" + id1 + ".tar.xz>`"
            txt1 = ", ".join([txt_header, txt_cost, txt_rmse, txt_pred, txt_mlp])
            print(txt1, file=f)
        else:
            txt1 = ", ".join([txt_header, txt_cost, txt_rmse])
            print(txt1 + ", --, --", file=f)
    print(file=f)

    print("Units:", file=f)
    print(file=f)
    print("* Time: [ms] (1core/36cores)", file=f)
    print("* RMSE: [meV/atom]/[eV/ang.]", file=f)
    print(file=f)
    print(
        'Column "Time" shows the time required to compute the energy'
        " and forces for **1 MD step** and **1 atom**, which is"
        " estimated from 10 runs for a large structure using"
        " a workstation with Intel(R) Xeon(R) CPU E5-2695 v4 @ 2.10GHz.",
        file=f,
    )
    print(
        "Note that these MLPs should be carefully used for extreme"
        " structures. The MLPs often return meaningless values for them.",
        file=f,
    )
    print(file=f)
    print(
        "- All Pareto optimal MLPs are available :download:`here"
        " <polymlps/polymlps-" + polymlps_id + ".tar.xz>`.",
        file=f,
    )
    print(file=f)

    f.close()


def generate_predictions_txt(
    path_web: str,
    path_prediction: str,
    polymlps_id: str,
    polymlps: list,
):
    """Generate text for summary."""
    for d in polymlps:
        if not d["active"]:
            continue
        path = path_web + "/predictions/" + d["id"] + "/"
        os.makedirs(path, exist_ok=True)

        f = open(path + "prediction.rst", "w")
        print(":orphan:", file=f)
        print(file=f)
        print("----------------------------------------------------", file=f)
        print(d["id"] + " (" + polymlps_id + ")", file=f)
        print("----------------------------------------------------", file=f)
        print("", file=f)

        if os.path.exists(path + "polymlp_distribution.png"):
            include_image(
                "polymlp_distribution.png",
                height=600,
                file=f,
                title="Energy distribution",
            )
        if os.path.exists(path + "polymlp_comparison.png"):
            include_image(
                "polymlp_comparison.png",
                height=300,
                file=f,
                title="Absolute errors for energy in prototype structures",
            )
        if os.path.exists(path + "polymlp_eos.png"):
            include_image(
                "polymlp_eos.png",
                height=350,
                file=f,
                title="Equation of state",
            )
        if os.path.exists(path + "polymlp_eos_sep.png"):
            include_image(
                "polymlp_eos_sep.png",
                height=700,
                file=f,
                title="Equation of state for each structure",
            )
        if os.path.exists(path + "polymlp_phonon_dos.png"):
            include_image(
                "polymlp_phonon_dos.png",
                height=600,
                file=f,
                title="Phonon density of states",
            )
        if os.path.exists(path + "polymlp_thermal_expansion.png"):
            include_image(
                "polymlp_thermal_expansion.png",
                height=400,
                file=f,
                title="Thermal expansion",
            )
        if os.path.exists(path + "polymlp_bulk_modulus.png"):
            include_image(
                "polymlp_bulk_modulus.png",
                height=400,
                file=f,
                title="Bulk modulus",
            )

        path_parse = path_prediction + "/predictions/" + d["id"] + "/"
        prototype_data = np.loadtxt(
            path_parse + "polymlp_comparison.dat", dtype=str, skiprows=1
        )
        array_to_csv_table(
            prototype_data[:, np.array([2, 1, 0])],
            title="Prototype structure energy",
            subtitle="Prototype structure energy",
            header="Prototype, MLP (eV/atom), DFT (eV/atom)",
            widths=[25, 10, 10],
            file=f,
        )

        print("**Lattice constants and elastic constants**", file=f)
        print(
            "If the lattice constants obtained from the MLP and DFT are significantly"
            " different, the converged structures may also differ "
            " between the MLP and DFT.",
            file=f,
        )
        print(file=f)

        files_pred = glob.glob(path_parse + "polymlp_*/polymlp_predictions.yaml")
        for f_pred in files_pred:
            if not os.path.exists(f_pred):
                continue

            data = yaml.safe_load(open(f_pred))
            st = data["structure_type"]
            if "lattice_constants" in data:
                lc = data["lattice_constants"]
                data_lc = [
                    np.round(data["unitcell"]["volume"], 3),
                    lc["a"],
                    lc["b"],
                    lc["c"],
                    lc["alpha"],
                    lc["beta"],
                    lc["gamma"],
                ]
                array_to_csv_table(
                    np.array([data_lc]),
                    subtitle="Lattice constants [" + st + "] (MLP)",
                    header=(
                        "volume (ang.^3), a (ang.), b (ang.), c (ang.),"
                        " alpha, beta, gamma"
                    ),
                    widths=[12, 10, 10, 10, 8, 8, 8],
                    file=f,
                )

            if "lattice_constants_dft" in data:
                lc = data["lattice_constants_dft"]
                data_lc = [
                    np.round(data["volume_dft"], 3),
                    lc["a"],
                    lc["b"],
                    lc["c"],
                    lc["alpha"],
                    lc["beta"],
                    lc["gamma"],
                ]
                array_to_csv_table(
                    np.array([data_lc]),
                    subtitle="Lattice constants [" + st + "] (DFT)",
                    header=(
                        "volume (ang.^3), a (ang.), b (ang.), c (ang.),"
                        " alpha, beta, gamma"
                    ),
                    widths=[12, 10, 10, 10, 8, 8, 8],
                    file=f,
                )

            if "elastic_constants" in data:
                elastic_constants = np.array(data["elastic_constants"])
                array_to_csv_table(
                    elastic_constants,
                    subtitle="Elastic constants [" + st + "] (MLP)",
                    header="1, 2, 3, 4, 5, 6",
                    widths=[12, 10, 10, 10, 8, 8, 8],
                    file=f,
                )

        f.close()
