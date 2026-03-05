"""Functions for plotting formation energies."""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator


def plot_binary_formation_energies(
    system: str,
    pot_id: str,
    data_dft_all: np.ndarray,
    data_dft_convex: np.ndarray,
    data_mlp_all: np.ndarray,
    data_mlp_convex: np.ndarray,
    data_mlp_go_all: Optional[np.ndarray] = None,
    data_mlp_go_convex: Optional[np.ndarray] = None,
    path_output: str = "./",
    filename_suffix: Optional[str] = None,
    use_eps: bool = False,
    dpi: int = 300,
):
    """Plot formation energies."""
    os.makedirs(path_output, exist_ok=True)

    plt.style.use("bmh")
    sns.set_context("paper", 1.0, {"lines.linewidth": 4})
    sns.set_palette("coolwarm_r", 8, 1)

    data_all = [data_dft_all, data_mlp_all, data_mlp_go_all]
    data_convex = [data_dft_convex, data_mlp_convex, data_mlp_go_convex]
    titles = [
        "DFT",
        "MLP (DFT Converged Structures)",
        "MLP (MLP Geometry Optimized Structures)",
    ]

    if data_mlp_go_all is not None:
        figsize1 = 7.5
        size = 3
    else:
        figsize1 = 5
        size = 2

    fig, ax = plt.subplots(size, 1, figsize=(5, figsize1))
    fig.suptitle(
        "Formation energy for prototypes (" + system + ", " + pot_id + ")",
        fontsize=10,
    )

    for i in range(size):
        if data_all[i] is None:
            continue

        ax[i].scatter(
            data_all[i][:, 1],
            data_all[i][:, 2],
            s=2.0,
            c="black",
            alpha=1.0,
            marker=".",
        )
        ax[i].plot(
            data_convex[i][:, 1],
            data_convex[i][:, 2],
            color="red",
            linewidth=0.5,
            alpha=0.5,
            marker="o",
            markersize=2,
        )

        ax[i].set_title(titles[i], fontsize=8)
        ax[i].set_xlabel("Composition", fontsize=8)
        ax[i].set_ylabel("Formation energy (eV/atom)", fontsize=8)
        ax[i].tick_params(axis="both", labelsize=8)

        e_min = np.min(data_all[i][:, -1])
        e_max = np.max(data_all[i][:, -1])
        buffer = (e_max - e_min) / 10
        ax[i].set_xticks(np.arange(0.0, 1.01, 0.2))
        ax[i].set_xlim(-0.02, 1.02)
        ax[i].yaxis.set_major_locator(
            MaxNLocator(
                nbins=6,
                min_n_ticks=4,
                steps=[1, 2, 2.5, 5, 10],
            )
        )
        ax[i].set_ylim(min(e_min - buffer, 0), max(e_max + buffer, 0))

    plt.tight_layout()
    filename = path_output + "/polymlp_formation_energy"
    if filename_suffix is not None:
        filename += "_" + filename_suffix
    if use_eps:
        filename += ".eps"
        plt.savefig(filename, format="eps")
    else:
        filename += ".png"
        plt.savefig(filename, format="png", dpi=dpi)
    plt.clf()
    plt.close()


#
# data_all = [
#     [0.8, 0.2, -0.03495486],
#     [0.5, 0.5, -0.04193504],
#     [0.75, 0.25, -0.04421533],
#     [0.75, 0.25, -0.04305931],
#     [0.33333333, 0.66666667, -0.04163314],
#     [0.42857143, 0.57142857, -0.03554862],
#     [0.25, 0.75, -0.04123442],
#     [0.33333333, 0.66666667, -0.01564521],
#     [0.5, 0.5, -0.03767163],
#     [0.75, 0.25, -0.04349769],
#     [0.5, 0.5, -0.06030366],
#     [0.5, 0.5, -0.04310616],
#     [0.5, 0.5, -0.06016557],
#     [0.66666667, 0.33333333, -0.00897219],
#     [0.75, 0.25, -0.03866726],
#     [0.25, 0.75, -0.04231097],
#     [0.33333333, 0.66666667, -0.03007397],
#     [0.25, 0.75, -0.04339408],
#     [0.55555556, 0.44444444, -0.02267213],
#     [0.25, 0.75, -0.01073302],
#     [0.5, 0.5, -0.02596357],
#     [0.75, 0.25, -0.00992114],
#     [0.5, 0.5, -0.02528843],
#     [0.75, 0.25, -0.04300292],
#     [0.75, 0.25, -0.0434132],
#     [0.375, 0.625, -0.04631074],
#     [0.625, 0.375, -0.00331497],
#     [0.75, 0.25, -0.03839232],
#     [0.5, 0.5, -0.04049716],
#     [0.4, 0.6, -0.02199164],
#     [0.6875, 0.3125, -0.01218939],
#     [0.6, 0.4, -0.05347057],
#     [0.33333333, 0.66666667, -0.04160422],
#     [0.25, 0.75, -0.04481298],
#     [0.4, 0.6, -0.04915533],
#     [0.83333333, 0.16666667, -0.02659686],
#     [0.33333333, 0.66666667, -0.01068447],
#     [0.25, 0.75, -0.04339358],
#     [0.5, 0.5, -0.06016916],
#     [0.75, 0.25, -0.04350109],
#     [0.5, 0.5, -0.0568585],
#     [0.4, 0.6, -0.05396222],
#     [0.5, 0.5, -0.0431169],
#     [0.3125, 0.6875, -0.04865644],
#     [0.75, 0.25, -0.04308536],
#     [0.66666667, 0.33333333, -0.00756026],
#     [0.16666667, 0.83333333, -0.02537713],
#     [0.6, 0.4, -0.02018703],
#     [0.75, 0.25, -0.04391786],
#     [0.66666667, 0.33333333, -0.00894187],
#     [0.25, 0.75, -0.04455425],
#     [0.5, 0.5, -0.05685651],
#     [0.5, 0.5, -0.03766161],
#     [0.25, 0.75, -0.04230973],
#     [0.5, 0.5, -0.06016147],
#     [0.625, 0.375, -0.0186377],
#     [0.25, 0.75, -0.04454176],
#     [0.5, 0.5, -0.04049892],
#     [0.57142857, 0.42857143, -0.04467297],
#     [0.5, 0.5, -0.01788027],
#     [0.5, 0.5, -0.0253875],
#     [0.6, 0.4, -0.02111435],
#     [0.25, 0.75, -0.03767626],
#     [0.25, 0.75, -0.03749841],
#     [0.5, 0.5, -0.06016959],
#     [0.33333333, 0.66666667, -0.04161281],
#     [0.75, 0.25, -0.00992112],
#     [0.5, 0.5, -0.04310629],
#     [0.66666667, 0.33333333, -0.02909442],
#     [0.75, 0.25, -0.04308299],
#     [0.44444444, 0.55555556, -0.01710763],
#     [0.75, 0.25, -0.04300284],
#     [0.25, 0.75, -0.01073302],
#     [0.2, 0.8, -0.0335608],
#     [0.5, 0.5, -0.03769961],
#     [0.8, 0.2, -0.01291336],
#     [0.66666667, 0.33333333, -0.00890328],
#     [0.25, 0.75, -0.04501682],
#     [0.25, 0.75, -0.04339547],
#     [0.5, 0.5, -0.04193472],
#     [0.66666667, 0.33333333, -0.0291716],
#     [0.75, 0.25, -0.04111663],
#     [0.57142857, 0.42857143, -0.03508764],
#     [0.25, 0.75, -0.04454846],
#     [0.1875, 0.8125, -0.00327337],
#     [1.0, 0.0, 0.0],
#     [0.0, 1.0, 0.0],
# ]
#
# data_convex = [
#     [1.0, 0.0, 0.0],
#     [0.75, 0.25, -0.04421533],
#     [0.5, 0.5, -0.06030366],
#     [0.25, 0.75, -0.04501682],
#     [0.0, 1.0, 0.0],
# ]
#
# data_all = np.array(data_all)
# data_convex = np.array(data_convex)
# plot_binary_formation_energies(
#     "Ag-Au",
#     "polymlp",
#     data_all,
#     data_convex,
#     data_all,
#     data_convex,
#     data_mlp_go_all=data_all,
#     data_mlp_go_convex=data_convex,
# )
