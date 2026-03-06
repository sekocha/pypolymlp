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
            alpha=0.7,
            marker="o",
            markersize=2,
        )

        ax[i].set_title(titles[i], fontsize=8)
        ax[i].set_xlabel("Composition", fontsize=8)
        ax[i].set_ylabel("Formation energy (eV/atom)", fontsize=8)
        ax[i].tick_params(axis="both", labelsize=8, length=0)

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
        ax[i].set_axisbelow(True)
        ax[i].grid(
            True,
            linestyle="--",
            linewidth=0.5,
        )

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
