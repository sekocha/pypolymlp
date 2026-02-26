"""Utility functions for generating figures."""

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_mlp_distribution(
    summary_all: np.ndarray,
    summary_convex: np.ndarray,
    system: str,
    path_output: str = "./",
    use_eps: bool = False,
    dpi: int = 300,
):
    """Plot error-cost distribution of MLPs.

    Parameters
    ----------
    summary_all: Error-cost data for all MLPs.
    summary_convex: Error-cost data for convex MLPs.
    system: System.
    path_output: Directory path for generating figures.
    dpi: Resolution of figures.
    """

    os.makedirs(path_output, exist_ok=True)
    if not isinstance(summary_all, np.ndarray):
        summary_all = np.array(summary_all)
    if not isinstance(summary_convex, np.ndarray):
        summary_convex = np.array(summary_convex)

    plt.style.use("bmh")
    sns.set_context("paper", 1.0, {"lines.linewidth": 4})
    sns.set_palette("coolwarm_r", 8, 1)

    plt.xlim(1e-5, 5e-2)
    plt.ylim(0, 25)
    plt.xscale("log")
    major_ticks = np.arange(0, 26, 5)
    minor_ticks = np.arange(0, 26, 1)
    plt.yticks(major_ticks)
    plt.yticks(minor_ticks, minor=True)

    plt.title("Polynomial MLP distribution (" + system + ")")
    plt.xlabel("Elapsed time [s/atom/step] (Single CPU core)")
    plt.ylabel("RMS error [meV/atom]")
    sns.set_style("whitegrid", {"grid.linestyle": "--"})
    plt.grid(which="minor", alpha=0.2)

    plt.scatter(
        summary_all[:, 0].astype(float) / 1000,
        summary_all[:, 2].astype(float),
        s=10,
        c="black",
        marker=".",
        alpha=1.0,
    )
    plt.plot(
        summary_convex[:, 0].astype(float) / 1000,
        summary_convex[:, 2].astype(float),
        marker=".",
        alpha=1.0,
        markersize=9,
        linewidth=0.8,
        markeredgecolor="k",
        markeredgewidth=0.4,
    )
    if use_eps:
        plt.savefig(path_output + "/polymlp_convex.eps", format="eps")
    else:
        plt.savefig(path_output + "/polymlp_convex.png", format="png", dpi=dpi)
    plt.clf()
    plt.close()


def plot_eqm_properties(
    prototype_data: list,
    times: np.ndarray,
    system: str,
    path_output: str = "./",
    use_eps: bool = False,
    dpi: int = 300,
):
    """Plot mlp-dependent properties of equilibrium prototypes.

    Parameters
    ----------
    prototype_data: List of Prototype instances including properties.
    times: List of computational costs.
    system: System.
    path_output: Directory path for generating figures.
    dpi: Resolution of figures.
    """
    os.makedirs(path_output, exist_ok=True)
    eqm_props_dict = [defaultdict(list), defaultdict(list), defaultdict(list)]
    for prototypes in prototype_data:
        for prot in prototypes:
            st = prot.name
            eqm_props_dict[0][prot.name].append(prot.energy)
            eqm_props_dict[1][prot.name].append(prot.volume)
            eqm_props_dict[2][prot.name].append(prot.bulk_modulus)

    plt.style.use("bmh")
    sns.set_context("paper", 1.0, {"lines.linewidth": 4})
    sns.set_palette("Paired", len(eqm_props_dict[0]), 1)
    marker_candidates = [".", "^", "d", "P", "X"]
    marker_sizes = [6, 3, 3, 3, 3]

    fig, ax = plt.subplots(1, 3, figsize=(7, 3))

    max_vals, min_vals = [[], [], []], [[], [], []]
    for i in range(3):
        for j, (st, list1) in enumerate(eqm_props_dict[i].items()):
            x = [t / 1000 for t, val in zip(times, list1) if val is not None]
            y = [val for val in list1 if val is not None]
            ax[i].plot(
                x,
                y,
                marker=marker_candidates[j // 12],
                markersize=marker_sizes[j // 12],
                linewidth=0.8,
                markeredgecolor="k",
                markeredgewidth=0.3,
                label=st,
            )
            max_vals[i].append(max(y[-8:]))
            min_vals[i].append(min(y[-8:]))

    xlabel = "Elapsed time [s/atom/step]"
    ax[0].set_title("Cohesive energy (" + system + ", Optimal MLPs)", fontsize=7)
    ax[0].set_xlabel(xlabel, fontsize=6)
    ax[0].set_ylabel("Cohesive energy [eV/atom]", fontsize=6)

    ax[1].set_title("Equilibrium volume (" + system + ", Optimal MLPs)", fontsize=7)
    ax[1].set_xlabel(xlabel, fontsize=6)
    ax[1].set_ylabel(r"Volume [$\mathrm{\AA}^3$/atom]", fontsize=6)

    ax[2].set_title("Bulk modulus (" + system + ", Optimal MLPs)", fontsize=7)
    ax[2].set_xlabel(xlabel, fontsize=6)
    ax[2].set_ylabel("Bulk modulus [GPa]", fontsize=6)

    for i in range(3):
        ax[i].legend(
            bbox_to_anchor=(0, 1),
            loc="upper left",
            borderaxespad=0,
            fontsize=5,
        )
        ax[i].tick_params(axis="both", labelsize=6)
        ax[i].set_xlim(5e-7, 5e-2)
        ax[i].set_xscale("log")

        dval = (max(max_vals[i]) - min(min_vals[i])) / 4
        if i == 1:
            dval = max(dval, 1.0)
        elif i == 2:
            dval = max(dval, 20.0)
        if not i == 2:
            ax[i].set_ylim(min(min_vals[i]) - dval, max(max_vals[i]) + dval)
        else:
            ax[i].set_ylim(0, max(max_vals[i]) + dval)

    plt.tight_layout()
    if use_eps:
        plt.savefig(path_output + "/polymlp_eqm_properties.eps", format="eps")
    else:
        plt.savefig(path_output + "/polymlp_eqm_properties.png", format="png", dpi=dpi)
    plt.clf()
    plt.close()
