#!/usr/bin/env python
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_mlp_distribution(d1_array, d2_array, system, path_output="./", dpi=300):
    """
    Parameters
    ----------
    d1_array: all,
    d2_array: convex
    """

    os.makedirs(path_output, exist_ok=True)
    if not isinstance(d1_array, np.ndarray):
        d1_array = np.array(d1_array)
    if not isinstance(d2_array, np.ndarray):
        d2_array = np.array(d2_array)

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

    plt.title("Optimal MLPs (" + system + ")")
    plt.xlabel("Elapsed time [s/atom/step] (Single CPU core)")
    plt.ylabel("RMS error [meV/atom]")
    sns.set_style("whitegrid", {"grid.linestyle": "--"})
    plt.grid(which="minor", alpha=0.2)

    plt.scatter(
        d1_array[:, 0].astype(float) / 1000,
        d1_array[:, 2].astype(float),
        s=10,
        c="black",
        marker=".",
        alpha=1.0,
    )
    plt.plot(
        d2_array[:, 0].astype(float) / 1000,
        d2_array[:, 2].astype(float),
        marker=".",
        alpha=1.0,
        markersize=9,
        linewidth=0.8,
        markeredgecolor="k",
        markeredgewidth=0.4,
    )
    plt.savefig(path_output + "/mlp_dist.png", format="png", dpi=dpi)
    plt.savefig(path_output + "/mlp_dist.eps", format="eps")
    plt.clf()
    plt.close()


def plot_eqm_properties(eqm_props_dict, system, path_output="./", dpi=300):
    """
    Parameters
    ----------
    eqm_props_dict: key: structure, value: [time, ecoh, volume, bm]
    """
    os.makedirs(path_output, exist_ok=True)

    plt.style.use("bmh")
    sns.set_context("paper", 1.0, {"lines.linewidth": 4})
    sns.set_palette("Paired", len(eqm_props_dict), 1)

    fig, ax = plt.subplots(1, 3, figsize=(7, 3))

    max_vals, min_vals = [[], [], []], [[], [], []]
    for st, list1 in eqm_props_dict.items():
        for i in range(3):
            ax[i].plot(
                list1[:, 0] / 1000,
                list1[:, i + 1],
                marker=".",
                markersize=6,
                linewidth=0.8,
                markeredgecolor="k",
                markeredgewidth=0.3,
                label=st,
            )
            max_vals[i].append(max(list1[:, i + 1][-8:]))
            min_vals[i].append(min(list1[:, i + 1][-8:]))

    xlabel = "Elapsed time [s/atom/step]"
    ax[0].set_title("Cohesive energy (" + system + ", Optimal MLPs)", fontsize=7)
    ax[0].set_xlabel(xlabel, fontsize=6)
    ax[0].set_ylabel("Cohesive energy [eV/atom]", fontsize=6)

    ax[1].set_title("Equilibrium volume (" + system + ", Optimal MLPs)", fontsize=7)
    ax[1].set_xlabel(xlabel, fontsize=6)
    ax[1].set_ylabel("Volume [$\mathrm{\AA}^3$/atom]", fontsize=6)

    ax[2].set_title("Bulk modulus (" + system + ", Optimal MLPs)", fontsize=7)
    ax[2].set_xlabel(xlabel, fontsize=6)
    ax[2].set_ylabel("Bulk modulus [GPa]", fontsize=6)

    for i in range(3):
        ax[i].legend(
            bbox_to_anchor=(0, 1),
            loc="upper left",
            borderaxespad=0,
            fontsize=6,
        )
        ax[i].tick_params(axis="both", labelsize=6)
        ax[i].set_xlim(2e-6, 5e-2)
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
    plt.savefig(path_output + "/eqm_properties.png", format="png", dpi=dpi)
    plt.savefig(path_output + "/eqm_properties.eps", format="eps")
    plt.clf()
    plt.close()
