#!/usr/bin/env python
import os
from math import ceil, floor

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter


def plot_icsd_prediction(
    icsd_dict_array,
    system,
    pot_id,
    path_output="./",
    dpi=300,
    figsize=None,
    fontsize=12,
):

    os.makedirs(path_output, exist_ok=True)

    tag, error = [], []
    for icsd_dict in icsd_dict_array:
        tag.append(icsd_dict["prototype"].replace("icsd-", ""))
        error.append(icsd_dict["dft"] - icsd_dict["mlp"])
    error = np.abs(error) * 1000

    limmax = max(((max(error) // 5) + 1) * 5, 10)

    sns.set_context("paper", 1.0, {"lines.linewidth": 2})
    sns.set_style("whitegrid", {"grid.linestyle": "--"})

    if figsize is not None:
        fig, ax = plt.subplots(figsize=figsize)

    plt.subplots_adjust(bottom=0.25, top=0.9)
    plt.ylim(0, limmax)

    b = sns.barplot(x=tag, y=error, color="royalblue")
    b.axes.set_title(
        "Prediction error for ICSD prototypes (" + system + ", " + pot_id + ")",
        fontsize=fontsize,
    )
    b.set_xlabel("ICSD prototype", fontsize=fontsize)
    b.set_ylabel("Absolute error (meV/atom)", fontsize=fontsize)
    b.tick_params(axis="x", labelsize=int(fontsize * 0.6), labelrotation=90)
    b.tick_params(axis="y", labelsize=fontsize)

    plt.tight_layout()
    plt.savefig(path_output + "/polymlp_icsd_pred.png", format="png", dpi=dpi)
    plt.savefig(path_output + "/polymlp_icsd_pred.eps", format="eps")
    plt.clf()
    plt.close()


def plot_energy(data_train, data_test, system, pot_id, path_output="./", dpi=300):

    os.makedirs(path_output, exist_ok=True)

    limmin = min(min(data_train[:, 0]), min(data_test[:, 0]))
    # limmax = max(max(data_train[:, 0]), max(data_test[:, 0]))
    limmin_int = floor(min(min(data_train[:, 0]), min(data_test[:, 0])))
    limmax_int = ceil(min(max(max(data_train[:, 0]), max(data_test[:, 0])), 20))
    x = np.arange(limmin_int, limmax_int + 1)
    y = x

    sns.set_context("paper", 1.0, {"lines.linewidth": 1})
    sns.set_style("whitegrid", {"grid.linestyle": "--"})

    fig, ax = plt.subplots(2, 2, figsize=(5, 5))
    fig.suptitle("Energy distribution (" + system + ", " + pot_id + ")", fontsize=10)

    for i in range(2):
        for j in range(2):
            ax[i][j].set_aspect("equal")
            ax[i][j].plot(x, y, color="gray", linestyle="dashed", linewidth=0.25)
        ax[i][0].scatter(
            data_train[:, 0],
            data_train[:, 1],
            s=0.25,
            c="black",
            alpha=1.0,
            marker=".",
            label="training",
        )
        ax[i][1].scatter(
            data_test[:, 0],
            data_test[:, 1],
            s=0.25,
            c="orangered",
            alpha=1.0,
            marker=".",
            label="test",
        )
        for j in range(2):
            ax[i][j].set_xlabel("DFT energy (eV/atom)", fontsize=8)
            ax[i][j].set_ylabel("MLP energy (eV/atom)", fontsize=8)
            ax[i][j].tick_params(axis="both", labelsize=6)
            ax[i][j].legend()

    interval = max(int((limmax_int - limmin_int) / 10), 1)
    for j in range(2):
        ax[0][j].set_xlim(limmin_int, limmax_int)
        ax[0][j].set_ylim(limmin_int, limmax_int)
        ax[0][j].set_xticks(np.arange(limmin_int, limmax_int + 1, interval))
        ax[0][j].set_yticks(np.arange(limmin_int, limmax_int + 1, interval))
        ax[1][j].set_xlim(limmin - 0.05, limmin + 1.05)
        ax[1][j].set_ylim(limmin - 0.05, limmin + 1.05)

    plt.tight_layout()
    plt.savefig(path_output + "/distribution.png", format="png", dpi=dpi)
    plt.savefig(path_output + "/distribution.eps", format="eps")
    plt.clf()
    plt.close()


def plot_eos(eos_dict, system, pot_id, emin=None, path_output="./", dpi=300):

    os.makedirs(path_output, exist_ok=True)

    v_array = np.ravel([ev[:, 0] for ev in eos_dict.values()])
    e_array = np.ravel([ev[:, 1] for ev in eos_dict.values()])

    limmin_x = np.min(v_array) - 1
    limmax_x = np.max(v_array) + 1
    limmax_y = np.max(e_array) + 0.1

    if emin is None:
        limmin_y = np.min(e_array) - 0.1
    else:
        limmin_y = emin - 0.1

    sns.set_context("paper", 1.0, {"lines.linewidth": 1})
    sns.set_style("whitegrid", {"grid.linestyle": "--"})
    sns.set_palette("hls", len(eos_dict))

    fig, ax = plt.subplots(1, 3, figsize=(9, 4))
    fig.suptitle("Equation of state (" + system + ", " + pot_id + ")", fontsize=10)

    for st, ev in eos_dict.items():
        ax[0].scatter(
            ev[:, 0],
            ev[:, 1],
            s=5,
            label=st,
            marker="o",
            linewidths=0.1,
            edgecolors="k",
        )
        ax[1].scatter(
            ev[:, 0],
            ev[:, 1],
            s=8,
            label=st,
            marker="o",
            linewidths=0.1,
            edgecolors="k",
        )
        ax[2].scatter(
            ev[:, 0],
            ev[:, 1],
            s=10,
            label=st,
            marker="o",
            linewidths=0.1,
            edgecolors="k",
        )

    ax[0].set_xlabel("Volume ($\mathrm{\AA}^3$/atom)", fontsize=10)
    ax[0].set_ylabel("Energy (eV/atom)", fontsize=10)
    ax[0].set_xlim(limmin_x, limmax_x)
    ax[0].set_ylim(limmin_y, limmax_y)
    ax[0].tick_params(axis="both", labelsize=8)
    ax[0].legend(fontsize=7)

    if emin is None:
        emin_i = np.argmin(e_array)
        emin_v = v_array[emin_i]
    else:
        ev1 = [(e, v) for e, v in zip(e_array, v_array) if e > emin]
        emin_v = min(ev1, key=lambda x: x[0])[1]
    limmin_y += 0.09

    ax[1].set_xlabel("Volume ($\mathrm{\AA}^3$/atom)", fontsize=10)
    ax[1].set_xlim(emin_v - 10, emin_v + 10)
    ax[1].set_ylim(limmin_y, limmin_y + 0.5)
    ax[1].tick_params(axis="both", labelsize=8)

    ax[2].set_xlabel("Volume ($\mathrm{\AA}^3$/atom)", fontsize=10)
    ax[2].set_xlim(emin_v - 3, emin_v + 3)
    ax[2].set_ylim(limmin_y, limmin_y + 0.1)
    ax[2].tick_params(axis="both", labelsize=8)

    plt.tight_layout()
    plt.savefig(path_output + "/polymlp_eos.png", format="png", dpi=dpi)
    plt.savefig(path_output + "/polymlp_eos.eps", format="eps")
    plt.clf()
    plt.close()


def plot_eos_separate(
    eos_dict,
    system,
    pot_id,
    emin=None,
    path_output="./",
    n_cols=3,
    figsize=(8, 12),
    dpi=300,
    fontsize=10,
):

    os.makedirs(path_output, exist_ok=True)

    v_array = np.ravel([ev[:, 0] for ev in eos_dict.values()])
    e_array = np.ravel([ev[:, 1] for ev in eos_dict.values()])

    limmin_x = np.min(v_array) - 1
    limmax_x = np.max(v_array) + 1

    if emin is None:
        limmin_y = np.min(e_array) - 0.1
    else:
        limmin_y = emin - 0.1
    limmax_y = min(max(e_array) + 0.1, 2)

    sns.set_palette("hls", len(eos_dict))
    sns.set_context("paper", 1.0, {"lines.linewidth": 1})
    sns.set_style("whitegrid", {"grid.linestyle": "--"})

    if len(eos_dict) % n_cols == 0:
        n_rows = round(len(eos_dict) / n_cols)
    else:
        n_rows = ceil(len(eos_dict) / n_cols)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle("Equation of state (" + system + ", " + pot_id + ")", fontsize=12)

    for i, (st, ev) in enumerate(eos_dict.items()):
        row = i // n_cols
        col = i % n_cols
        ax[row][col].scatter(
            ev[:, 0],
            ev[:, 1],
            color="turquoise",
            s=6,
            marker="o",
            linewidths=0.1,
            edgecolors="k",
        )
        ax[row][col].set_title(st, fontsize=fontsize, loc="left")
        ax[row][col].set_xlim(limmin_x, limmax_x)
        ax[row][col].set_ylim(limmin_y, limmax_y)
        ax[row][col].tick_params(axis="both", labelsize=fontsize)

    for i in range(n_rows):
        ax[i][0].set_ylabel("Energy (eV/atom)", fontsize=fontsize)

    for i in range(n_cols):
        ax[-1][i].set_xlabel("Volume ($\mathrm{\AA}^3$/atom)", fontsize=fontsize)
        ax[-1][i].tick_params(axis="both", labelsize=fontsize)

    plt.tight_layout()
    plt.savefig(path_output + "/polymlp_eos_sep.png", format="png", dpi=dpi)
    plt.savefig(path_output + "/polymlp_eos_sep.eps", format="eps")
    plt.clf()
    plt.close()


def plot_phonon(
    phonon_dict,
    system,
    pot_id,
    path_output="./",
    dpi=300,
    n_cols=3,
    figsize=(8, 8),
    fontsize=10,
):

    os.makedirs(path_output, exist_ok=True)

    freq_array = [d[:, 0] for d in phonon_dict.values()]
    val_array = [d[:, 1] for d in phonon_dict.values()]

    limmin_x = floor(np.min(freq_array))
    limmax_x = ceil(np.max(freq_array))
    limmin_y = -0.01
    limmax_y = np.max(val_array) * 2 / 3

    sns.set_context("paper", 1.0, {"lines.linewidth": 2})
    sns.set_style("whitegrid", {"grid.linestyle": "--"})

    if len(phonon_dict) % n_cols == 0:
        n_rows = round(len(phonon_dict) / n_cols)
    else:
        n_rows = ceil(len(phonon_dict) / n_cols)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle("Phonon DOS (" + system + ", " + pot_id + ")", fontsize=10)
    for i, (st, dos) in enumerate(phonon_dict.items()):
        row = i // n_cols
        col = i % n_cols
        ax[row][col].plot(dos[:, 0], dos[:, 1], color="darkcyan", linewidth=1)
        ax[row][col].set_title(st, fontsize=fontsize, loc="left")
        ax[row][col].set_xlim(limmin_x, limmax_x)
        ax[row][col].set_ylim(limmin_y, limmax_y)
        ax[row][col].tick_params(axis="both", labelsize=8)

        # ax[row][col].axvline(0, color='black')

    for i in range(n_rows):
        ax[i][0].set_ylabel("DOS", fontsize=fontsize)
    for i in range(n_cols):
        ax[-1][i].set_xlabel("Frequency (THz)", fontsize=fontsize)
        ax[-1][i].tick_params(axis="both", labelsize=8)

    plt.tight_layout()
    plt.savefig(path_output + "/polymlp_phonon_dos.png", format="png", dpi=dpi)
    plt.savefig(path_output + "/polymlp_phonon_dos.eps", format="eps")
    plt.clf()
    plt.close()


def plot_phonon_qha_thermal_expansion(
    thermal_expansion_dict,
    system,
    pot_id,
    path_output="./",
    dpi=300,
    n_cols=3,
    figsize=(8, 6),
    fontsize=10,
):

    os.makedirs(path_output, exist_ok=True)

    val_array = [d[:, 1] for d in thermal_expansion_dict.values()]

    limmin_x = 0
    limmax_x = 1000
    limmin_y = min(np.min(val_array) - 0.1 * np.max(val_array), 0)
    limmax_y = np.max(val_array) * 1.1

    sns.set_context("paper", 1.0, {"lines.linewidth": 2})
    sns.set_style("whitegrid", {"grid.linestyle": "--"})

    if len(thermal_expansion_dict) % n_cols == 0:
        n_rows = round(len(thermal_expansion_dict) / n_cols)
    else:
        n_rows = ceil(len(thermal_expansion_dict) / n_cols)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle("Thermal expansion (" + system + ", " + pot_id + ")", fontsize=10)
    for i, (st, te) in enumerate(thermal_expansion_dict.items()):
        row = i // n_cols
        col = i % n_cols
        ax_obj = ax[col] if n_rows == 1 else ax[row][col]
        ax_obj.plot(te[:, 0], te[:, 1], color="mediumvioletred", linewidth=1)
        ax_obj.set_title(st, fontsize=fontsize, loc="left")
        ax_obj.set_xlim(limmin_x, limmax_x)
        ax_obj.set_ylim(limmin_y, limmax_y)
        ax_obj.tick_params(axis="both", labelsize=8)
        ax_obj.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax_obj.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    for i in range(n_rows):
        ax_obj = ax[0] if n_rows == 1 else ax[i][0]
        ax_obj.set_ylabel("Thermal expansion (K$^{-1}$)", fontsize=fontsize)

    for i in range(n_cols):
        ax_obj = ax[i] if n_rows == 1 else ax[-1][i]
        ax_obj.set_xlabel("Temperature (K)", fontsize=fontsize)
        ax_obj.tick_params(axis="both", labelsize=8)

    plt.tight_layout()
    plt.savefig(
        path_output + "/polymlp_thermal_expansion.png",
        format="png",
        dpi=dpi,
    )
    plt.savefig(path_output + "/polymlp_thermal_expansion.eps", format="eps")
    plt.clf()
    plt.close()


def plot_phonon_qha_bulk_modulus(
    bm_dict,
    system,
    pot_id,
    path_output="./",
    dpi=300,
    n_cols=3,
    figsize=(8, 6),
    fontsize=10,
):

    os.makedirs(path_output, exist_ok=True)

    val_array = [d[:, 1] for d in bm_dict.values()]

    limmin_x = 0
    limmax_x = 1000
    limmin_y = min(np.min(val_array) - 0.1 * np.max(val_array), 0)
    limmax_y = np.max(val_array) * 1.1

    sns.set_context("paper", 1.0, {"lines.linewidth": 2})
    sns.set_style("whitegrid", {"grid.linestyle": "--"})

    if len(bm_dict) % n_cols == 0:
        n_rows = round(len(bm_dict) / n_cols)
    else:
        n_rows = ceil(len(bm_dict) / n_cols)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle("Bulk modulus (" + system + ", " + pot_id + ")", fontsize=10)

    for i, (st, te) in enumerate(bm_dict.items()):
        row = i // n_cols
        col = i % n_cols
        ax_obj = ax[col] if n_rows == 1 else ax[row][col]
        ax_obj.plot(te[:, 0], te[:, 1], color="mediumvioletred", linewidth=1)
        ax_obj.set_title(st, fontsize=fontsize, loc="left")
        ax_obj.set_xlim(limmin_x, limmax_x)
        ax_obj.set_ylim(limmin_y, limmax_y)
        ax_obj.tick_params(axis="both", labelsize=8)
        ax_obj.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    for i in range(n_rows):
        ax_obj = ax[0] if n_rows == 1 else ax[i][0]
        ax_obj.set_ylabel("Bulk modulus (GPa)", fontsize=fontsize)

    for i in range(n_cols):
        ax_obj = ax[i] if n_rows == 1 else ax[-1][i]
        ax_obj.set_xlabel("Temperature (K)", fontsize=fontsize)
        ax_obj.tick_params(axis="both", labelsize=8)

    plt.tight_layout()
    plt.savefig(path_output + "/polymlp_bulk_modulus.png", format="png", dpi=dpi)
    plt.savefig(path_output + "/polymlp_bulk_modulus.eps", format="eps")
    plt.clf()
    plt.close()
