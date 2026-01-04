"""Utility functions for plotting properties."""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pypolymlp.calculator.auto.autocalc_utils import Prototype

# from matplotlib.ticker import ScalarFormatter
# from math import ceil, floor


def plot_prototype_prediction(
    data: np.ndarray,
    system: str,
    pot_id: str,
    path_output: str = "./",
    dpi: int = 300,
    figsize: tuple = None,
    fontsize: int = 10,
    use_eps: bool = False,
):
    """Plot errors between DFT and MLP for prototype structures.

    Parameters
    ----------
    data: DFT and MLP energies for prototype structures.
          First column: DFT energies.
          Second column: MLP energies.
          Third column: Structure identifiers.
    system: System.
    pot_id: Potential ID.
    """
    os.makedirs(path_output, exist_ok=True)
    error = np.abs(data[:, 0].astype(float) - data[:, 1].astype(float)) * 1000
    tag = data[:, 2]

    limmax = max(((max(error) // 5) + 1) * 5, 10)

    sns.set_context("paper", 1.0, {"lines.linewidth": 2})
    sns.set_style("whitegrid", {"grid.linestyle": "--"})

    if figsize is not None:
        fig, ax = plt.subplots(figsize=figsize)

    plt.subplots_adjust(bottom=0.25, top=0.9)
    plt.ylim(0, limmax)

    b = sns.barplot(x=tag, y=error, color="royalblue")
    b.axes.set_title(
        "Prediction error for prototypes (" + system + ", " + pot_id + ")",
        fontsize=fontsize,
    )
    b.set_xlabel("Prototype", fontsize=fontsize)
    b.set_ylabel("Absolute error (meV/atom)", fontsize=fontsize)
    b.tick_params(axis="x", labelsize=int(fontsize * 0.6), labelrotation=90)
    b.tick_params(axis="y", labelsize=fontsize)

    plt.tight_layout()
    if use_eps:
        plt.savefig(path_output + "/polymlp_comparison.eps", format="eps")
    else:
        plt.savefig(path_output + "/polymlp_comparison.png", format="png", dpi=dpi)
    plt.clf()
    plt.close()


def _set_eos_minmax(
    prototypes: list[Prototype],
    emin: Optional[float] = None,
    use_enlarged: bool = True,
):
    """Set minimum and maximum values for EOS plot."""
    limmin_x, limmax_x, limmin_y, limmax_y = [], [], [], []
    v_array = np.ravel([p.eos_mlp[:, 0] for p in prototypes])
    e_array = np.ravel([p.eos_mlp[:, 1] for p in prototypes])

    vmin, vmax = np.min(v_array), np.max(v_array)
    emax = np.max(e_array)
    if emin is None:
        emin = np.min(e_array)
        v_emin = v_array[np.argmin(e_array)]
    else:
        ev1 = [(e, v) for e, v in zip(e_array, v_array) if e > emin]
        v_emin = min(ev1, key=lambda x: x[0])[1]

    limmin_x.append(vmin - 1)
    limmax_x.append(vmax + 1)
    limmin_y.append(emin - 0.1)
    limmax_y.append(emax + 0.1)

    if not use_enlarged:
        return (limmin_x[0], limmax_x[0]), (limmin_y[0], limmax_y[0])

    limmin_x.append(v_emin - 8)
    limmax_x.append(v_emin + 8)
    limmin_y.append(emin - 0.01)
    limmax_y.append(emin + 0.5)

    limmin_x.append(v_emin - 3)
    limmax_x.append(v_emin + 3)
    limmin_y.append(emin - 0.01)
    limmax_y.append(emin + 0.1)

    return (limmin_x, limmax_x), (limmin_y, limmax_y)


def plot_eos(
    prototypes: list[Prototype],
    system: str,
    pot_id: str,
    emin: Optional[float] = None,
    path_output: str = "./",
    use_eps: bool = False,
    dpi: int = 300,
):
    """Plot EOS functions for prototype structures."""
    os.makedirs(path_output, exist_ok=True)
    (limmin_x, limmax_x), (limmin_y, limmax_y) = _set_eos_minmax(prototypes, emin=emin)

    sns.set_context("paper", 1.0, {"lines.linewidth": 1})
    sns.set_style("whitegrid", {"grid.linestyle": "--"})
    sns.set_palette("Paired", len(prototypes), 1)

    fig, ax = plt.subplots(1, 3, figsize=(9, 4))
    fig.suptitle("Equation of state (" + system + ", " + pot_id + ")", fontsize=10)

    marker_candidates = ["o", "^", "d"]
    marker_sizes = [5, 8, 10]
    for i, prot in enumerate(prototypes):
        st, ev = prot.name, prot.eos_mlp
        marker = marker_candidates[i // 12]
        for j in range(3):
            ax[j].scatter(
                ev[:, 0],
                ev[:, 1],
                s=marker_sizes[j],
                label=st,
                marker=marker,
                linewidths=0.1,
                edgecolors="k",
            )
    ax[0].set_ylabel("Energy (eV/atom)", fontsize=10)
    ax[0].legend(fontsize=7)
    for i in range(3):
        ax[i].set_xlabel(r"Volume ($\mathrm{\AA}^3$/atom)", fontsize=10)
        ax[i].set_xlim(limmin_x[i], limmax_x[i])
        ax[i].set_ylim(limmin_y[i], limmax_y[i])
        ax[i].tick_params(axis="both", labelsize=8)

    plt.tight_layout()
    if use_eps:
        plt.savefig(path_output + "/polymlp_eos.eps", format="eps")
    else:
        plt.savefig(path_output + "/polymlp_eos.png", format="png", dpi=dpi)
    plt.clf()
    plt.close()


def plot_eos_separate(
    prototypes: list[Prototype],
    system: str,
    pot_id: str,
    emin: Optional[float] = None,
    path_output: str = "./",
    use_eps: bool = False,
    dpi: int = 300,
    n_cols: int = 3,
    figsize: tuple = (8, 12),
    fontsize: int = 10,
):
    """Plot EOS functions for prototype structures separately."""
    os.makedirs(path_output, exist_ok=True)
    (limmin_x, limmax_x), (limmin_y, limmax_y) = _set_eos_minmax(
        prototypes, emin=emin, use_enlarged=False
    )
    limmax_y = min(limmax_y, 2.0)
    n_rows = int(np.ceil(len(prototypes) / n_cols - 1e-10))

    sns.set_context("paper", 1.0, {"lines.linewidth": 1})
    sns.set_style("whitegrid", {"grid.linestyle": "--"})
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle("Equation of state (" + system + ", " + pot_id + ")", fontsize=12)

    for i, prot in enumerate(prototypes):
        st, ev = prot.name, prot.eos_mlp
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
        ax[-1][i].set_xlabel(r"Volume ($\mathrm{\AA}^3$/atom)", fontsize=fontsize)
        ax[-1][i].tick_params(axis="both", labelsize=fontsize)

    plt.tight_layout()
    if use_eps:
        plt.savefig(path_output + "/polymlp_eos_sep.eps", format="eps")
    else:
        plt.savefig(path_output + "/polymlp_eos_sep.png", format="png", dpi=dpi)
    plt.clf()
    plt.close()


def _set_phonon_minmax(prototypes: list[Prototype]):
    """Set minimum and maximum values for phonon DOS plot."""
    freq_array = np.ravel([p.phonon_dos[:, 0] for p in prototypes])
    dos_array = np.ravel([p.phonon_dos[:, 1] / p.n_atom for p in prototypes])

    limmin_x = np.floor(np.min(freq_array))
    limmax_x = np.ceil(np.max(freq_array))
    limmin_y = -0.01
    limmax_y = np.max(dos_array) * 2 / 3

    return (limmin_x, limmax_x), (limmin_y, limmax_y)


def plot_phonon(
    prototypes: list[Prototype],
    system: str,
    pot_id: str,
    path_output: str = "./",
    use_eps: bool = False,
    dpi: int = 300,
    n_cols: int = 3,
    figsize: tuple = (8, 8),
    fontsize: int = 10,
):
    """Plot phonon DOS for prototype structures separately."""
    os.makedirs(path_output, exist_ok=True)

    (limmin_x, limmax_x), (limmin_y, limmax_y) = _set_phonon_minmax(prototypes)
    n_rows = int(np.ceil(len(prototypes) / n_cols - 1e-10))

    sns.set_context("paper", 1.0, {"lines.linewidth": 2})
    sns.set_style("whitegrid", {"grid.linestyle": "--"})
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle("Phonon DOS (" + system + ", " + pot_id + ")", fontsize=10)

    for i, prot in enumerate(prototypes):
        st, dos = prot.name, prot.phonon_dos
        row = i // n_cols
        col = i % n_cols
        ax[row][col].plot(
            dos[:, 0], dos[:, 1] / prot.n_atom, color="darkcyan", linewidth=1
        )
        ax[row][col].set_title(st, fontsize=fontsize, loc="left")
        ax[row][col].set_xlim(limmin_x, limmax_x)
        ax[row][col].set_ylim(limmin_y, limmax_y)
        ax[row][col].tick_params(axis="both", labelsize=8)

    for i in range(n_rows):
        ax[i][0].set_ylabel("DOS", fontsize=fontsize)
    for i in range(n_cols):
        ax[-1][i].set_xlabel("Frequency (THz)", fontsize=fontsize)
        ax[-1][i].tick_params(axis="both", labelsize=8)

    plt.tight_layout()
    if use_eps:
        plt.savefig(path_output + "/polymlp_phonon_dos.eps", format="eps")
    else:
        plt.savefig(path_output + "/polymlp_phonon_dos.png", format="png", dpi=dpi)
    plt.clf()
    plt.close()


# def plot_energy(data_train, data_test, system, pot_id, path_output="./", dpi=300):
#
#     os.makedirs(path_output, exist_ok=True)
#
#     limmin = min(min(data_train[:, 0]), min(data_test[:, 0]))
#     # limmax = max(max(data_train[:, 0]), max(data_test[:, 0]))
#     limmin_int = floor(min(min(data_train[:, 0]), min(data_test[:, 0])))
#     limmax_int = ceil(min(max(max(data_train[:, 0]), max(data_test[:, 0])), 20))
#     x = np.arange(limmin_int, limmax_int + 1)
#     y = x
#
#     sns.set_context("paper", 1.0, {"lines.linewidth": 1})
#     sns.set_style("whitegrid", {"grid.linestyle": "--"})
#
#     fig, ax = plt.subplots(2, 2, figsize=(5, 5))
#     fig.suptitle("Energy distribution (" + system + ", " + pot_id + ")", fontsize=10)
#
#     for i in range(2):
#         for j in range(2):
#             ax[i][j].set_aspect("equal")
#             ax[i][j].plot(x, y, color="gray", linestyle="dashed", linewidth=0.25)
#         ax[i][0].scatter(
#             data_train[:, 0],
#             data_train[:, 1],
#             s=0.25,
#             c="black",
#             alpha=1.0,
#             marker=".",
#             label="training",
#         )
#         ax[i][1].scatter(
#             data_test[:, 0],
#             data_test[:, 1],
#             s=0.25,
#             c="orangered",
#             alpha=1.0,
#             marker=".",
#             label="test",
#         )
#         for j in range(2):
#             ax[i][j].set_xlabel("DFT energy (eV/atom)", fontsize=8)
#             ax[i][j].set_ylabel("MLP energy (eV/atom)", fontsize=8)
#             ax[i][j].tick_params(axis="both", labelsize=6)
#             ax[i][j].legend()
#
#     interval = max(int((limmax_int - limmin_int) / 10), 1)
#     for j in range(2):
#         ax[0][j].set_xlim(limmin_int, limmax_int)
#         ax[0][j].set_ylim(limmin_int, limmax_int)
#         ax[0][j].set_xticks(np.arange(limmin_int, limmax_int + 1, interval))
#         ax[0][j].set_yticks(np.arange(limmin_int, limmax_int + 1, interval))
#         ax[1][j].set_xlim(limmin - 0.05, limmin + 1.05)
#         ax[1][j].set_ylim(limmin - 0.05, limmin + 1.05)
#
#     plt.tight_layout()
#     plt.savefig(path_output + "/distribution.png", format="png", dpi=dpi)
#     plt.savefig(path_output + "/distribution.eps", format="eps")
#     plt.clf()
#     plt.close()
#
#

#
# def plot_phonon(
#     phonon_dict,
#     system,
#     pot_id,
#     path_output="./",
#     dpi=300,
#     n_cols=3,
#     figsize=(8, 8),
#     fontsize=10,
# ):
#
#     os.makedirs(path_output, exist_ok=True)
#
#     freq_array = [d[:, 0] for d in phonon_dict.values()]
#     val_array = [d[:, 1] for d in phonon_dict.values()]
#
#     limmin_x = floor(np.min(freq_array))
#     limmax_x = ceil(np.max(freq_array))
#     limmin_y = -0.01
#     limmax_y = np.max(val_array) * 2 / 3
#
#     sns.set_context("paper", 1.0, {"lines.linewidth": 2})
#     sns.set_style("whitegrid", {"grid.linestyle": "--"})
#
#     if len(phonon_dict) % n_cols == 0:
#         n_rows = round(len(phonon_dict) / n_cols)
#     else:
#         n_rows = ceil(len(phonon_dict) / n_cols)
#     fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
#     fig.suptitle("Phonon DOS (" + system + ", " + pot_id + ")", fontsize=10)
#     for i, (st, dos) in enumerate(phonon_dict.items()):
#         row = i // n_cols
#         col = i % n_cols
#         ax[row][col].plot(dos[:, 0], dos[:, 1], color="darkcyan", linewidth=1)
#         ax[row][col].set_title(st, fontsize=fontsize, loc="left")
#         ax[row][col].set_xlim(limmin_x, limmax_x)
#         ax[row][col].set_ylim(limmin_y, limmax_y)
#         ax[row][col].tick_params(axis="both", labelsize=8)
#
#         # ax[row][col].axvline(0, color='black')
#
#     for i in range(n_rows):
#         ax[i][0].set_ylabel("DOS", fontsize=fontsize)
#     for i in range(n_cols):
#         ax[-1][i].set_xlabel("Frequency (THz)", fontsize=fontsize)
#         ax[-1][i].tick_params(axis="both", labelsize=8)
#
#     plt.tight_layout()
#     plt.savefig(path_output + "/polymlp_phonon_dos.png", format="png", dpi=dpi)
#     plt.savefig(path_output + "/polymlp_phonon_dos.eps", format="eps")
#     plt.clf()
#     plt.close()
#
#
# def plot_phonon_qha_thermal_expansion(
#     thermal_expansion_dict,
#     system,
#     pot_id,
#     path_output="./",
#     dpi=300,
#     n_cols=3,
#     figsize=(8, 6),
#     fontsize=10,
# ):
#
#     os.makedirs(path_output, exist_ok=True)
#
#     val_array = [d[:, 1] for d in thermal_expansion_dict.values()]
#
#     limmin_x = 0
#     limmax_x = 1000
#     limmin_y = min(np.min(val_array) - 0.1 * np.max(val_array), 0)
#     limmax_y = np.max(val_array) * 1.1
#
#     sns.set_context("paper", 1.0, {"lines.linewidth": 2})
#     sns.set_style("whitegrid", {"grid.linestyle": "--"})
#
#     if len(thermal_expansion_dict) % n_cols == 0:
#         n_rows = round(len(thermal_expansion_dict) / n_cols)
#     else:
#         n_rows = ceil(len(thermal_expansion_dict) / n_cols)
#
#     fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
#     fig.suptitle("Thermal expansion (" + system + ", " + pot_id + ")", fontsize=10)
#     for i, (st, te) in enumerate(thermal_expansion_dict.items()):
#         row = i // n_cols
#         col = i % n_cols
#         ax_obj = ax[col] if n_rows == 1 else ax[row][col]
#         ax_obj.plot(te[:, 0], te[:, 1], color="mediumvioletred", linewidth=1)
#         ax_obj.set_title(st, fontsize=fontsize, loc="left")
#         ax_obj.set_xlim(limmin_x, limmax_x)
#         ax_obj.set_ylim(limmin_y, limmax_y)
#         ax_obj.tick_params(axis="both", labelsize=8)
#         ax_obj.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#         ax_obj.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
#
#     for i in range(n_rows):
#         ax_obj = ax[0] if n_rows == 1 else ax[i][0]
#         ax_obj.set_ylabel("Thermal expansion (K$^{-1}$)", fontsize=fontsize)
#
#     for i in range(n_cols):
#         ax_obj = ax[i] if n_rows == 1 else ax[-1][i]
#         ax_obj.set_xlabel("Temperature (K)", fontsize=fontsize)
#         ax_obj.tick_params(axis="both", labelsize=8)
#
#     plt.tight_layout()
#     plt.savefig(
#         path_output + "/polymlp_thermal_expansion.png",
#         format="png",
#         dpi=dpi,
#     )
#     plt.savefig(path_output + "/polymlp_thermal_expansion.eps", format="eps")
#     plt.clf()
#     plt.close()
#
#
# def plot_phonon_qha_bulk_modulus(
#     bm_dict,
#     system,
#     pot_id,
#     path_output="./",
#     dpi=300,
#     n_cols=3,
#     figsize=(8, 6),
#     fontsize=10,
# ):
#
#     os.makedirs(path_output, exist_ok=True)
#
#     val_array = [d[:, 1] for d in bm_dict.values()]
#
#     limmin_x = 0
#     limmax_x = 1000
#     limmin_y = min(np.min(val_array) - 0.1 * np.max(val_array), 0)
#     limmax_y = np.max(val_array) * 1.1
#
#     sns.set_context("paper", 1.0, {"lines.linewidth": 2})
#     sns.set_style("whitegrid", {"grid.linestyle": "--"})
#
#     if len(bm_dict) % n_cols == 0:
#         n_rows = round(len(bm_dict) / n_cols)
#     else:
#         n_rows = ceil(len(bm_dict) / n_cols)
#
#     fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
#     fig.suptitle("Bulk modulus (" + system + ", " + pot_id + ")", fontsize=10)
#
#     for i, (st, te) in enumerate(bm_dict.items()):
#         row = i // n_cols
#         col = i % n_cols
#         ax_obj = ax[col] if n_rows == 1 else ax[row][col]
#         ax_obj.plot(te[:, 0], te[:, 1], color="mediumvioletred", linewidth=1)
#         ax_obj.set_title(st, fontsize=fontsize, loc="left")
#         ax_obj.set_xlim(limmin_x, limmax_x)
#         ax_obj.set_ylim(limmin_y, limmax_y)
#         ax_obj.tick_params(axis="both", labelsize=8)
#         ax_obj.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#
#     for i in range(n_rows):
#         ax_obj = ax[0] if n_rows == 1 else ax[i][0]
#         ax_obj.set_ylabel("Bulk modulus (GPa)", fontsize=fontsize)
#
#     for i in range(n_cols):
#         ax_obj = ax[i] if n_rows == 1 else ax[-1][i]
#         ax_obj.set_xlabel("Temperature (K)", fontsize=fontsize)
#         ax_obj.tick_params(axis="both", labelsize=8)
#
#     plt.tight_layout()
#     plt.savefig(path_output + "/polymlp_bulk_modulus.png", format="png", dpi=dpi)
#     plt.savefig(path_output + "/polymlp_bulk_modulus.eps", format="eps")
#     plt.clf()
#     plt.close()
