#!/usr/bin/env python
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def plot_mlp_distribution(d1_array, 
                          d2_array, 
                          system,
                          path_output='./',
                          dpi=300):

    '''
    Parameters
    ----------
    d1_array: all, 
    d2_array: convex
    '''

    if not isinstance(d1_array, np.ndarray):
        d1_array = np.array(d1_array)
    if not isinstance(d2_array, np.ndarray):
        d2_array = np.array(d2_array)

    plt.style.use('bmh')
    sns.set_context("paper", 1.0, {"lines.linewidth": 4})
    sns.set_palette("coolwarm_r", 8, 1)

    plt.xlim(1e-5,5e-2)
    plt.ylim(0,25)
    plt.xscale('log')
    major_ticks = np.arange(0,26,5)
    minor_ticks = np.arange(0,26,1)
    plt.yticks(major_ticks)
    plt.yticks(minor_ticks, minor=True)

    plt.title("Optimal MLPs (" + system + ")")
    plt.xlabel("Elapsed time [s/atom/step] (Single CPU core)")
    plt.ylabel("RMS error [meV/atom]")
    sns.set_style('whitegrid', {'grid.linestyle': '--'})
    plt.grid(which='minor', alpha=0.2)

    plt.scatter(d1_array[:,0].astype(float) / 1000,
                d1_array[:,2].astype(float),
                s=10, c='black', marker='.', alpha=1.0)
    plt.plot(d2_array[:,0].astype(float) / 1000,
             d2_array[:,2].astype(float),
             marker='.', alpha=1.0, markersize=9,
             linewidth=0.8, markeredgecolor='k',markeredgewidth=0.4)
    plt.savefig(path_output + '/mlp_dist.png',format='png',dpi=dpi)
    plt.savefig(path_output + '/mlp_dist.eps',format='eps')
    plt.clf()


def plot_ecoh_volume(energy_volume_dict,
                     system,
                     output_dir,
                     dpi=300):

    plt.style.use('bmh')
    sns.set_context("paper", 1.0, {"lines.linewidth": 4})
    sns.set_palette("Paired", len(energy_volume_dict), 1)

    fig, ax = plt.subplots(1, 2, figsize=(7,3))

    ecoh_max, ecoh_min, vol_max, vol_min = [], [], [], []
    for st, list1 in energy_volume_dict.items():
        times1, ecoh1, volume1 = [], [], []
        for t, e, v in list1:
            times1.append(t)
            ecoh1.append(e)
            volume1.append(v)
        ax[0].plot(times1,
                   ecoh1,
                   marker='.',
                   markersize=6,
                   linewidth=0.8,
                   markeredgecolor='k',
                   markeredgewidth=0.3,
                   label=st)
        ax[1].plot(times1,
                   volume1,
                   marker='.',
                   markersize=6,
                   linewidth=0.8,
                   markeredgecolor='k',
                   markeredgewidth=0.3,
                   label=st)
        ecoh_max.append(max(ecoh1[-5:]))
        ecoh_min.append(min(ecoh1[-5:]))
        vol_max.append(max(volume1[-5:]))
        vol_min.append(min(volume1[-5:]))


'''
def plot_icsd(icsd_dict,
              pot_dict,
              dpi=200,
              figsize=None,
              fontsize=10):

    system = pot_dict['system']
    potname = pot_dict['id']
    output_dir = pot_dict['target'] + '/plot-icsd/'
    os.makedirs(output_dir, exist_ok=True)

    tag, error = [], []
    f = open(output_dir + '/icsd-pred.dat', 'w')
    print('# DFT, MLP, id', file=f)
    for st, prop in sorted(icsd_dict.items()):
        print(prop['DFT_energy'], prop['MLP_energy'], st, file=f)
        tag.append(st.replace('icsd-',''))
        error.append(abs(prop['DFT_energy'] - prop['MLP_energy']) * 1000)
    f.close()

    limmax = max(((max(error) // 5) + 1) * 5, 10)

    sns.set_context("paper", 1.0, {"lines.linewidth": 2})
    sns.set_style('whitegrid', {'grid.linestyle': '--'})

    if figsize is not None:
        fig, ax = plt.subplots(figsize=figsize)

    plt.subplots_adjust(bottom=0.25, top=0.9)
    plt.ylim(0,limmax)

    b = sns.barplot(x=tag, y=error, color="royalblue")
    b.axes.set_title("Prediction error for ICSD prototypes ("
                     +system+', '+potname+')',
                     fontsize=fontsize)
    b.set_xlabel('ICSD prototype', fontsize=fontsize)
    b.set_ylabel('Absolute error (meV/atom)', fontsize=fontsize)
    b.tick_params(axis='x', labelsize=4, labelrotation=90)
    b.tick_params(axis='y', labelsize=fontsize)

    plt.tight_layout()
    plt.savefig(output_dir+'/icsd-pred.png',format='png',dpi=dpi)
    plt.savefig(output_dir+'/icsd-pred.eps',format='eps')
    plt.clf()
'''


'''
def plot_energy(data_train, data_test, pot_dict, dpi=300):

    system = pot_dict['system']
    potname = pot_dict['id']
    output_dir = pot_dict['target'] + '/plot-energy-dist/'
    os.makedirs(output_dir, exist_ok=True)

    limmin = min(min(data_train[:,0]), min(data_test[:,0]))
    limmax = max(max(data_train[:,0]), max(data_test[:,0]))
    limmin_int = floor(min(min(data_train[:,0]), min(data_test[:,0])))
    limmax_int = ceil(min(max(max(data_train[:,0]), max(data_test[:,0])), 20))
    x = np.arange(limmin_int,limmax_int+1)
    y = x

    sns.set_context("paper", 1.0, {"lines.linewidth": 1})
    sns.set_style('whitegrid', {'grid.linestyle': '--'})

    fig, ax = plt.subplots(1, 2, figsize=(6,3))
    fig.suptitle("Energy distribution ("+system+', '+potname+')', fontsize=10)

    for i in range(2):
        ax[i].set_aspect('equal')
        ax[i].plot(x,y,color='gray',linestyle='dashed',linewidth=0.25)
        ax[i].scatter(data_train[:,0],
                      data_train[:,1],
                      s=0.25, c='black',
                      alpha=1.0,
                      marker='.',
                      label='training')
        ax[i].scatter(data_test[:,0],
                      data_test[:,1],
                      s=0.25, c='darkorange',
                      alpha=1.0,
                      marker='.',
                      label='test')
        ax[i].set_xlabel('DFT energy (eV/atom)', fontsize=8)
        ax[i].set_ylabel('MLP energy (eV/atom)', fontsize=8)
        ax[i].tick_params(axis='both', labelsize=6)
        ax[i].legend()

    interval = int((limmax_int - limmin_int) / 10)
    ax[0].set_xlim(limmin_int,limmax_int)
    ax[0].set_ylim(limmin_int,limmax_int)
    ax[0].set_xticks(np.arange(limmin_int,limmax_int+1,interval))
    ax[0].set_yticks(np.arange(limmin_int,limmax_int+1,interval))
    ax[1].set_xlim(limmin - 0.05, limmin + 1.05)
    ax[1].set_ylim(limmin - 0.05, limmin + 1.05)

    plt.tight_layout()
    plt.savefig(output_dir + '/distribution.png',format='png',dpi=dpi)
    plt.savefig(output_dir + '/distribution.eps',format='eps')
    plt.clf()
'''
