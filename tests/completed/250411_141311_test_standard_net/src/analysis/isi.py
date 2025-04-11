
import matplotlib

from analysis.frequencies import plot_smoothed_rate

# matplotlib.use('Agg')
import matplotlib.pyplot as pl
from matplotlib import rc, cycler
from scipy.signal import welch

rc('text', usetex=True)
pl.rcParams['text.latex.preamble'] = [
    r'\usepackage{tgheros}',
    r'\usepackage{sansmath}',
    r'\sansmath'               
    r'\usepackage{siunitx}',
    r'\sisetup{detect-all}',
]

import argparse, sys, os, itertools, pickle
import numpy as np
from brian2.units import mV, ms, second


from .axes.neuron import *
from .axes.synapse import *
from .axes.parameter_display import *

import scipy.stats as scistats
from typing import Dict


ISI_BIN_WIDTH = 1*ms
XMAX = 3000


def calc_isi(spks, tmin, tmax, i):
    indices = spks["i"] == i
    ts = spks["t"][indices]
    ts = ts[np.logical_and(ts >= tmin, ts <= tmax)]
    ts = ts * second / ms
    ts = sorted(ts)
    isi = np.diff(ts)
    return isi


def plot_isi_per_neuron(ax, nsp, spks, tmin, tmax, n_neurons = 3):
    neurons, counts = np.unique(spks["i"], return_counts=True)
    sort_nc = sorted(zip(neurons, counts), key=lambda x: x[1], reverse=True)
    neurons, counts = zip(*sort_nc)

    neurons, counts = neurons[:n_neurons], counts[:n_neurons]
    for i in neurons:
        isi = calc_isi(spks, tmin, tmax, i)
        bins = np.arange(0.0, np.max(isi) + ISI_BIN_WIDTH/ms, ISI_BIN_WIDTH/ms)
        ax.hist(isi, histtype="step", label=f"neuron {i:4d}", bins=bins)
    ax.set_xlabel("ISI [ms]")
    ax.set_ylabel("count")
    ax.set_title("Inter Spike Interval of highest frequency exc. neurons")
    ax.set_xticks(np.arange(0, XMAX, 10), minor=True)
    ax.set_xticks(np.arange(0, XMAX+1, 250), minor=False)
    ax.set_xlim(0, XMAX)
    ax.legend()


def plot_isi_all(ax, nsp, spks, tmin, tmax, xmax=XMAX):
    isis = []

    for i in range(nsp['N_e']):
        isi = calc_isi(spks, tmin, tmax, i)
        isis.append(isi)
    isis = np.hstack(isis)

    bins = np.arange(0.0, np.max(isis) + ISI_BIN_WIDTH/ms, ISI_BIN_WIDTH/ms)
    ax.hist(isis, histtype="step", bins=bins, label=f"[{tmin:4.0f}, {tmax:4.0f}] s", density=True, ls="--")
    ax.set_xlabel("ISI [ms]")
    ax.set_ylabel("density")
    ax.set_title("Inter Spike Interval of all exc. neurons")
    xstep = 250 if xmax > 1000 else 100
    ax.set_xticks(np.arange(0, xmax, 10), minor=True)
    ax.set_xticks(np.arange(0, xmax+1, xstep), minor=False)
    ax.set_xlim(0, xmax)


def isi_figure(bpath):
    nsp = None
    try:
        with open(bpath+'/raw/namespace.p', 'rb') as pfile:
            nsp=pickle.load(pfile)
    except FileNotFoundError:
        print(bpath[-4:], "reports: No namespace data. Skipping.")
        return

    size_factor = 2
    rc('font', size=str(8*size_factor))

    fig: pl.Figure = pl.figure()
    ax_lines, ax_cols = 3,3
    axs: Dict[str, pl.Axes] = {}
    for x,y in itertools.product(range(ax_lines),range(ax_cols)):
        axs['%d,%d'%(x+1,y+1)] = pl.subplot2grid((ax_lines, ax_cols), (x, y))

    fig.set_size_inches(ax_cols*size_factor*6*1.6/2,ax_lines*size_factor*2.*1.6)

    tmin1, tmax1 = 0.0 * ms, nsp['T1']
    tmin5 = nsp['T1']+nsp['T2']+nsp['T3']+nsp['T4']
    tmax5 = tmin5+nsp["T5"]

    raster_plot(axs['1,1'], bpath, nsp, tmin=tmin5, tmax=tmax5)


    with open(bpath + '/raw/gexc_spks.p', 'rb') as pfile:
        spks = pickle.load(pfile)
    plot_isi_all(axs['1,2'], nsp, spks, tmin=tmin1, tmax=tmax1)
    plot_isi_all(axs['1,2'], nsp, spks, tmin=tmin5, tmax=tmax5)
    axs['1,2'].set_title("Inter Spike Interval of all exc. neurons")
    axs['1,2'].legend()
    plot_isi_all(axs['2,2'], nsp, spks, tmin=tmin1, tmax=tmax1, xmax=750)
    plot_isi_all(axs['2,2'], nsp, spks, tmin=tmin5, tmax=tmax5, xmax=750)
    axs['2,2'].set_title("Inter Spike Interval of all exc. neurons")
    axs['2,2'].legend()

    with open(bpath + '/raw/ginh_spks.p', 'rb') as pfile:
        spks = pickle.load(pfile)
    plot_isi_all(axs['1,3'], nsp, spks, tmin=tmin1, tmax=tmax1)
    plot_isi_all(axs['1,3'], nsp, spks, tmin=tmin5, tmax=tmax5)
    axs['1,3'].set_title("Inter Spike Interval of all inh. neurons")
    plot_isi_all(axs['2,3'], nsp, spks, tmin=tmin1, tmax=tmax1, xmax=750)
    plot_isi_all(axs['2,3'], nsp, spks, tmin=tmin5, tmax=tmax5, xmax=750)
    axs['2,3'].set_title("Inter Spike Interval of all inh. neurons")

    # smoothed firing rate
    plot_smoothed_rate(axs['2,1'], bpath, nsp)

    # firing rates histogram
    axs['3,2'].set_title("Neuron Firing Rates (exc)")
    firing_rates_plot(axs['3,2'], bpath, "exc", 0.0, nsp['T1'], histargs=dict(histtype="step", color="cornflowerblue"))
    firing_rates_plot(axs['3,2'], bpath, "exc", tmin5, tmax5, histargs=dict(histtype="step", color="darkblue"))
    axs['3,2'].legend()

    axs['3,3'].set_title("Neuron Firing Rates (inh)")
    firing_rates_plot(axs['3,3'], bpath, "inh", 0.0, nsp['T1'], histargs=dict(histtype="step", color="indianred"))
    firing_rates_plot(axs['3,3'], bpath, "inh", tmin5, tmax5, histargs=dict(histtype="step", color="darkred"))
    axs['3,3'].legend()

    pl.tight_layout()

    directory = "figures/isi"
    if not os.path.exists(directory):
        os.makedirs(directory)

    pl.savefig(directory+"/{:s}.png".format(bpath[-4:]), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # return a list of each build (simulations run)
    # e.g. build_dirs = ['builds/0003', 'builds/0007', ...]
    # sorted to ensure expected order
    build_dirs = sorted(['builds/'+pth for pth in next(os.walk("builds/"))[1]])

    for bpath in build_dirs:
        isi_figure(bpath)
        print(f"{bpath[-4:]}...done")
