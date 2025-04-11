
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as pl
from matplotlib import rc, cycler
from scipy.signal import welch

rc('text', usetex=True)
# pl.rcParams['text.latex.preamble'] = [
#     r'\usepackage{tgheros}',   
#     r'\usepackage{sansmath}',  
#     r'\sansmath'               
#     r'\usepackage{siunitx}',   
#     r'\sisetup{detect-all}',   
# ]  
pl.rcParams['text.latex.preamble'] = r'\usepackage{tgheros} \usepackage{sansmath} \sansmath \usepackage{siunitx} \sisetup{detect-all}'

import argparse, sys, os, itertools, pickle
import numpy as np
from brian2.units import mV, ms, second


from .axes.neuron import *
from .axes.synapse import *
from .axes.parameter_display import *

import scipy.stats as scistats
from typing import Dict


def fourier(bpath, nsp, population, ax1, ax2, colormap=pl.cm.Blues):
    # binning
    dt = 0.1 * ms
    try:
        with open(bpath + f'/raw/{population}.p', 'rb') as pfile:
            spk_counts = pickle.load(pfile)
        spk_count = spk_counts['spk_count']

        fourier_data(ax1, ax2, colormap, dt, nsp, spk_count)
    except FileNotFoundError as e:
        print("skipping fourier analysis due to following exception")
        print(e)


def fourier_data(ax1, ax2, colormap, dt, nsp, spk_count, nperseg = 10240):
    start_i = int((nsp['T1'] + nsp['T2'] + nsp['T3'] + nsp['T4']) / ms / (dt / ms))
    t5_length = len(spk_count) - start_i
    # calculate time sections & colors
    end_first = (0 + t5_length)
    start_icenter = (start_i - end_first) * np.array([1 / 4, 2 / 4, 3 / 4])
    start_icenter = np.ceil(start_icenter * dt / second / 10).astype(int) / dt / second * 10  # ceil to 10 seconds
    start_is = np.hstack([[0], start_icenter, [start_i]]).astype(int)
    colors = list(colormap(np.linspace(0, 1, len(start_is) + 2)))[2:]
    # fourier over time
    ax1.set_prop_cycle(cycler('color', colors))
    ax2.set_prop_cycle(cycler('color', colors))
    for start_i in start_is:
        end_i = start_i + t5_length
        xs, ys = welch(spk_count[start_i:end_i], fs=(dt ** -1) / Hz, nperseg=nperseg)
        indices = np.logical_and(xs > 0, xs <= 1000)  # we only want up to 1 kHz
        xs, ys = xs[indices], ys[indices]
        label = f"[{start_i * dt / second}, {end_i * dt / second}] s"
        ax1.semilogy(xs, ys, label=label)
        ax2.loglog(xs, ys, label=label)
        # ax2.set_xticks(np.arange(np.min(xs), np.max(xs), 10), minor=True)
        ax1.set_ylabel("log density")
        ax1.set_xlabel("frequency [Hz]")
        ax2.set_ylabel("log density")
        ax2.set_xlabel("log frequency [Hz]")
    ax1.legend()
    ax2.legend()


def fourier_loglog_single(ax2, dt, nsp, spk_count, color, label, nperseg = 10240):
    start_i = int((nsp['T1'] + nsp['T2'] + nsp['T3'] + nsp['T4']) / ms / (dt / ms))
    end_i = int((nsp['T1'] + nsp['T2'] + nsp['T3'] + nsp['T4'] + nsp['T5']) / ms / (dt / ms))
    xs, ys = welch(spk_count[start_i:end_i], fs=(dt ** -1) / Hz, nperseg=nperseg)
    indices = np.logical_and(xs > 0, xs <= 1000)  # we only want up to 1 kHz
    xs, ys = xs[indices], ys[indices]
    ax2.loglog(xs, ys, label=label, color=color)
    ax2.set_ylabel("log density")
    ax2.set_xlabel("log frequency [Hz]")
    ax2.legend()


def plot_smoothed_rate(axr, bpath, nsp):
    with open(bpath+'/raw/gexc_rate.p', 'rb') as pfile:
        rate = pickle.load(pfile)
        smth = pickle.load(pfile)
    begin_t = (nsp['T1']+nsp['T2']+nsp['T3']+nsp['T4'])
    end_t = begin_t + nsp['T5']
    indices = rate['t'] > begin_t
    xs = rate['t'][indices]/second
    axr.plot(xs, smth[indices], color="green", label="exc. rate", lw=0.75)
    axr.set_ylabel("population rate [Hz]")
    axr.set_xlabel("time [s]")
    yticks = axr.get_yticks()
    axr.set_yticks(np.arange(np.min(yticks), np.max(yticks), 1), minor=True)
    axr.set_ylim(0)
    axr.set_xlim(begin_t/second, end_t/second)

def frequencies_figure(bpath):
    nsp = None
    try:
        with open(bpath+'/raw/namespace.p', 'rb') as pfile:
            nsp=pickle.load(pfile)
    except FileNotFoundError:
        print(bpath[-4:], "reports: No namespace data. Skipping.")
        return

    if nsp['population_binned_rec'] == 0:
        return

    size_factor = 2
    rc('font', size=str(8*size_factor))

    fig: pl.Figure = pl.figure()
    ax_lines, ax_cols = 3,3
    axs: Dict[str, pl.Axes] = {}
    for x,y in itertools.product(range(ax_lines),range(ax_cols)):
        axs['%d,%d'%(x+1,y+1)] = pl.subplot2grid((ax_lines, ax_cols), (x, y))

    fig.set_size_inches(ax_cols*size_factor*6*1.6/2,ax_lines*size_factor*2.*1.6)

    tbegin5 =  nsp['T1']+nsp['T2']+nsp['T3']+nsp['T4']
    tend5 = tbegin5 + nsp["T5"]

    raster_plot(axs['1,1'], bpath, nsp, tmin=tbegin5, tmax=tend5)

    axs['1,2'].set_title("Power Spectrum Density (exc)")
    fourier(bpath, nsp, 'gexc_binned', axs['1,2'], axs['2,2'], colormap=pl.cm.Blues)

    axs['1,3'].set_title("Power Spectrum Density (inh)")
    fourier(bpath, nsp, 'ginh_binned', axs['1,3'], axs['2,3'], colormap=pl.cm.Reds)

    # smoothed firing rate
    plot_smoothed_rate(axs['2,1'], bpath, nsp)

    # firing rates histogram
    axs['3,2'].set_title("Neuron Firing Rates (exc)")
    firing_rates_plot(axs['3,2'], bpath, "exc", 0.0, nsp['T1'], histargs=dict(histtype="step", color="cornflowerblue"))
    firing_rates_plot(axs['3,2'], bpath, "exc", tbegin5, tend5, histargs=dict(histtype="step", color="darkblue"))
    axs['3,2'].legend()

    axs['3,3'].set_title("Neuron Firing Rates (inh)")
    firing_rates_plot(axs['3,3'], bpath, "inh", 0.0, nsp['T1'], histargs=dict(histtype="step", color="indianred"))
    firing_rates_plot(axs['3,3'], bpath, "inh", tbegin5, tend5, histargs=dict(histtype="step", color="darkred"))
    axs['3,3'].legend()

    pl.tight_layout()

    directory = "figures/frequencies"
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    pl.savefig(directory+"/{:s}.png".format(bpath[-4:]), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # return a list of each build (simulations run)
    # e.g. build_dirs = ['builds/0003', 'builds/0007', ...]
    # sorted to ensure expected order
    build_dirs = sorted(['builds/'+pth for pth in next(os.walk("builds/"))[1]])

    for bpath in build_dirs:
        frequencies_figure(bpath)
        print(f"{bpath[-4:]}...done")
