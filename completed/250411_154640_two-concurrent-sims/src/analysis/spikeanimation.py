import itertools
import os
from brian2.units import *
import numpy as np

import h5py
import matplotlib
from typing import Dict

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle

rc('text', usetex=True)
# plt.rcParams['text.latex.preamble'] = [
#     r'\usepackage{tgheros}',
#     r'\usepackage{sansmath}',
#     r'\sansmath'               
#     r'\usepackage{siunitx}',
#     r'\sisetup{detect-all}',
# ]
pl.rcParams['text.latex.preamble'] = r'\usepackage{tgheros} \usepackage{sansmath} \sansmath \usepackage{siunitx} \sisetup{detect-all}'


def ddcon_figure(bpath):
    nsp = None
    try:
        with open(bpath+'/raw/namespace.p', 'rb') as pfile:
            nsp=pickle.load(pfile)
    except FileNotFoundError:
        print(bpath[-4:], "reports: No namespace data. Skipping.")
        return

    hdf5 = h5py.File('data/hdf5_data.hdf5', 'r')
    print(bpath[-4:], bpath)
    run_results = hdf5['tr1']['results']['runs'][f'run_0000{bpath[-4:]}']
    GExc_x, GExc_y = run_results["GExc_x"]["GExc_x"]*meter, run_results["GExc_y"]["GExc_y"]*meter
    GInh_x, GInh_y = run_results["GInh_x"]["GInh_x"]*meter, run_results["GInh_y"]["GInh_y"]*meter
    begin = nsp['T1']+nsp['T2']+nsp['T3']+nsp['T4']

    with open(f"{bpath}/raw/gexc_spks.p", "rb") as f:
        spikes = pickle.load(f)
        indices = spikes['t'] >= begin
        Espikes = dict(t=spikes['t'][indices], i=spikes['i'][indices])
    with open(f"{bpath}/raw/ginh_spks.p", "rb") as f:
        spikes = pickle.load(f)
        indices = spikes['t'] >= begin
        Ispikes = dict(t=spikes['t'][indices], i=spikes['i'][indices])

    fig: plt.Figure = plt.figure()
    ax_lines, ax_cols = 1, 1
    axs: Dict[str, plt.Axes] = {}
    for x, y in itertools.product(range(ax_lines), range(ax_cols)):
        axs['%d,%d' % (x + 1, y + 1)] = plt.subplot2grid((ax_lines, ax_cols), (x, y))

    plt.gca().set_aspect('equal', adjustable='box')

    dt = 5*ms
    T = 5*second
    ts = np.arange(begin/ms, (begin+T)/ms, dt/ms)
    for t in ts:
        axs['1,1'].plot(GExc_x/um, GExc_y/um, color="blue", ls="None", marker=".", markersize="0.5")
        axs['1,1'].plot(GInh_x/um, GInh_y/um, color="red", ls="None", marker=".", markersize="0.5")

        axs['1,1'].set_xlabel("x [$\\mu \\text{m}$]")
        axs['1,1'].set_ylabel("y [$\\mu \\text{m}$]")

        spikes['i'] = Espikes['i'][np.logical_and(Espikes['t'] >= t*ms, Espikes['t'] < (t*ms + dt))]
        axs['1,1'].plot(GExc_x[spikes['i']] / um, GExc_y[spikes['i']] / um, color="blue", ls="None", marker=".", markersize="6")
        
        spikes['i'] = Ispikes['i'][np.logical_and(Ispikes['t'] >= t*ms, Ispikes['t'] < (t*ms + dt))]
        axs['1,1'].plot(GInh_x[spikes['i']] / um, GInh_y[spikes['i']] / um, color="red", ls="None", marker=".", markersize="6")

        plt.tight_layout()

        directory = f"figures/spikeanimation/{bpath[-4:]}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f"{directory}/{t*ms/ms}.png", dpi=100, bbox_inches='tight')
        plt.cla()


if __name__ == "__main__":
    # return a list of each build (simulations run)
    # e.g. build_dirs = ['builds/0003', 'builds/0007', ...]
    # sorted to ensure expected order
    build_dirs = sorted(['builds/' + pth for pth in next(os.walk("builds/"))[1]])

    for bpath in build_dirs:
        ddcon_figure(bpath)
