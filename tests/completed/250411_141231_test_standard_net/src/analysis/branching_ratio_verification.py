
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from matplotlib import rc

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


from .axes.network import *
from .axes.neuron import *
from .axes.synapse import *
from .axes.parameter_display import *

import mrestimator as mre
import scipy.stats as scistats
from typing import Dict


def branching_ratio_verification_figure(bpath):
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
    ax_lines, ax_cols = 2,2
    axs: Dict[str, pl.Axes] = {}
    for x,y in itertools.product(range(ax_lines),range(ax_cols)):
        axs['%d,%d'%(x+1,y+1)] = pl.subplot2grid((ax_lines, ax_cols), (x, y))

    fig.set_size_inches(ax_cols*size_factor*6*1.6/2,ax_lines*size_factor*2.*1.6)

    tmin5 =  nsp['T1']+nsp['T2']+nsp['T3']+nsp['T4']

    raster_plot(axs['1,1'], bpath, nsp, tmin=tmin5, tmax=tmin5+nsp["T5"])

    bin_w = 5*ms
    chunk_n = 10

    results = branching_ratio_chunks(None, bpath, nsp, bin_w, chunk_n=chunk_n)

    rk, ft, _ = branching_ratio(None, bpath, nsp, bin_w)
    complex_fit = mre.fit(rk, fitfunc=mre.f_complex)


    tbl = axs['2,1'].table(loc="center",
                           cellText=(("$\Delta{}t$", str(bin_w)),
                                     ("Estimates over all sections", ""),
                                     ("mre", f"{ft.mre:0.6f}"),
                                     ("mre (complex)", f"{complex_fit.mre:0.6f}"),
                                     ("ATotalMax", nsp['ATotalMax']),
                                     ("iATotalMax", nsp['iATotalMax']),
                                     ("$mu_e$", nsp["mu_e"]),
                                     ("$mu_i$", nsp["mu_i"]),
                                     )
                           )
    tbl.scale(1, 0.6*size_factor)
    axs['2,1'].axis('tight')
    axs['2,1'].axis('off')

    markersize = str(int(5 * size_factor))
    xs = range(len(results))
    fts = [ft.mre for (rk, ft, cft) in results]
    cfts = [cft.mre for (rk, ft, cft) in results]
    axs['1,2'].plot(xs, fts, color="green", markersize=markersize, label="exponential fit", ls="None", marker="x")
    axs['1,2'].plot(xs, cfts, color="pink", markersize=markersize, label="complex fit", ls="None", marker="x")
    axs['1,2'].legend()
    axs['1,2'].set_title(f"Branching Factor per Section")
    axs['1,2'].set_xlabel("$section number$")
    axs['1,2'].set_ylabel("$\hat{m}$")
    axs['1,2'].set_ylim(0, 1.2)

    # smoothed firing rate
    axr = axs['2,2'].twinx()
    with open(bpath+'/raw/gexc_rate.p', 'rb') as pfile:
        rate = pickle.load(pfile)
        smth = pickle.load(pfile)
    begin_t = (nsp['T1']+nsp['T2']+nsp['T3']+nsp['T4'])
    indices = rate['t'] > begin_t
    axr.plot(rate['t'][indices]/second - begin_t/second, smth[indices],
             color="green", label="exc. rate", lw=0.75)
    axr.set_ylabel("population rate [Hz]")
    yticks = axr.get_yticks()
    yticks = yticks[yticks >= 0]
    axr.set_yticks(yticks)
    axr.set_yticks(np.arange(np.min(yticks), np.max(yticks), 1), minor=True)

    pl.tight_layout()

    directory = "figures/branching_ratio_verification"
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    pl.savefig(directory+"/{:s}.png".format(bpath[-4:]),
               dpi=300, bbox_inches='tight')



if __name__ == "__main__":
    # return a list of each build (simulations run)
    # e.g. build_dirs = ['builds/0003', 'builds/0007', ...]
    # sorted to ensure expected order
    build_dirs = sorted(['builds/'+pth for pth in next(os.walk("builds/"))[1]])

    for bpath in build_dirs:
        branching_ratio_verification_figure(bpath)
