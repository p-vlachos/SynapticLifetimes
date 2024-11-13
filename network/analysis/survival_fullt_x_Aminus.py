
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

# from .axes.synapse import *
# from .axes.survival import *
# from .axes.parameter_display import *


def add_survival(ax, bin_w, bpath, nsp):

    label = '$D=' +'%.2f' %(nsp['Aminus']/(-0.00075/10)) + '$'

    with open(bpath+'/raw/survival_full_t.p', 'rb') as pfile:
        df=pickle.load(pfile)

    # out = {'t_split': t_split, 't_cut': t_cut,
    #        'full_t': full_t, 'excluded_ids': ex_ids}

    bins = np.arange(bin_w/second, df['t_split']/second+2*bin_w/second,
                     bin_w/second)

    counts, edges = np.histogram(df['full_t'], bins=bins,
                                 density=True)
    centers = (edges[:-1] + edges[1:])/2.

    xx = np.cumsum(counts[::-1])[::-1]/np.sum(counts)

    ax.plot(centers, xx, '.', markersize=2., label=label)


    
if __name__ == "__main__":
    
    # return a list of each build (simulations run)
    # e.g. build_dirs = ['builds/0003', 'builds/0007', ...]
    # sorted to ensure expected order
    build_dirs = sorted(['builds/'+pth for pth in next(os.walk("builds/"))[1]])

    Tmax, bin_w = 12501*second,  1*second
    fit = False

   
    fig, ax = pl.subplots()


    for bpath in build_dirs:

        try:

            with open(bpath+'/raw/namespace.p', 'rb') as pfile:
                nsp=pickle.load(pfile)
                
            add_survival(ax, bin_w, bpath, nsp)


        except FileNotFoundError:
            print(bpath[-4:], "reports: No namespace data. Skipping.")

            

    directory = 'figures/fullt_x_Aminus'
    if fit:
        directory += '_fit'
    if not os.path.exists(directory):
        os.makedirs(directory)

    pl.legend()

    fname = "srv_x_Aminus_linear_4x"
    fig.savefig(directory+'/'+fname+'.png', dpi=150, bbox_inches='tight')

    ax.set_yscale('log')
    ax.set_xscale('log')

    fname = "srv_x_Aminus_log_4x"
    fig.savefig(directory+'/'+fname+'.png', dpi=150, bbox_inches='tight')

    xs = np.logspace(0, np.log10(Tmax/second), num=10000)
    ys = (xs/(25.)+1)**(-1.384)
    ax.plot(xs,ys, 'r', linestyle='dashed')

    fname = "srv_x_Aminus_logfit_4x"
    fig.savefig(directory+'/'+fname+'.png', dpi=150, bbox_inches='tight')

    ax.set_yscale('linear')
    ax.set_xscale('linear')

    fname = "srv_x_Aminus_linearfit_4x"
    fig.savefig(directory+'/'+fname+'.png', dpi=150, bbox_inches='tight')




