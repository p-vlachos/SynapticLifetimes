
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

from .axes.synapse import *
from .axes.parameter_display import *


def synw_figure(bpath, nsp, connections='EE'):
    
    if connections=='EE':
        with open(bpath+'/raw/synee_a.p', 'rb') as pfile:
            syn_a = pickle.load(pfile)
    elif connections=='EI':
        with open(bpath+'/raw/synei_a.p', 'rb') as pfile:
            syn_a = pickle.load(pfile)

    nrec_pts = len(syn_a['t'])

    if nrec_pts < 7:
        tsteps_plot = range(nrec_pts)
    else:
        tsteps_plot = np.linspace(start=0,stop=nrec_pts-1,
                                  num=6, dtype=int, endpoint=True)
        
    fig = pl.figure()
    ax_lines, ax_cols = 6,4
    axs = {}
    for x,y in itertools.product(range(ax_lines),range(ax_cols)):
        axs['%d,%d'%(x+1,y+1)] = pl.subplot2grid((ax_lines, ax_cols), (x, y))

    fig.set_size_inches(1920/150*5/4,1080/150*7/3)


    for j,tstep in enumerate(tsteps_plot):

        synapse_weights_linear(axs['%d,1' %(j+1)], bpath, nsp, tstep=tstep,
                               bins=50, cutoff=0.,
                               connections=connections)

        synapse_weights_linear(axs['%d,2' %(j+1)], bpath, nsp, tstep=tstep,
                               bins=50, cutoff=10.**(-20),
                               connections=connections)

        synapse_weights_linear(axs['%d,3' %(j+1)], bpath, nsp, tstep=tstep,
                               bins=50, cutoff=10**(-10), 
                               connections=connections)
        

    netw_params_display(axs['1,4'], bpath, nsp)
    neuron_params_display(axs['2,4'], bpath, nsp)
    synapse_params_display(axs['3,4'], bpath, nsp)
    stdp_params_display(axs['4,4'], bpath, nsp)
    sn_params_display(axs['5,4'], bpath, nsp)
    strct_params_display(axs['6,4'], bpath, nsp)
 

    pl.tight_layout()

    directory = "figures/synw_" + connections + "_linear"
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    pl.savefig(directory+"/{:s}.png".format(bpath[-4:]),
               dpi=100, bbox_inches='tight')

    

    
if __name__ == "__main__":

    
    # return a list of each build (simulations run)
    # e.g. build_dirs = ['builds/0003', 'builds/0007', ...]
    # sorted to ensure expected order
    build_dirs = sorted(['builds/'+pth for pth in next(os.walk("builds/"))[1]])



    for bpath in build_dirs:

        try:
            with open(bpath+'/raw/namespace.p', 'rb') as pfile:
                nsp=pickle.load(pfile)

            synw_figure(bpath, nsp, connections='EE')

            if nsp['istdp_active']:
                synw_figure(bpath, nsp, connections='EI')

        except FileNotFoundError:
            print(bpath[-4:], "reports: No namespace data. Skipping.")


