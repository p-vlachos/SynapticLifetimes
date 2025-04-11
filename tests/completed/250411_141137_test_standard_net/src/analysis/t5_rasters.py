
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


def t5_raster_figure(bpath, nsp):

    fig, axes = pl.subplots(4,1)
    fig.set_size_inches(1.5*4,1.5*5.75)

    tmin = nsp['T1']+nsp['T2']+nsp['T3']+nsp['T4']+36*ms
    # tmax = tmin + 5*second
    tmax = tmin + 35*ms
    
    raster_plot(axes[0], bpath, nsp, tmin, tmax)
    axes[0].set_ylabel('neuron index')
    
    lookup = raster_plot_sorted(axes[1], bpath, nsp, tmin, tmax)
    axes[1].set_ylabel('neuron index')


    tmin = nsp['T1']+nsp['T2']+nsp['T3']+nsp['T4']+315*ms+36*ms
    # tmax = tmin + 5*second
    tmax = tmin + 35*ms
    raster_plot(axes[2], bpath, nsp, tmin, tmax)
    axes[2].set_ylabel('neuron index')
    
    lookup = raster_plot_sorted(axes[3], bpath, nsp, tmin, tmax,
                                lookup=lookup)
    axes[3].set_ylabel('neuron index')



    
    # netw_params_display(axs['1,3'], bpath, nsp)
    # neuron_params_display(axs['1,4'], bpath, nsp)
    # poisson_input_params_display(axs['2,4'], bpath, nsp)
    # synapse_params_display(axs['3,4'], bpath, nsp)
    # stdp_params_display(axs['4,4'], bpath, nsp)
    # sn_params_display(axs['5,4'], bpath, nsp)
    # strct_params_display(axs['6,4'], bpath, nsp)
   

    pl.tight_layout()

    directory = "figures/t5_raster"
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

        if bpath == 'builds/0007':

            try:
                with open(bpath+'/raw/namespace.p', 'rb') as pfile:
                    nsp=pickle.load(pfile)

                t5_raster_figure(bpath, nsp)

            except FileNotFoundError:
                print(bpath[-4:], "reports: No namespace data. Skipping.")
