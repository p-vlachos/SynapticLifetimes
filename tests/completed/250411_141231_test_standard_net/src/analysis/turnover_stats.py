
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


def overview_figure(bpath, nsp):

    fig = pl.figure()
    ax_lines, ax_cols = 5,5
    axs = {}
    for x,y in itertools.product(range(ax_lines),range(ax_cols)):
        axs['%d,%d'%(x+1,y+1)] = pl.subplot2grid((ax_lines, ax_cols), (x, y))

    fig.set_size_inches((1920/150*5/4)*6/4,1080/150*7/3)

    for _,ax in axs.items():
        ax.axis('off')

    # --------------------------------------------------------------

    tmin1, tmax1 = 0*second, nsp['T1']
    tmin3, tmax3 = nsp['T1']+nsp['T2'], nsp['T1']+nsp['T2']+nsp['T3']


    syn_turnover_histogram(axs['1,1'], bpath, nsp, cn='EE', ztype='ins')
    syn_turnover_histogram(axs['2,1'], bpath, nsp, cn='EE', ztype='prn')

    syn_turnover_histogram(axs['1,2'], bpath, nsp, cn='EI', ztype='ins')
    syn_turnover_histogram(axs['2,2'], bpath, nsp, cn='EI', ztype='prn')
    

    syn_turnover_ins_prn_correlation(axs['3,1'], bpath, nsp, cn='EE')
    syn_turnover_ins_prn_correlation(axs['3,2'], bpath, nsp, cn='EI')
                           

    syn_turnover_EE_EI_correlation(axs['4,1'], bpath, nsp)
    syn_turnover_EE_EI_correlation(axs['4,2'], bpath, nsp,
                                   xtype='prn', ytype='prn')
    syn_turnover_EE_EI_correlation(axs['4,3'], bpath, nsp,
                                   xtype='prn', ytype='ins')
    syn_turnover_EE_EI_correlation(axs['4,4'], bpath, nsp,
                                   xtype='ins', ytype='prn')
    
        
    netw_params_display(axs['5,3'], bpath, nsp)
    neuron_params_display(axs['1,5'], bpath, nsp)
    poisson_input_params_display(axs['2,5'], bpath, nsp)
    synapse_params_display(axs['3,5'], bpath, nsp)
    stdp_params_display(axs['4,5'], bpath, nsp)
    sn_params_display(axs['5,4'], bpath, nsp)
    strct_params_display(axs['5,5'], bpath, nsp)
   
    # --------------------------------------------------------------
    

    pl.tight_layout()

    directory = "figures/turnover_stats"
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

            overview_figure(bpath, nsp)

        except FileNotFoundError:
            print(bpath[-4:], "reports: No namespace data. Skipping.")
