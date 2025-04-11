
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
from .axes.turnover import *
from .axes.parameter_display import *


def turnover_figure(bpath, nsp, fit):


    fig = pl.figure()
    ax_lines, ax_cols = 6,5
    axs = {}
    for x,y in itertools.product(range(ax_lines),range(ax_cols)):
        axs['%d,%d'%(x+1,y+1)] = pl.subplot2grid((ax_lines, ax_cols), (x, y))

    fig.set_size_inches(1920/150*5/4*5/4,1080/150*7/3)


    # --------------------------------------------------------------
    

    # lifetime_distribution_loglog(axs['1,1'], bpath, nsp, bins=50,
    #                              discard_t=0., with_starters=True)

    lifetime_distribution_loglog_linear_bins(axs['1,1'], bpath, nsp,
                                             bin_w=100*ms, discard_t=0.*second,
                                             initial='without')

    lifetime_distribution_loglog_linear_bins(axs['2,1'], bpath, nsp,
                                             bin_w=1000*ms, discard_t=0.*second,
                                             initial='without')

    lifetime_distribution_loglog_linear_bins(axs['3,1'], bpath, nsp,
                                             bin_w=10000*ms, discard_t=0.*second,
                                             initial='without')

    lifetime_distribution_loglog_linear_bins(axs['4,1'], bpath, nsp,
                                             bin_w=100000*ms, discard_t=0.*second,
                                             initial='without')



    # fit
    lifetime_distribution_loglog_linear_bins(axs['1,2'], bpath, nsp,
                                             bin_w=100*ms, discard_t=0.*second,
                                             initial='without', fit=False,
                                             density=True)

    lifetime_distribution_loglog_linear_bins(axs['2,2'], bpath, nsp,
                                             bin_w=1000*ms, discard_t=0.*second,
                                             initial='without', fit=False,
                                             density=True)

    lifetime_distribution_loglog_linear_bins(axs['3,2'], bpath, nsp,
                                             bin_w=10000*ms, discard_t=0.*second,
                                             initial='without', fit=False,
                                             density=True)

    lifetime_distribution_loglog_linear_bins(axs['4,2'], bpath, nsp,
                                             bin_w=100000*ms, discard_t=0.*second,
                                             initial='without', fit=False,
                                             density=True)

    if fit:
        lifetime_distribution_loglog_add_fit([axs['1,2'], axs['2,2'],
                                              axs['3,2'], axs['4,2']],
                                             bpath, nsp, discard_t=0.*second,
                                             initial='without')
    

    # without starters
    lifetime_distribution_loglog_linear_bins(axs['1,3'], bpath, nsp,
                                             bin_w=100*ms, discard_t=0.*second,
                                             initial='only')

    lifetime_distribution_loglog_linear_bins(axs['2,3'], bpath, nsp,
                                             bin_w=1000*ms, discard_t=0.*second,
                                             initial='only')

    lifetime_distribution_loglog_linear_bins(axs['3,3'], bpath, nsp,
                                             bin_w=10000*ms, discard_t=0.*second,
                                             initial='only')

    lifetime_distribution_loglog_linear_bins(axs['4,3'], bpath, nsp,
                                             bin_w=100000*ms, discard_t=0.*second,
                                             initial='only')


    # without starters, fit
    lifetime_distribution_loglog_linear_bins(axs['1,4'], bpath, nsp,
                                             bin_w=100*ms, discard_t=0.*second,
                                             initial='only', fit=False,
                                             density=True)

    lifetime_distribution_loglog_linear_bins(axs['2,4'], bpath, nsp,
                                             bin_w=1000*ms, discard_t=0.*second,
                                             initial='only', fit=False,
                                             density=True)

    lifetime_distribution_loglog_linear_bins(axs['3,4'], bpath, nsp,
                                             bin_w=10000*ms, discard_t=0.*second,
                                             initial='only', fit=False,
                                             density=True)

    lifetime_distribution_loglog_linear_bins(axs['4,4'], bpath, nsp,
                                             bin_w=100000*ms, discard_t=0.*second,
                                             initial='only', fit=False,
                                             density=True)

    if fit:
        lifetime_distribution_loglog_add_fit([axs['1,4'], axs['2,4'],
                                              axs['3,4'], axs['4,4']],
                                             bpath, nsp, discard_t=0.*second,
                                             initial='only')


    netw_params_display(axs['1,5'], bpath, nsp)
    neuron_params_display(axs['2,5'], bpath, nsp)
    synapse_params_display(axs['3,5'], bpath, nsp)
    stdp_params_display(axs['4,5'], bpath, nsp)
    sn_params_display(axs['5,5'], bpath, nsp)
    strct_params_display(axs['6,5'], bpath, nsp)


    # --------------------------------------------------------------
    

    pl.tight_layout()

    directory = 'figures/turnover'
    if fit:
        directory += '_fit'
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    pl.savefig(directory+"/{:s}.png".format(bpath[-4:]),
               dpi=150., bbox_inches='tight')

    



    
if __name__ == "__main__":

    
    # return a list of each build (simulations run)
    # e.g. build_dirs = ['builds/0003', 'builds/0007', ...]
    # sorted to ensure expected order
    build_dirs = sorted(['builds/'+pth for pth in next(os.walk("builds/"))[1]])



    for bpath in build_dirs:

        try:
            with open(bpath+'/raw/namespace.p', 'rb') as pfile:
                nsp=pickle.load(pfile)

            turnover_figure(bpath, nsp, fit=False)
            # print('Found ', bpath)

        except FileNotFoundError:
            print(bpath[-4:], "reports: No namespace data. Skipping.")


