
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


def single_wtraces(bpath, nsp):

    with open(bpath+'/raw/synee_stat.p', 'rb') as pfile:
        synee_stat = pickle.load(pfile)

    tmin1, tmax1 = 0*second, nsp['T1']
    tmin3, tmax3 = nsp['T1']+nsp['T2'], nsp['T1']+nsp['T2']+nsp['T3']


    # for i in range(np.shape(synee_stat['a'])[1]):
    indx = np.logical_and(synee_stat['t'] > 12*second,
                          synee_stat['t'] < tmax3)

    directory = "figures/single_wtraces/" +bpath[-4:]
    if not os.path.exists(directory):
        os.makedirs(directory)

    

    for i in range(nsp['n_synee_traces_rec']):

        fig, ax = pl.subplots()

        ax.plot(synee_stat['t'][indx],synee_stat['a'][:,i][indx], color='grey')

        if "syn_active" in synee_stat:
            # we did record information about which synapses where active
            # so let's display it as well
            inac_t = synee_stat['t'][indx][synee_stat['syn_active'][:,i][indx]==0]
            ax.plot(inac_t, np.zeros_like(inac_t), '.', color='red')

        # ax.text(0.47, 0.95,
        # 'ascale='+'%.2E' % Decimal(tr.ascale),
        # horizontalalignment='left',
        # verticalalignment='top',
        # bbox={'boxstyle': 'square, pad=0.3', 'facecolor':'white',
        #       'alpha':1, 'edgecolor':'none'},
        # transform = ax.transAxes)

        # if plot_thresholds:
        #     ax.axhline(tr.a_insert, linestyle='dashed', color='grey')
        #     ax.axhline(tr.prn_thrshld, linestyle='dashed', color='grey')
        #     ax.axhline(tr.amax, color='red')

        # if ylim_top>0:
        #     ax.set_ylim(0,ylim_top)

        ax.set_title('Synaptic Weight Traces')
        ax.set_xlabel('time [s]')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # pl.tight_layout()


        pl.savefig(directory+"/%.4d.png" %(i),
                   dpi=100, bbox_inches='tight')

        pl.close()



    
if __name__ == "__main__":

    
    # return a list of each build (simulations run)
    # e.g. build_dirs = ['builds/0003', 'builds/0007', ...]
    # sorted to ensure expected order
    build_dirs = sorted(['builds/'+pth for pth in next(os.walk("builds/"))[1]])
    
    for bpath in build_dirs:

        try:
            with open(bpath+'/raw/namespace.p', 'rb') as pfile:
                nsp=pickle.load(pfile)

            single_wtraces(bpath, nsp)

        except FileNotFoundError:
            print(bpath[-4:], "reports: No namespace data. Skipping.")
