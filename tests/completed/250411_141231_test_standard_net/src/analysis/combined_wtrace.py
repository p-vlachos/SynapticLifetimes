
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


def combined_wtrace(bpath, nsp):

    with open(bpath+'/raw/synee_stat.p', 'rb') as pfile:
        synee_stat = pickle.load(pfile)

    tmin1, tmax1 = 0*second, nsp['T1']
    tmin3, tmax3 = nsp['T1']+nsp['T2'], nsp['T1']+nsp['T2']+nsp['T3']


    # for i in range(np.shape(synee_stat['a'])[1]):
    indx = np.logical_and(synee_stat['t'] > tmin3,
                          synee_stat['t'] < tmax3)

    directory = "figures/combined_wtrace/" 
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig, ax = pl.subplots()
    fig.set_size_inches(8,3)
    

    for i in range(nsp['n_synee_traces_rec']):

        ax.plot(synee_stat['t'][indx],synee_stat['a'][:,i][indx], color='grey')
    
    ax.set_title('Synaptic Weight Traces')
    ax.set_xlabel('time [s]')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # pl.tight_layout()


    pl.savefig(directory+"/%s.png" %(bpath[-4:]),
               dpi=150, bbox_inches='tight')

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

            combined_wtrace(bpath, nsp)

        except FileNotFoundError:
            print(bpath[-4:], "reports: No namespace data. Skipping.")
