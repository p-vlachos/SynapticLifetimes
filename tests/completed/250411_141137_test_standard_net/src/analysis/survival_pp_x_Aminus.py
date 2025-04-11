
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
from .axes.survival import *
from .axes.parameter_display import *


   

    
if __name__ == "__main__":
    
    # return a list of each build (simulations run)
    # e.g. build_dirs = ['builds/0003', 'builds/0007', ...]
    # sorted to ensure expected order
    build_dirs = sorted(['builds/'+pth for pth in next(os.walk("builds/"))[1]])

    fit = False
    
    fig, ax = pl.subplots()

    t_splits = []

    for bpath in build_dirs:

        try:

            with open(bpath+'/raw/namespace.p', 'rb') as pfile:
                nsp=pickle.load(pfile)

            label = '$D =' +'%.2f' %(-1*nsp['Aminus']/(0.5*nsp['Aplus'])) + '$'

            with open(bpath+'/raw/survival.p', 'rb') as pfile:
                srv_pp=pickle.load(pfile)

            t_splits.append(srv_pp['t_split'])

            ax.plot(srv_pp['s_times'],
                    srv_pp['s_counts']/srv_pp['s_counts'][0],
                    label=label)


        except FileNotFoundError:
            print(bpath[-4:], "reports: No namespace data. Skipping.")



    assert t_splits.count(t_splits[0]) == len(t_splits)
    Tmax = t_splits[0]/second


    ax.set_xlabel('time since synapse growth [s]')

    ax.set_ylabel('survival probability')
            
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
            

    

    directory = 'figures/survival/'
    if fit:
        directory += '_fit'
    if not os.path.exists(directory):
        os.makedirs(directory)

    pl.legend()

    fname = "srvpp_x_Aminus_linear" #_4x_T%d_NPInp%d" %(Tmax,NPInp)
    fig.savefig(directory+'/'+fname+'.png', dpi=150, bbox_inches='tight')

    ax.set_yscale('log')
    ax.set_xscale('log')

    fname = "srv_x_Aminus_log" #_4x_T%d_NPInp%d" %(Tmax,NPInp)
    fig.savefig(directory+'/'+fname+'.png', dpi=150, bbox_inches='tight')

    xs = np.logspace(0, np.log10(Tmax), num=10000)
    ys = (xs/(25.)+1)**(-1.384)
    ax.plot(xs,ys, 'r', linestyle='dashed')

    fname = "srv_x_Aminus_log_fit" #_4x_T%d_NPInp%d" %(Tmax,NPInp)
    fig.savefig(directory+'/'+fname+'.png', dpi=150, bbox_inches='tight')

    # ax.set_yscale('linear')
    # ax.set_xscale('linear')

    # fname = "srv_x_Aminus_linearfit_4x_T%d_NPInp%d" %(Tmax,NPInp)
    # fig.savefig(directory+'/'+fname+'.png', dpi=150, bbox_inches='tight')




        # fname = "srv_x_Aminus_tsplit%ds_tcut%ds_insP%.7f" %(int(t_split/second),                                                                       int(t_cut/second),
    #                                                     insP)

