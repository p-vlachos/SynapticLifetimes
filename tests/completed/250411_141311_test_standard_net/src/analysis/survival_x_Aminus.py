
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


def add_survival(ax, bin_w, bpath, nsp, t_split, t_cut):

    label = '$D=' +'%.2f' %(nsp['Aminus']/(-0.00075/10)) + '$'

    survival_probabilities_linear(ax, bpath, nsp,
                                  bin_w=bin_w,
                                  t_split=t_split,
                                  t_cut=t_cut,
                                  density=True,
                                  label=label)
    

    
if __name__ == "__main__":
    
    # return a list of each build (simulations run)
    # e.g. build_dirs = ['builds/0003', 'builds/0007', ...]
    # sorted to ensure expected order
    build_dirs = sorted(['builds/'+pth for pth in next(os.walk("builds/"))[1]])

    bin_w = 1*second
    fit = False
    Tmax = 2000
    NPInp = 1000

    # insert_P = [0.0000125,0.000015,0.0000175]

    # for insP in insert_P:
    
    fig, ax = pl.subplots()


    for bpath in build_dirs:

        try:

            # print('Found ', bpath)
            # if bpath=='builds/0000':

            #     with open(bpath+'/raw/namespace.p', 'rb') as pfile:
            #         nsp=pickle.load(pfile)

            #     add_turnover(ax, bin_w, bpath, nsp, fit, starters, t_cut)
            #     print('hi')


            with open(bpath+'/raw/namespace.p', 'rb') as pfile:
                nsp=pickle.load(pfile)

            t_split = nsp['T2']/2
            t_cut = nsp['T1']+nsp['T3']

            # if nsp['T2'] > t_cut and nsp['insert_P']==insP:
            if nsp['T2'] == Tmax*second and nsp['NPInp'] == NPInp:

                add_survival(ax, bin_w, bpath, nsp,
                             t_split=t_split, t_cut=t_cut)



        except FileNotFoundError:
            print(bpath[-4:], "reports: No namespace data. Skipping.")



            

    directory = 'figures/survival_x_Aminus'
    if fit:
        directory += '_fit'
    if not os.path.exists(directory):
        os.makedirs(directory)

    pl.legend()

    fname = "srv_x_Aminus_linear_4x_T%d_NPInp%d" %(Tmax,NPInp)
    fig.savefig(directory+'/'+fname+'.png', dpi=150, bbox_inches='tight')

    ax.set_yscale('log')
    ax.set_xscale('log')

    fname = "srv_x_Aminus_log_4x_T%d_NPInp%d" %(Tmax,NPInp)
    fig.savefig(directory+'/'+fname+'.png', dpi=150, bbox_inches='tight')

    xs = np.logspace(0, np.log10(Tmax), num=10000)
    ys = (xs/(25.)+1)**(-1.384)
    ax.plot(xs,ys, 'r', linestyle='dashed')

    fname = "srv_x_Aminus_logfit_4x_T%d_NPInp%d" %(Tmax,NPInp)
    fig.savefig(directory+'/'+fname+'.png', dpi=150, bbox_inches='tight')

    ax.set_yscale('linear')
    ax.set_xscale('linear')

    fname = "srv_x_Aminus_linearfit_4x_T%d_NPInp%d" %(Tmax,NPInp)
    fig.savefig(directory+'/'+fname+'.png', dpi=150, bbox_inches='tight')




        # fname = "srv_x_Aminus_tsplit%ds_tcut%ds_insP%.7f" %(int(t_split/second),                                                                       int(t_cut/second),
    #                                                     insP)

