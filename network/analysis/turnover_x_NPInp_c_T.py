
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


def add_turnover(ax, bin_w, bpath, nsp, fit, starters, t_cut):

    label = str(nsp['Aminus']) 

    lifetime_distribution_loglog_linear_bins(ax, bpath, nsp,
                                             bin_w=bin_w,
                                             discard_t=t_cut,
                                             initial=starters,
                                             density=True,
                                             label=label)
    

    
if __name__ == "__main__":
    
    # return a list of each build (simulations run)
    # e.g. build_dirs = ['builds/0003', 'builds/0007', ...]
    # sorted to ensure expected order
    build_dirs = sorted(['builds/'+pth for pth in next(os.walk("builds/"))[1]])

    starters, t_cut, fit = 'without', 1000*second, False
    bin_w = 1*second

    Ts = []

    # get all Ts
    for bpath in build_dirs:

        try:
            with open(bpath+'/raw/namespace.p', 'rb') as pfile:
                nsp=pickle.load(pfile)
            
            Ts.append(nsp['T2'])

        except FileNotFoundError:
            print(bpath[-4:], "reports: No namespace data. Skipping.")

    Ts = list(set([T/second for T in Ts]))*second

    print(Ts)
    

    for T in Ts:
    
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

                if nsp['T2'] > t_cut and nsp['T2']==T:
                    
                    add_turnover(ax, bin_w, bpath, nsp, fit, starters, t_cut)



            except FileNotFoundError:
                print(bpath[-4:], "reports: No namespace data. Skipping.")



        directory = 'figures/turnover_x_NPInp_c_T'
        if fit:
            directory += '_fit'
        if not os.path.exists(directory):
            os.makedirs(directory)

        pl.legend()

        fname = "trn_x_p_%ds_starter-%s_tcut%ds_T%ds" %(int(bin_w/second),
                                                            starters,
                                                            int(t_cut/second),
                                                            int(T/second))

        fig.savefig(directory+'/'+fname+'.png', dpi=150, bbox_inches='tight')

