
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


# def add_survival(ax, 

#     label = '$D=' +'%.2f' %(nsp['Aminus']/(-0.00075/10)) + '$'


    
if __name__ == "__main__":
    
    # return a list of each build (simulations run)
    # e.g. build_dirs = ['builds/0003', 'builds/0007', ...]
    # sorted to ensure expected order
    build_dirs = sorted(['builds/'+pth for pth in next(os.walk("builds/"))[1]])

    bins = 75
    tstep=-1
    cutoff = 0.00001
    
    fit = False
    Tmax = 50000
    NPInp = 100

    # insert_P = [0.0000125,0.000015,0.0000175]

    # for insP in insert_P:
    



    for bpath in build_dirs:

        fig, ax = pl.subplots()

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

                label = '$D=' +'%.2f' %(nsp['Aminus']/(-0.00075/10)) + '$'

                synapse_weights_log(ax, bpath, nsp,
                                       tstep, bins, cutoff,
                                       label=label, fit=False)
    



        except FileNotFoundError:
            print(bpath[-4:], "reports: No namespace data. Skipping.")



            

        directory = 'figures/synw_single'
        if fit:
            directory += '_fit'
        if not os.path.exists(directory):
            os.makedirs(directory)

        pl.legend()

        fname = "synw_%.8f"  %(nsp['Aminus'])
        fig.savefig(directory+'/'+fname+'.png', dpi=150, bbox_inches='tight')

    # ax.set_yscale('log')
    # ax.set_xscale('log')

    # fname = "srv_x_Aminus_log_4x_T%d_NPInp%d" %(Tmax,NPInp)
    # fig.savefig(directory+'/'+fname+'.png', dpi=150, bbox_inches='tight')

    # xs = np.logspace(0, np.log10(Tmax), num=10000)
    # ys = (xs/(4*24*3600.)+1)**(-1.384)
    # ax.plot(xs,ys, 'r', linestyle='dashed')

    # fname = "srv_x_Aminus_logfit_4x_T%d_NPInp%d" %(Tmax,NPInp)
    # fig.savefig(directory+'/'+fname+'.png', dpi=150, bbox_inches='tight')

    # ax.set_yscale('linear')
    # ax.set_xscale('linear')

    # fname = "srv_x_Aminus_linearfit_4x_T%d_NPInp%d" %(Tmax,NPInp)
    # fig.savefig(directory+'/'+fname+'.png', dpi=150, bbox_inches='tight')




        # fname = "srv_x_Aminus_tsplit%ds_tcut%ds_insP%.7f" %(int(t_split/second),                                                                       int(t_cut/second),
    #                                                     insP)

