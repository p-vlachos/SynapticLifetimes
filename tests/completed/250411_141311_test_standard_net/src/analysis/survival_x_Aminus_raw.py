
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
from scipy.optimize import curve_fit

from brian2.units import mV, ms, second

from .axes.synapse import *
from .axes.survival import *
from .axes.parameter_display import *


def add_survival(ax, bpath, nsp):

    label = str(nsp['Aminus']) 

    with open(bpath+'/raw/survival.p', 'rb') as pfile:
        df = pickle.load(pfile)

    ax.plot(df['s_times'][df['s_times']<150*second]/second, df['s_counts'][df['s_times']<150*second]/df['s_counts'][0], '.',
            markersize=2., label=label)

    # https://stackoverflow.com/questions/41109122

    print(df['s_counts'][-100:])

    
if __name__ == "__main__":
    
    # return a list of each build (simulations run)
    # e.g. build_dirs = ['builds/0003', 'builds/0007', ...]
    # sorted to ensure expected order
    build_dirs = sorted(['builds/'+pth for pth in next(os.walk("builds/"))[1]])

    bin_w = 1*second
    fit = False

    insert_P = [0.0000125,0.000015,0.0000175]

    for insP in insert_P:
    
        fig, ax = pl.subplots()

        for bpath in build_dirs:

            try:

                print('Found ', bpath)

                with open(bpath+'/raw/namespace.p', 'rb') as pfile:
                    nsp=pickle.load(pfile)

                if nsp['insert_P']==insP:

                    add_survival(ax, bpath, nsp)

                    
                # with open(bpath+'/raw/namespace.p', 'rb') as pfile:
                #     nsp=pickle.load(pfile)
            
                # add_survival(ax, bpath, nsp)


            except FileNotFoundError:
                print(bpath[-4:], "reports: No namespace data. Skipping.")



        directory = 'figures/survival_x_Aminus_raw'
        if fit:
            directory += '_fit'
        if not os.path.exists(directory):
            os.makedirs(directory)

        pl.legend()

        # ax.set_yscale('log')
        # ax.set_xscale('log')

        fname = "srv_x_Aminus_insP%.7f" %(insP)# tsplit%ds_tcut%ds_insP%.7f" %(int(t_split/second),                                                                       int(t_cut/second),
                               #                              insP)

        fig.savefig(directory+'/'+fname+'.png', dpi=150, bbox_inches='tight')

