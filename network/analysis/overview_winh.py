
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
    ax_lines, ax_cols = 6,6
    axs = {}
    for x,y in itertools.product(range(ax_lines),range(ax_cols)):
        axs['%d,%d'%(x+1,y+1)] = pl.subplot2grid((ax_lines, ax_cols), (x, y))

    fig.set_size_inches((1920/150*5/4)*6/4,1080/150*7/3)


    # --------------------------------------------------------------

    tmin1, tmax1 = 0*second, nsp['T1']
    tmin3, tmax3 = nsp['T1']+nsp['T2'], nsp['T1']+nsp['T2']+nsp['T3']

    
    raster_plot(axs['1,1'], bpath, nsp, tmin=tmin1, tmax=tmax1)
    raster_plot(axs['1,2'], bpath, nsp, tmin=tmin3, tmax=tmax3)

    firing_rates_plot_exc(axs['1,3'], bpath, nsp)
    firing_rates_plot_inh(axs['1,4'], bpath, nsp)

    inst_rates(axs['2,1'], bpath, nsp, tmin=tmin1, tmax=tmax1)
    inst_rates(axs['2,2'], bpath, nsp, tmin=tmin3, tmax=tmax3)
    n_active_synapses_exc(axs['2,3'], bpath, nsp)
    n_active_synapses_inh(axs['2,4'], bpath, nsp)

    insP_trace(axs['2,5'], bpath, nsp)
    
    synapse_weight_traces(axs['3,1'], bpath, nsp, tmin=0*second,
                          tmax=nsp['T1'])

    synapse_weight_traces(axs['3,2'], bpath, nsp,
                          tmin=nsp['T2']+nsp['T1'],
                          tmax=nsp['T2']+nsp['T1']+nsp['T3'])

    synapse_weight_traces(axs['3,3'], bpath, nsp,
                          tmin=nsp['T2']+nsp['T1'],
                          tmax=nsp['T2']+nsp['T1']+nsp['T3'],
                          ylim_top=0.001)

    synEEdyn_data(axs['3,4'], bpath, nsp, when='start')
    synEEdyn_data(axs['3,5'], bpath, nsp, when='end')
    
    
    synapse_weight_traces(axs['4,1'], bpath, nsp,
                          connections = 'EI',
                          tmin=0*second, tmax=nsp['T1'])
    
    synapse_weight_traces(axs['4,2'], bpath, nsp,
                          connections = 'EI',
                          tmin=nsp['T2']+nsp['T1'],
                          tmax=nsp['T2']+nsp['T1']+nsp['T3'])


    synEIdyn_data(axs['4,4'], bpath, nsp, when='start')
    synEIdyn_data(axs['4,5'], bpath, nsp, when='end')

    axs['5,1'].set_title("Conductance Trace of Single Exc Neuron")
    ge_plot(axs['5,1'], bpath, nsp, tmin=tmin1, tmax=tmax1, i=0)
    gi_plot(axs['5,1'], bpath, nsp, tmin=tmin1, tmax=tmax1, i=0)
    gegi_plot(axs['5,1'], bpath, nsp, tmin=tmin1, tmax=tmax1, i=0)

    axs['5,2'].set_title("Conductance Trace of Single Exc Neuron")
    ge_plot(axs['5,2'], bpath, nsp, tmin=tmin3, tmax=tmax3, i=1)
    gi_plot(axs['5,2'], bpath, nsp, tmin=tmin3, tmax=tmax3, i=1)
    gegi_plot(axs['5,2'], bpath, nsp, tmin=tmin3, tmax=tmax3, i=1)

    axs['5,3'].set_title("Conductance Trace of Single Exc Neuron")
    ge_plot(axs['5,3'], bpath, nsp, tmin=tmin3, tmax=tmin3+0.1*second, i=1)

    voltage_traces(axs['6,1'], bpath, nsp, tmin=tmin1, tmax=tmax1)
    voltage_traces(axs['6,2'], bpath, nsp, tmin=tmin3, tmax=tmax3)




    
    synapse_weights_linear(axs['5,4'], bpath, nsp, tstep=-1, bins=50,
                           cutoff=-1)
    synapse_weights_log(   axs['5,5'], bpath, nsp, tstep=-1, bins=50,
                           cutoff=10.**(-4))


    synapse_weights_linear(axs['6,4'], bpath, nsp, tstep=-1, bins=50,
                           cutoff=-1, connections='EI')
    synapse_weights_log(   axs['6,5'], bpath, nsp, tstep=-1, bins=50,
                           cutoff=10.**(-4), connections='EI')

    


    # raster_plot_poisson(axs['6,1'], bpath, nsp, tmin=tmin1, tmax=tmax1)
    # raster_plot_poisson(axs['6,2'], bpath, nsp, tmin=tmin3, tmax=tmax3)

        
    netw_params_display(axs['1,5'], bpath, nsp)
    neuron_params_display(axs['1,6'], bpath, nsp)
    poisson_input_params_display(axs['2,6'], bpath, nsp)
    synapse_params_display(axs['3,6'], bpath, nsp)
    stdp_params_display(axs['4,6'], bpath, nsp)
    sn_params_display(axs['5,6'], bpath, nsp)
    strct_params_display(axs['6,6'], bpath, nsp)
   
    # --------------------------------------------------------------
    

    pl.tight_layout()

    directory = "figures/overview_winh"
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

        try:
            with open(bpath+'/raw/namespace.p', 'rb') as pfile:
                nsp=pickle.load(pfile)

            overview_figure(bpath, nsp)

        except FileNotFoundError:
            print(bpath[-4:], "reports: No namespace data. Skipping.")
