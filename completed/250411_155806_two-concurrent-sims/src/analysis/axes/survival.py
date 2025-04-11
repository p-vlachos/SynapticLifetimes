
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl

import numpy as np
from brian2.units import mV, ms, second

import pickle, powerlaw

from ..methods.process_survival import extract_survival


def survival_probabilities_linear(ax, bpath, nsp, bin_w,
                                 t_split,
                                 t_cut, 
                                 label = '',
                                 fit=False,
                                 density=False):


    with open(bpath+'/raw/turnover.p', 'rb') as pfile:
        turnover = pickle.load(pfile)


    if not len(turnover) == 0:
    
        s_times, s_counts = extract_survival(turnover, bin_w,
                                             nsp['N_e'],
                                             t_split,
                                             t_cut = t_cut)

        if density:
            s_counts = s_counts/s_counts[0]

        ax.plot(s_times/second, s_counts, '.', markersize=2., label=label)

        
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_title('synapse lifetimes (' + \
    #              r'$\text{bin width} = \text{\SI{%d}{s}}$)' % (int(bin_w/second)))

    ax.set_xlabel('time since synapse growth [s]')

    if density:
        ax.set_ylabel('survival probability')
    else:
        ax.set_ylabel('number of suriving synapses')
            
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
            
