
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
import scipy.stats

from .axes.network import *
from .axes.neuron import *
from .axes.synapse import *
from .axes.parameter_display import *
from .axes.peaks import *
from .utils import *


def t5_raster_figure(bpath, nsp):

    fig, axes = pl.subplots(5,1)
    fig.set_size_inches(1.5*4,1.5*5.75)

    tmin = nsp['T1']+nsp['T2']+nsp['T3']+nsp['T4']
    tmax, tmax_vis = tmin + nsp['T5'], tmin + 2.5*second

    raster_plot(axes[0], bpath, nsp, tmin, tmax_vis)
    axes[0].set_ylabel('neuron index')
    

    peak_times = inst_rates_with_peaks(axes[1], bpath, nsp, tmin, tmax,
                                       tmax_vis=tmax_vis)
    peak_width = 28*ms
    lookup = {}
    
    print(peak_times)

    for j,pt in enumerate(peak_times[:3]):

        raster_plot(axes[2+j], bpath, nsp,
                    tmin=pt-peak_width/2,
                    tmax=pt+peak_width/2)
        axes[2+j].set_ylabel('neuron index')

        
    pl.tight_layout()

    directory = "figures/t5_peaks"
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    pl.savefig(directory+"/{:s}.png".format(bpath[-4:]),
               dpi=300, bbox_inches='tight')
    pl.clf()

    
    # fig, axes = pl.subplots(6,4)
    # fig.set_size_inches(1.5*4*1.25*1.1,1.5*5.25*1.1)

    # axes = axes.flatten()

    # for j,pt in enumerate(peak_times[:24]):
        
    #     lookup = raster_plot_sorted(axes[j], bpath, nsp,
    #                                 tmin=pt-peak_width/2,
    #                                 tmax=pt+peak_width/2,
    #                                 lookup=lookup)


    # pl.tight_layout()

    # directory = "figures/t5_peaks/sorted"
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
        
    # pl.savefig(directory+"/{:s}.png".format(bpath[-4:]),
    #            dpi=300, bbox_inches='tight')
    # pl.clf()

   
    peak_stats = get_peak_statistics(bpath, nsp, peak_times, peak_width)

    #print(peak_stats)

    peaks_participants = []
    peak_sizes = []

    for statistic in peak_stats:
        peaks_participants.extend(statistic['participating_neurons'])
        peak_sizes.append(len(statistic['participating_neurons']))


        
    fig, axes = pl.subplots(3,1)
    fig.set_size_inches(1.5*4,1.5*5.25)

    axes = axes.flatten()

    axes[0].hist(peaks_participants, bins=np.arange(0,400,1))
    axes[0].set_xlabel('neuron index')
    axes[0].set_ylabel('number of participated bursts')

    topright_axis_off(axes[0])

    counts, bins = np.histogram(peaks_participants, bins=np.arange(0,400,1))

    axes[1].hist(counts, np.arange(0, max(counts)+2,5))

    axes[1].set_xlabel('number of participated bursts')
    axes[1].set_ylabel('number of neurons')

    
    
    axes[2].hist(counts, np.arange(0, max(counts)+2,5), density=True)

    N = len(peak_stats)
    p = np.mean(peak_sizes)/nsp['N_e']

    axes[2].plot(np.arange(0,N+1,1),
                 scipy.stats.binom.pmf(np.arange(0,N+1,1), N, p),
                 color='red', lw=2, label='expected (binomial)\n $p=%.2f$' %p)

    axes[2].set_xlabel('number of participated bursts')
    axes[2].set_ylabel('relative number of neurons')

    directory = "figures/t5_peaks/stats"
    if not os.path.exists(directory):
        os.makedirs(directory)

    topright_axis_off(axes[1])
    topright_axis_off(axes[2])
    
    axes[2].legend(frameon=False)

    

    pl.tight_layout()
        
    pl.savefig(directory+"/{:s}.png".format(bpath[-4:]),
               dpi=300, bbox_inches='tight')
    pl.clf()





    # ------------------------------------------------

    fig, axes = pl.subplots(3,2)
    fig.set_size_inches(1.5*4,1.5*5.25)

    axes = axes.flatten()
    
   #get_neuron_idx_with_participated_bursts_greater_n(n)

    
    burst_nidx = bins[:-1][counts>62]
    print(burst_nidx)
    quiet_nidx = bins[:-1][counts<40]
    
    # new_pp = []
    # for x in peaks_participants:
    #     if x in burst_nidx:
    #         new_pp.append(x)

    # axes[0].hist(new_pp, bins=np.arange(0,400,1), color='red')



    # load weight input distributions from frequent burst participants

    with open(bpath+'/raw/synee_a.p', 'rb') as pfile:
        synee_a = pickle.load(pfile)

    print(np.shape(synee_a['a'][-1]), synee_a['t'][-1])

    A = np.reshape(synee_a['a'][-1], (400,399))

    # high bursts input distribution
    collect_burst_inputs, burst_sums, burst_ns = [], [], []
    for n in burst_nidx:
        collect_burst_inputs.extend(list(A[:,n]))
        burst_sums.append(np.sum(A[:,n]))
        burst_ns.append(len(A[:,n][A[:,n]>10**(-4)]))

    # print(burst_sums)
    # print(collect_burst_inputs)
    collect_burst_inputs = np.array(collect_burst_inputs)
    collect_burst_inputs = collect_burst_inputs[collect_burst_inputs>10**(-4)]
    # print(collect_burst_inputs)
    
        
    collect_quiet_inputs, quiet_sums, quiet_ns = [], [], []
    for n in quiet_nidx:
        collect_quiet_inputs.extend(list(A[:,n]))
        quiet_sums.append(np.sum(A[:,n]))
        quiet_ns.append(len(A[:,n][A[:,n]>10**(-4)]))

    # print(quiet_sums)
    collect_quiet_inputs = np.array(collect_quiet_inputs)
    collect_quiet_inputs = collect_quiet_inputs[collect_quiet_inputs>10**(-4)]


    bins = np.linspace(0., 0.02, 100)

    axes[0].hist(collect_burst_inputs, bins=bins, density=True,
                 label='avg.~input %.2f' %(np.mean(burst_sums)))

    axes[2].hist(collect_quiet_inputs, bins=bins, density=True,
                 label='avg.~input %.2f' %(np.mean(quiet_sums)))


    axes[1].hist(burst_ns, bins=20)
    axes[3].hist(quiet_ns, bins=20)

    # bins = np.linspace(0., 0.02, 100)
    # axes[1].hist(np.log10(collect_burst_inputs), bins=100, density=True,
    #              label='avg.~input %.2f' %(np.mean(burst_sums)))

    # axes[3].hist(np.log10(collect_quiet_inputs), bins=100, density=True,
    #              label='avg.~input %.2f' %(np.mean(quiet_sums)))

    
    for ax in axes:
        topright_axis_off(ax)
        ax.set_xlabel('synaptic input weight')
        ax.set_ylabel('relative frequency')
        ax.legend(frameon=False)

    axes[0].set_title('burst participants')
    axes[2].set_title('burst absentees')

    axes[1].set_xlabel('number of inputs per neuron')
    axes[1].set_ylabel('occurrence')
    axes[1].set_ylim(0,25)

    axes[3].set_xlabel('number of inputs per neuron')
    axes[3].set_ylabel('occurrence')
    axes[3].set_ylim(0,25)

    axes[4].axis('off')
    axes[5].axis('off')
    
    pl.tight_layout()

    directory = "figures/t5_peaks/weights"
    if not os.path.exists(directory):
        os.makedirs(directory)

        
    pl.savefig(directory+"/{:s}.png".format(bpath[-4:]),
               dpi=300, bbox_inches='tight')
    pl.clf()

    
    
    # netw_params_display(axs['1,3'], bpath, nsp)
    # neuron_params_display(axs['1,4'], bpath, nsp)
    # poisson_input_params_display(axs['2,4'], bpath, nsp)
    # synapse_params_display(axs['3,4'], bpath, nsp)
    # stdp_params_display(axs['4,4'], bpath, nsp)
    # sn_params_display(axs['5,4'], bpath, nsp)
    # strct_params_display(axs['6,4'], bpath, nsp)
   


    



    
if __name__ == "__main__":

    
    # return a list of each build (simulations run)
    # e.g. build_dirs = ['builds/0003', 'builds/0007', ...]
    # sorted to ensure expected order
    build_dirs = sorted(['builds/'+pth for pth in next(os.walk("builds/"))[1]])
    
    for bpath in build_dirs:

        if bpath == 'builds/0011':

            try:
                with open(bpath+'/raw/namespace.p', 'rb') as pfile:
                    nsp=pickle.load(pfile)

                t5_raster_figure(bpath, nsp)

            except FileNotFoundError:
                print(bpath[-4:], "reports: No namespace data. Skipping.")
