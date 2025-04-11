
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

import sys, os, itertools, pickle, decimal
import numpy as np
import scipy.stats
from brian2.units import second


def yasumatsu_graph(bpath, nsp):

    fig, axes = pl.subplots(3,3)
    fig.set_size_inches(7.35*(3/2.)*1.3,8*1.3)

    try:
        
        with open(bpath+'/raw/namespace.p', 'rb') as pfile:
            nsp=pickle.load(pfile)

        with open(bpath+'/raw/synee_a.p', 'rb') as pfile:
            synee_a = pickle.load(pfile)


        for k,stepsize in enumerate([1,2,5]): 

            tstart = 0
            tstop = 50
            label = 't=\SI{' + '%d' %((synee_a['t'][tstart+stepsize] - synee_a['t'][tstart])/second) + '}{s}'


            syn_a = synee_a['a'][tstart:tstop,:]
            syn_a = syn_a[np.arange(0,tstop-tstart,stepsize),:]

            dsyn_a = np.diff(syn_a, axis=0)

            syn_a, dsyn_a = syn_a[:-1].flatten(), dsyn_a.flatten()

            idx = np.logical_or(syn_a!=0, dsyn_a!=0)
            syn_a, dsyn_a = syn_a[idx], dsyn_a[idx]


            print(len(syn_a), len(dsyn_a))

            bin_w = 0.005
            bins=np.arange(0,nsp['ATotalMax']+bin_w, bin_w)
            stat, edges, _  = scipy.stats.binned_statistic(x=syn_a,
                                                           values= dsyn_a,
                                                           bins=bins,
                                                           statistic='mean')
            centers = (edges[:-1] + edges[1:]) / 2
            axes[k,0].plot(centers, stat, label=label, lw=1.25)

            bin_w = 0.001
            bins=np.arange(0,nsp['ATotalMax']+bin_w, bin_w)
            stat, edges, _  = scipy.stats.binned_statistic(x=syn_a,
                                                           values= dsyn_a,
                                                           bins=bins,
                                                           statistic='mean')
            centers = (edges[:-1] + edges[1:]) / 2
            axes[k,1].plot(centers, stat, label=label, lw=1.25)

            bin_w = 0.0001
            bins=np.arange(0,nsp['ATotalMax']+bin_w, bin_w)
            stat, edges, _  = scipy.stats.binned_statistic(x=syn_a,
                                                           values= dsyn_a,
                                                           bins=bins,
                                                           statistic='mean')
            centers = (edges[:-1] + edges[1:]) / 2
            axes[k,2].plot(centers, stat, label=label, lw=1.25)

            stat=stat[np.logical_not(np.isnan(stat))]
            stat=stat[np.logical_not(np.isinf(stat))]
            axes[k,0].set_ylim(-1*np.max(np.abs(stat))/3.5,np.max(np.abs(stat))/3.5)
            axes[k,1].set_ylim(-1*np.max(np.abs(stat))/2.5,np.max(np.abs(stat))/2.5)
            axes[k,2].set_ylim(-1*np.max(np.abs(stat))/1.5,np.max(np.abs(stat))/1.5)
            

    except FileNotFoundError:
        print(bpath[-4:], "reports: No namespace data. Skipping.")



    for ax in list(axes.flatten()):

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # ax.spines['bottom'].set_position('center')
        # ax.xaxis.set_ticks_position('bottom')

        ax.legend(frameon=False, prop={'size': 12})

        ax.set_xlabel('synaptic weight')
        ax.set_ylabel('$\Delta$ synaptic weight')

        ax.axhline(color='grey', zorder=-1.)

        ax.set_xlim(0,0.05)


    # axes[0].set_xlabel('synaptic weight')
    # axes[0].set_ylabel('$\Delta$ synaptic weight')

    # axes[1].set_xlabel('$\log_{10}(\mathrm{synaptic\,weight})$')
    # axes[1].set_ylabel('probability density')

    # # axes[0].set_xlim(left=0)

    # # axes[2,2].set_title('E-E connection density')
    # # axes[2,2].set_xlabel('network simulation time [s]')
    # # axes[2,2].set_ylabel('E-E connection density')

    # axes[1,0].set_ylim(-1.2,6)
    # axes[1,1].set_ylim(-1.2,6)
    # axes[1,2].set_ylim(-1.2,2)

    # fig.suptitle('$c=' + strct_c + '$')
    fig.tight_layout(rect=[0., 0., 1, 0.95])


    directory = "figures/yasumatsu_graph"
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    fig.savefig(directory+"/{:s}.png".format(bpath[-4:]),
               dpi=100, bbox_inches='tight')



if __name__ == "__main__":

    
    # return a list of each build (simulations run)
    # e.g. build_dirs = ['builds/0003', 'builds/0007', ...]
    # sorted to ensure expected order
    build_dirs = sorted(['builds/'+pth for pth in next(os.walk("builds/"))[1]])

    for bpath in build_dirs:

        try:
            with open(bpath+'/raw/namespace.p', 'rb') as pfile:
                nsp=pickle.load(pfile)

            yasumatsu_graph(bpath, nsp)

        except FileNotFoundError:
            print(bpath[-4:], "reports: No namespace data. Skipping.")
        
