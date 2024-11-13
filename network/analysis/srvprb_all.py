
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

import sys, os, itertools, pickle
import numpy as np
from brian2.units import second


def srvprb_all_figure(bpath):
    
    fig, axes = pl.subplots(3,3)
    fig.set_size_inches(7.35*(3/2.),8)
        
    bin_w = 1*second

    try:

        print('trying to open ',  bpath+'/raw/namespace.p')
        with open(bpath+'/raw/namespace.p', 'rb') as pfile:
            nsp=pickle.load(pfile)

        # E<-E data 

        with open(bpath+'/raw/survival_full_t.p', 'rb') as pfile:
            df=pickle.load(pfile)

        bin_w = 1*second
        bins = np.arange(bin_w/second,
                         df['t_split']/second+2*bin_w/second,
                         bin_w/second)

        counts, edges = np.histogram(df['full_t'], bins=bins,
                                     density=False)
        centers = (edges[:-1] + edges[1:])/2.
        xx = np.cumsum(counts[::-1])[::-1]/np.sum(counts)

        label = '$P_{\mathrm{growth}} = %.4f$' %(nsp['insert_P'])

        axes[0,0].plot(centers, xx, label=label)
        axes[0,1].plot(centers, xx, label=label)


        # ---------------
        bins = np.arange(1,101+0.5,0.5)

        xyz = np.array(df['full_t'])
        counts, edges = np.histogram(xyz[xyz<101], bins=bins,
                                     density=True)
        centers = (edges[:-1] + edges[1:])/2.
        axes[1,0].plot(centers, counts, label=label)


        # ---------------

        bins = np.logspace(np.log10(1),
                           np.log10(df['t_split']/second+0.1),
                           num=100)

        counts, edges = np.histogram(df['full_t'], bins=bins,
                                     density=True)
        centers = (edges[:-1] + edges[1:])/2.
        axes[1,1].plot(centers, counts, label=label)

        # ---------------

        with open(bpath+'/raw/lts_wthsrv_lognbin100.p', 'rb') as pfile:
            df=pickle.load(pfile)

        counts, centers = df['counts'], df['centers']

        axes[1,2].plot(centers, counts, label=label)


        with open(bpath+'/raw/synee_a.p', 'rb') as pfile:
            synee_a = pickle.load(pfile)

        active_at_t = np.sum(synee_a['syn_active'], axis=1)

        all_at_t = np.shape(synee_a['syn_active'])[1]
        assert all_at_t == nsp['N_e']*(nsp['N_e']-1)

        axes[2,2].plot(synee_a['t'], active_at_t/all_at_t, lw=2,
                     label=label)

        # ----------------
        tstep, bin_w, cutoff = -1, 1*second, 0

        weight_at_t = synee_a['a'][tstep,:]
        states_at_t = synee_a['syn_active'][tstep,:]

        active_weight_at_t = weight_at_t[states_at_t==1]
        active_weight_at_t_cutoff = active_weight_at_t[active_weight_at_t>cutoff]

        fraction_of_cutoff = len(active_weight_at_t_cutoff)/len(active_weight_at_t)    

        weights = active_weight_at_t_cutoff
        bins = np.arange(0,0.2+0.0001,0.0001)
        counts, edges = np.histogram(weights, bins=bins,
                                     density=True)
        centers = (edges[:-1] + edges[1:])/2.
        axes[2,0].plot(centers, counts, label=label)


        tstep, bin_w, cutoff = -1, 1*second, 0
        active_weight_at_t_cutoff = active_weight_at_t[active_weight_at_t>cutoff]       
        bins = np.arange(-8, 0+0.01, 0.01)                    
        log_weights = np.log10(active_weight_at_t_cutoff)
        counts, edges = np.histogram(log_weights, bins=bins,
                                     density=True)
        centers = (edges[:-1] + edges[1:])/2.
        axes[2,1].plot(centers, counts, label=label)

        # E<-I data
        with open(bpath+'/raw/survival_EI_full_t.p', 'rb') as pfile:
            df=pickle.load(pfile)

        bin_w = 1*second
        bins = np.arange(bin_w/second,
                         df['t_split']/second+2*bin_w/second,
                         bin_w/second)

        counts, edges = np.histogram(df['full_t'], bins=bins,
                                     density=False)
        centers = (edges[:-1] + edges[1:])/2.
        xx = np.cumsum(counts[::-1])[::-1]/np.sum(counts)

        label = '$P_{\mathrm{growth}} = %.4f$' %(nsp['insert_P'])

        axes[0,0].plot(centers, xx, label=label)
        axes[0,1].plot(centers, xx, label=label)


        # ---------------
        bins = np.arange(1,101+0.5,0.5)

        xyz = np.array(df['full_t'])
        counts, edges = np.histogram(xyz[xyz<101], bins=bins,
                                     density=True)
        centers = (edges[:-1] + edges[1:])/2.
        axes[1,0].plot(centers, counts, label=label)


        # ---------------

        bins = np.logspace(np.log10(1),
                           np.log10(df['t_split']/second+0.1),
                           num=100)

        counts, edges = np.histogram(df['full_t'], bins=bins,
                                     density=True)
        centers = (edges[:-1] + edges[1:])/2.
        axes[1,1].plot(centers, counts, label=label)

        # ---------------

        with open(bpath+'/raw/lts_EI_wthsrv_lognbin100.p', 'rb') as pfile:
            df=pickle.load(pfile)

        counts, centers = df['counts'], df['centers']

        axes[1,2].plot(centers, counts, label=label)


        with open(bpath+'/raw/synei_a.p', 'rb') as pfile:
            synei_a = pickle.load(pfile)

        active_at_t = np.sum(synei_a['syn_active'], axis=1)

        all_at_t = np.shape(synei_a['syn_active'])[1]
        assert all_at_t == nsp['N_e']*nsp['N_i']

        axes[2,2].plot(synei_a['t'], active_at_t/all_at_t, lw=2,
                     label=label)

        # ----------------
        tstep, bin_w, cutoff = -1, 1*second, 0

        weight_at_t = synei_a['a'][tstep,:]
        states_at_t = synei_a['syn_active'][tstep,:]

        active_weight_at_t = weight_at_t[states_at_t==1]
        active_weight_at_t_cutoff = active_weight_at_t[active_weight_at_t>cutoff]

        fraction_of_cutoff = len(active_weight_at_t_cutoff)/len(active_weight_at_t)    

        weights = active_weight_at_t_cutoff
        bins = np.arange(0,0.2+0.0001,0.0001)
        counts, edges = np.histogram(weights, bins=bins,
                                     density=True)
        centers = (edges[:-1] + edges[1:])/2.
        axes[2,0].plot(centers, counts, label=label)


        tstep, bin_w, cutoff = -1, 1*second, 0
        active_weight_at_t_cutoff = active_weight_at_t[active_weight_at_t>cutoff]       
        bins = np.arange(-8, 0+0.01, 0.01)                    
        log_weights = np.log10(active_weight_at_t_cutoff)
        counts, edges = np.histogram(log_weights, bins=bins,
                                     density=True)
        centers = (edges[:-1] + edges[1:])/2.
        axes[2,1].plot(centers, counts, label=label)


    except FileNotFoundError:
        print(bpath[-4:], "reports: No namespace data. Skipping.")



    for ax in list(axes.flatten()):

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.legend(frameon=False, prop={'size': 8})


    # fig.suptitle('$P=' + p_given + '$')

    axes[0,0].set_title('survival probability')
    axes[0,1].set_title('survival probability')
    axes[0,2].set_title('survival probability')

    axes[1,0].set_title('lifetime distribution')
    axes[1,1].set_title('lifetime distribution')
    axes[1,2].set_title('lifetime distribution')

    axes[2,0].set_title('weight distribution')
    axes[2,1].set_title('weight distribution')
        
    axes[0,0].set_xlabel('time since synapse growth [s]')
    axes[0,0].set_ylabel('probability of survival')

    axes[0,1].set_xlabel('time since synapse growth [s]')
    axes[0,1].set_ylabel('probability of survival')

    axes[1,0].set_xlabel('synapse lifetime [s]')
    axes[1,0].set_ylabel('probability density')

    axes[1,1].set_xlabel('synapse lifetime [s]')
    axes[1,1].set_ylabel('probability density')

    axes[1,2].set_xlabel('synapse lifetime [s]')
    axes[1,2].set_ylabel('probability density')

        
    axes[0,1].set_yscale('log')
    axes[0,1].set_xscale('log')
    axes[0,1].set_ylim(top=1.05)

    axes[0,2].set_yscale('log')
    axes[0,2].set_xscale('log')
    axes[0,2].set_ylim(top=1.05)


    axes[2,0].set_xlim(left=0.,right=0.01)

    axes[2,2].set_ylim(0,0.15)

    axes[1,0].set_yscale('log')
    axes[1,0].set_xscale('log')
    
    axes[1,1].set_yscale('log')
    axes[1,1].set_xscale('log')

    axes[1,2].set_yscale('log')
    axes[1,2].set_xscale('log')

    axes[2,0].set_xlabel('synaptic weight')
    axes[2,0].set_ylabel('probability density')

    axes[2,1].set_xlabel('$\log_{10}(\mathrm{synaptic\,weight})$')
    axes[2,1].set_ylabel('probability density')

    axes[2,1].set_xlim(-4.5,-1)

    axes[2,2].set_title('E-E connection density')
    axes[2,2].set_xlabel('network simulation time [s]')
    axes[2,2].set_ylabel('E-E connection density')
    


    # axes[1,1].set_ylim(top=1.05)

    fig.tight_layout(rect=[0., 0., 1, 0.95])

    directory = "figures/srvprb_all"
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    fig.savefig(directory+"/{:s}.png".format(bpath[-4:]),
               dpi=100, bbox_inches='tight')




if __name__ == "__main__":

    build_dirs = sorted(['builds/'+pth for pth in next(os.walk("builds/"))[1]])
    
    for bpath in build_dirs:

        try:
            with open(bpath+'/raw/namespace.p', 'rb') as pfile:
                nsp=pickle.load(pfile)

            srvprb_all_figure(bpath)

        except FileNotFoundError:
            print(bpath[-4:], "reports: No namespace data. Skipping.")
