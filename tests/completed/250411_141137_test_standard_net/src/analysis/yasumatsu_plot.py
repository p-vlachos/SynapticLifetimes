
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
from brian2.units import second


def yasumatsu_plot(bpath, nsp):

    fig, axes = pl.subplots(2,1)
    fig.set_size_inches(7.35,8)

    try:
        
        with open(bpath+'/raw/namespace.p', 'rb') as pfile:
            nsp=pickle.load(pfile)

        with open(bpath+'/raw/synee_a.p', 'rb') as pfile:
            synee_a = pickle.load(pfile)


        for tstart,tstop in [(5,6)]:

            label = 't=\SI{' + '%d' %((synee_a['t'][tstop] - synee_a['t'][tstart])/second) + '}{s}' 
            
            bin_w, cutoff = 1*second, 0

            weight_dt = synee_a['a'][tstop,:] - synee_a['a'][tstart,:]

            weight_tstart, weight_dt = synee_a['a'][tstart,:].flatten(), weight_dt.flatten()

            idx = np.logical_or(weight_dt!=0,weight_tstart!=0)
            weight_tstart_slct = weight_tstart[idx]
            weight_dt_slct = weight_dt[idx]

            axes[0].scatter(weight_tstart_slct, weight_dt_slct, marker='o',
                            lw=0, s=(72./fig.dpi)**2, label=label)


            # idx = np.random.choice(range(len(weight_dt)),
            #                        replace=False, size=4000)
            # weight_dt_subsamp = weight_dt[idx]
            # weight_tstart_subsamp = weight_tstart[idx]
            # axes[1].scatter(weight_tstart_subsamp, weight_dt_subsamp, marker='o',
            #                 lw=0, s=0.5, label=label)

            idx = weight_tstart > 0
            weight_tstart_slct = weight_tstart[idx]
            weight_dt_slct = weight_dt[idx]

            axes[1].scatter(weight_tstart_slct,
                            weight_dt_slct/weight_tstart_slct,
                            marker='o', lw=0, s=(72./fig.dpi)**2,
                            label=label)

            



    except FileNotFoundError:
        print(bpath[-4:], "reports: No namespace data. Skipping.")



    for ax in list(axes.flatten()):

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.legend(frameon=False, prop={'size': 8})



    axes[0].set_xlabel('synaptic weight')
    axes[0].set_ylabel('$\Delta$ synaptic weight')

    axes[1].set_xlabel('$\log_{10}(\mathrm{synaptic\,weight})$')
    axes[1].set_ylabel('probability density')

    # axes[0].set_xlim(left=0)

    # axes[2,2].set_title('E-E connection density')
    # axes[2,2].set_xlabel('network simulation time [s]')
    # axes[2,2].set_ylabel('E-E connection density')

    axes[1].set_ylim(-1.2,6)

    # fig.suptitle('$c=' + strct_c + '$')
    fig.tight_layout(rect=[0., 0., 1, 0.95])


    directory = "figures/yasumatsu"
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

            yasumatsu_plot(bpath, nsp)

        except FileNotFoundError:
            print(bpath[-4:], "reports: No namespace data. Skipping.")
        
