
import pickle

import numpy as np
import scipy.stats
from scipy.stats import norm, lognorm
from brian2.units import mV, ms, second
from decimal import Decimal

from ..utils import WeightSnapshots


def synapse_weights_linear(ax, bpath, nsp, tstep, bins, cutoff,
                           label='', fit=False, connections='EE'):


    try:
        if connections=='EE':
            syn_a = WeightSnapshots(bpath+'/raw/synee_a.p')
            color, alpha = '#1f77b4', 1
        elif connections=='EI':
            syn_a = WeightSnapshots(bpath+'/raw/synei_a.p')
            color, alpha = '#d62728', 0.7

        _, _, active_weight_at_t = syn_a.active(tstep)

        active_weight_at_t_cutoff = active_weight_at_t[active_weight_at_t>cutoff]

        fraction_of_cutoff = len(active_weight_at_t_cutoff)/len(active_weight_at_t)

        print(fraction_of_cutoff)

        if fit:
            ax.hist(active_weight_at_t_cutoff, bins=bins, label=label,
                    density=True, color=color, alpha=alpha)
        else:
            ax.hist(active_weight_at_t_cutoff, bins=bins, label=label,
                    color=color, alpha=alpha)

        if fit:
            fs, floc, fscale = lognorm.fit(active_weight_at_t_cutoff, floc=0)
            f_rv = lognorm(fs, loc=0, scale=fscale)
            xs = np.logspace(start=np.log(cutoff),
                             stop=np.log(10**1.0),
                             base=10., num=5000)
            ax.plot(xs, f_rv.pdf(xs), 'r')

        if connections=='EE':
            ax.set_title('E'+r'$\leftarrow$'+'E weights at t=\\SI{'+ \
                         str(syn_a.t[tstep]/second)+'}{s}')
        elif connections=='EI':
            ax.set_title('E'+r'$\leftarrow$'+'I weights at t=\\SI{'+ \
                         str(syn_a.t[tstep]/second)+'}{s}')

        ax.set_xlabel('synaptic weight')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')


    except FileNotFoundError:
        print("Couldn't find " + connections + " data")   


def synapse_weights_log(ax, bpath, nsp, tstep, bins, cutoff, label='',
                        fit=True, connections='EE'):

    try:
        if connections=='EE':
            syn_a = WeightSnapshots(bpath+'/raw/synee_a.p')
            color, alpha = '#1f77b4', 1
        elif connections=='EI':
            syn_a = WeightSnapshots(bpath+'/raw/synei_a.p')
            color, alpha = '#d62728', 0.7

        _, _, active_weight_at_t = syn_a.active(tstep)

        active_weight_at_t_cutoff = active_weight_at_t[active_weight_at_t>cutoff]

        fraction_of_cutoff = len(active_weight_at_t_cutoff)/len(active_weight_at_t)    

        log_weights = np.log10(active_weight_at_t_cutoff)

        if len(log_weights)>0:

            ax.hist(log_weights, bins=bins, density=True, label=label,
                    color=color, alpha=alpha)

            if fit:
                floc, fscale = norm.fit(log_weights)
                f_rv = norm(loc=floc, scale=fscale)
                xs = np.linspace(start=np.min(log_weights),
                                 stop=np.max(log_weights),
                                 num = 1000)
                ax.plot(xs, f_rv.pdf(xs), lw=2, color='red',
                        linestyle='-')

        if connections=='EE':
            ax.set_title('E'+r'$\leftarrow$'+'E weights at t=\\SI{'+ \
                         str(syn_a.t[tstep]/second)+'}{s}')
        elif connections=='EI':
            ax.set_title('E'+r'$\leftarrow$'+'I weights at t=\\SI{'+ \
                         str(syn_a.t[tstep]/second)+'}{s}')

        ax.set_xlabel('log10 of synaptic weight')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlim(np.log10(cutoff), 1.0)

    except FileNotFoundError:
        print("Couldn't find " + connections + " data")   



def synapse_weight_traces(ax, bpath, nsp, tmin, tmax,
                          connections='EE', ylim_top=-1,
                          plot_thresholds=False):


    try:

        if connections=='EE':
            with open(bpath+'/raw/synee_stat.p', 'rb') as pfile:
                syn_stat = pickle.load(pfile)
        elif connections=='EI':
            with open(bpath+'/raw/synei_stat.p', 'rb') as pfile:
                syn_stat = pickle.load(pfile)

        
        for i in range(np.shape(syn_stat['a'])[1]):
            indx = np.logical_and(syn_stat['t'] > tmin,
                                  syn_stat['t'] < tmax)
            ax.plot(syn_stat['t'][indx],syn_stat['a'][:,i][indx],
                    color='grey')

        if ylim_top > 0:
            ax.set_ylim(0, ylim_top)
            ax.set_ylabel(f"weight (cut off at {ylim_top})")

        ax.set_title(f'Synaptic Weight Traces {connections}')
        ax.set_xlabel('time [s]')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
    

    except (KeyError, FileNotFoundError) as e:
        ax.axis('off')


    
def insP_trace(ax, bpath, nsp):


    try: 
        with open(bpath+'/raw/insP_stat.p', 'rb') as pfile:
            insP_stat = pickle.load(pfile)

        ax.plot(insP_stat['t'], insP_stat['insert_P'], color='blue', lw=2)
        
    except FileNotFoundError:
        print(bpath[-4:], "reports: No n_active data! (EE)")
        ax.set_title("No data found")


    try: 
        with open(bpath+'/raw/insP_EI_stat.p', 'rb') as pfile:
            insP_stat = pickle.load(pfile)

        ax.plot(insP_stat['t'], insP_stat['insert_P'], color='red', lw=2)
        
    except FileNotFoundError:
        print(bpath[-4:], "reports: No n_active data (EI)!")
        ax.set_title("No data found")
 
    
    ax.set_xlabel('time [s]')
    ax.set_ylabel('$p_{\\mathrm{insert}}$')
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


    
def synEEdyn_data(ax, bpath, nsp, when, bins=50):

    try:
        syndyn = WeightSnapshots(bpath+'/raw/syneedynrec.p')

        nrec = nsp['syndynrec_npts']

        assert len(syndyn) == nrec*2

        # weight changes between consecutive snapshots
        first = 0 if when=='start' else nrec
        dsyn_a = np.concatenate([syndyn.weight_changes(k, k+1)[1]
                                 for k in range(first, first+nrec-1)])
          
        ax.hist(dsyn_a, bins=bins)
        
        label = '$\\Delta$ synaptic weight in ' + \
                r'$dt=\text{\SI{' +\
                '%d' %(int(nsp['syndynrec_dt']/second)) +\
                '}{s}}$'
      
        ax.set_xlabel(label)

        if when=='start':
            title = r'$T=\text{\SI{'+ \
                    '%d' %(int(syndyn.t[0]/second)) +\
                    r'}{s}}$ until $T=\text{\SI{'+\
                    '%d' %(int(syndyn.t[nrec-1]/second)) +\
                    '}{s}}$'

        elif when=='end':

            title = r'$T=\text{\SI{'+ \
                    '%d' %(int(syndyn.t[nrec]/second)) +\
                    r'}{s}}$ until $T=\text{\SI{'+\
                    '%d' %(int(syndyn.t[-1]/second)) +\
                    '}{s}}$'

        ax.set_title(title)


        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        
    except FileNotFoundError:
        ax.set_title("No data found")
        ax.axis('off')
    


def synEIdyn_data(ax, bpath, nsp, when, bins=50):

    color, alpha = '#d62728', 0.7

    try:
        syndyn = WeightSnapshots(bpath+'/raw/syneidynrec.p')

        nrec = nsp['syndynrec_npts']

        assert len(syndyn) == nrec*2

        # weight changes between consecutive snapshots
        first = 0 if when=='start' else nrec
        dsyn_a = np.concatenate([syndyn.weight_changes(k, k+1)[1]
                                 for k in range(first, first+nrec-1)])
          
        ax.hist(dsyn_a, bins=bins,
                color=color, alpha=alpha)
        
        label = '$\\Delta$ synaptic weight in ' + \
                r'$dt=\text{\SI{' +\
                '%d' %(int(nsp['syndynrec_dt']/second)) +\
                '}{s}}$'
      
        ax.set_xlabel(label)

        if when=='start':
            title = r'$T=\text{\SI{'+ \
                    '%d' %(int(syndyn.t[0]/second)) +\
                    r'}{s}}$ until $T=\text{\SI{'+\
                    '%d' %(int(syndyn.t[nrec-1]/second)) +\
                    '}{s}}$'

        elif when=='end':

            title = r'$T=\text{\SI{'+ \
                    '%d' %(int(syndyn.t[nrec]/second)) +\
                    r'}{s}}$ until $T=\text{\SI{'+\
                    '%d' %(int(syndyn.t[-1]/second)) +\
                    '}{s}}$'

        ax.set_title(title)


        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        
    except FileNotFoundError:
        ax.set_title("No data found")
        ax.axis('off')
    
        


def n_active_synapses_exc(ax, bpath, nsp):

    try: 
        synee_a = WeightSnapshots(bpath+'/raw/synee_a.p')

        # trace 1: data from synEE_a (~11 data points)
        active_at_t = synee_a.n_active()

        print(active_at_t)
        print(synee_a.t)

        all_at_t = synee_a.n_pairs
        assert all_at_t == nsp['N_e']*(nsp['N_e']-1)

        ax.plot(synee_a.t, active_at_t/all_at_t, lw=2, label="active")

        # trace2 - only synapse active and larger than c
        measurable_at_t = [np.sum(synee_a.active(k)[2] > nsp['strct_c'])
                           for k in range(len(synee_a))]

        ax.plot(synee_a.t, np.array(measurable_at_t)/all_at_t,
                lw=2, color='deepskyblue', label=f'active and a > {nsp["strct_c"]}')
    
    except FileNotFoundError:
        print(bpath[-4:], "reports: No n_active data!")
        ax.set_title("No data found")

    try: 
        with open(bpath+'/raw/c_stat.p', 'rb') as pfile:
            c_stat = pickle.load(pfile)

        print(c_stat['t'])
        print(c_stat['c'])

        ax.plot(c_stat['t'], c_stat['c'], color='red', lw=2)
        
    except FileNotFoundError:
        print(bpath[-4:], "reports: No n_active data!")
        ax.set_title("no cstat data")

    ax.set_xlabel('time [s]')
    ax.set_ylabel('fraction of synapses active')
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')        
