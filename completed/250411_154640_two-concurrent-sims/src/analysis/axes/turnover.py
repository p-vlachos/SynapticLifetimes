
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl

import numpy as np
from brian2.units import mV, ms, second

import pickle, powerlaw

from ..methods.process_turnover import extract_lifetimes, \
                                       extract_active_synapse_count, \
                                       extract_delta_a_on_spike




def lifetime_distribution_loglog(ax, bpath, nsp, bins,
                                 discard_t, with_starters, label_key=''):
    ''' 
    discard all values until discard_t
    '''
    if discard_t!=0.:
        raise NotImplementedError
    else:
        print("not discarding any ts")


    with open(bpath+'/raw/turnover.p', 'rb') as pfile:
        turnover = pickle.load(pfile)


    if not len(turnover) == 0:
        
        _lt, _dt = extract_lifetimes(turnover, nsp['N_e'], with_starters=with_starters)
        life_t, death_t = _lt*second, _dt*second

        if len(life_t) == 0:
            print('No recorded lifetimes, not plotting distribution')
            ax.set_title('No recorded lifetimes')
        else:
            b_min, b_max = nsp['dt']/ms, np.max(life_t/ms)
            bins = np.linspace(np.log10(b_min), np.log10(b_max), bins)

            label = ''
            if label_key!='':
                label=  r'$\text{' + label_key +\
                        '} = %f $' %(getattr(tr, label_key))
                
            ax.hist(life_t/ms, 10**bins, log=True, density=True, label=label)

            ax.set_title('Lifetime distribution')
            ax.set_xlabel('time [ms]')
            ax.set_xscale('log')
            ax.set_xlim(b_min,b_max)

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')



def lifetime_distribution_loglog_linear_bins(ax, bpath, nsp, bin_w,
                                             discard_t, initial,
                                             label = '', n_color=0,
                                             fit=False, density=False):


    with open(bpath+'/raw/turnover.p', 'rb') as pfile:
        turnover = pickle.load(pfile)


    if not len(turnover) == 0:
    
        _lt, _dt = extract_lifetimes(turnover, nsp['N_e'], initial=initial,
                                     t_cut = discard_t)
        life_t, death_t = _lt*second, _dt*second

        T = nsp['T1']+nsp['T2']+nsp['T3']
        print('Using T1+T2+T3 as T!!')

        counts, edges = np.histogram(life_t/second,
                                     bins=np.arange(nsp['dt']/second,T/second,
                                                    bin_w/second),
                                     density=density)
        centers = (edges[:-1] + edges[1:])/2.

        # label = ''
        # if label_key!='':
        #     label=  r'$\text{' + label_key +\
        #             r'} = \text{'+'%.2E' %(Decimal(getattr(tr, label_key))) +\
        #             r'}$'

        ax.plot(centers, counts, '.', markersize=2., label=label)#,
                #color=pl.cm.Greens(np.linspace(0.2,1,5)[n_color]))

        if fit and len(life_t)>25:
            fit = powerlaw.Fit(life_t/second, discrete=True)

            fit.plot_pdf(ax=ax, color='r')
            fit.power_law.plot_pdf(ax=ax,  color='r', linestyle='--')

            alpha = fit.power_law.alpha

            ax.text(0.65, 0.95, r'$\alpha = '+'%.4f' %(alpha) +'$',
                        horizontalalignment='left',
                        verticalalignment='top',
                        linespacing = 1.95,
                        fontsize=10,
                        color='r',
                        bbox={'boxstyle': 'square, pad=0.3',
                              'facecolor':'white', 'alpha':1,
                              'edgecolor':'none'},
                        transform = ax.transAxes,
                        clip_on=False)

        # if with_starters:
        #     ax.text(0.05, 0.1, 'with starters',
        #             horizontalalignment='left',
        #             verticalalignment='top',
        #             linespacing = 1.95,
        #             fontsize=10,
        #             color='green',
        #             bbox={'boxstyle': 'square, pad=0.3',
        #                   'facecolor':'white', 'alpha':1,
        #                   'edgecolor':'none'},
        #             transform = ax.transAxes,
        #             clip_on=False)
        # else:
        #     ax.text(0.05, 0.1, 'without starters',
        #             horizontalalignment='left',
        #             verticalalignment='top',
        #             linespacing = 1.95,
        #             fontsize=10,
        #             color='red',
        #             bbox={'boxstyle': 'square, pad=0.3',
        #                   'facecolor':'white', 'alpha':1,
        #                   'edgecolor':'none'},
        #             transform = ax.transAxes,
        #             clip_on=False)
        
            
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('synapse lifetimes (' + \
                 r'$\text{bin width} = \text{\SI{%d}{s}}$)' % (int(bin_w/second)))
    ax.set_xlabel('synapse lifetime [s]')

    if density:
        ax.set_ylabel('probability density')
    else:
        ax.set_ylabel('absolute occurrence')
            
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
            



def lifetime_distribution_loglog_add_fit(axs, bpath, nsp, discard_t,
                                         with_starters):
    ''' 
    discard all values until discard_t
    '''
    if discard_t!=0.:
        raise NotImplementedError
    else:
        print("not discarding any ts")


    with open(bpath+'/raw/turnover.p', 'rb') as pfile:
        turnover = pickle.load(pfile)


    if not len(turnover) == 0:
    
        _lt, _dt = extract_lifetimes(turnover, nsp['N_e'], with_starters)
        life_t, death_t = _lt*second, _dt*second

        if len(life_t)>25:
                                         
            fit = powerlaw.Fit(life_t/ms, discrete=True)
            alpha = fit.power_law.alpha

            for ax in axs:
                fit.plot_pdf(ax=ax, color='r')
                fit.power_law.plot_pdf(ax=ax,  color='r', linestyle='--')
                                         
                ax.text(0.65, 0.95, r'$\alpha = '+'%.4f' %(alpha) +'$',
                        horizontalalignment='left',
                        verticalalignment='top',
                        linespacing = 1.95,
                        fontsize=10,
                        color='r',
                        bbox={'boxstyle': 'square, pad=0.3',
                              'facecolor':'white', 'alpha':1,
                              'edgecolor':'none'},
                        transform = ax.transAxes,
                        clip_on=False)

