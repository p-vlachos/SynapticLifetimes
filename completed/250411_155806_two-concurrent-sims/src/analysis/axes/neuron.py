
import pickle
import numpy as np
import scipy.stats
from brian2.units import mV, ms, second


def raster_plot(ax, bpath, nsp, tmin, tmax):
    '''
    '''

    with open(bpath+'/raw/gexc_spks.p', 'rb') as pfile:
        GExc_spks = pickle.load(pfile)
    with open(bpath+'/raw/ginh_spks.p', 'rb') as pfile:
        GInh_spks = pickle.load(pfile)

    try:
        indx = np.logical_and(GExc_spks['t']/ms>tmin/ms, GExc_spks['t']/ms<tmax/ms)
        ax.scatter(GExc_spks['t'][indx]/second, GExc_spks['i'][indx], marker=",", s=0.1, linewidths=0.0)
    except AttributeError:
        print(bpath[-4:], "reports: AttributeError. Guess: no exc. spikes from",
              "{:d}s to {:d}s".format(int(tmin/second),int(tmax/second)))

    try:
        indx = np.logical_and(GInh_spks['t']/ms>tmin/ms, GInh_spks['t']/ms<tmax/ms)
        ax.scatter(GInh_spks['t'][indx]/second, GInh_spks['i'][indx]+nsp['N_e'],
                   marker=",", s=0.1, linewidths=0.0, color='red')
    except AttributeError:
        print(bpath[-4:], "reports: AttributeError. Guess: no inh. spikes from",
              "{:d}s to {:d}s".format(int(tmin/second),int(tmax/second)))


    ax.set_xlim(tmin/second, tmax/second)
    ax.set_xlabel('time [s]')
    ax.set_ylim(0, nsp['N_e'] + nsp['N_i'])
    
    # ax.set_title('T='+str(T/second)+' s')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')




def raster_plot_sorted(ax, bpath, nsp, tmin, tmax, lookup={}):
    '''
    '''

    with open(bpath+'/raw/gexc_spks.p', 'rb') as pfile:
        GExc_spks = pickle.load(pfile)
    with open(bpath+'/raw/ginh_spks.p', 'rb') as pfile:
        GInh_spks = pickle.load(pfile)

    try:

        t_idx = np.logical_and(GExc_spks['t']/ms>tmin/ms,
                              GExc_spks['t']/ms<tmax/ms)

        peak_center = (tmax+tmin)/(2*ms)

        times = GExc_spks['t'][t_idx]/ms - peak_center
        n_idx = GExc_spks['i'][t_idx]

        np.testing.assert_array_equal(np.sort(times),
                                      np.array(times))

        print(lookup)
        if lookup != {}:
            k = np.max(list(lookup.values()))+1
        else:
            k = 0

        print(k)
            
        n_idx_sorted = []

        for i in n_idx:

            if not lookup.get(i):
                lookup[i] = k
                k+=1 

            n_idx_sorted.append(lookup[i])


        ax.plot(times, n_idx_sorted, marker='.', color='blue',
                markersize=.5, linestyle='None')

        
    except AttributeError:

        print(bpath[-4:], "reports: AttributeError. Guess: no exc. spikes from",
              "{:d}s to {:d}s".format(int(tmin/ms),int(tmax/ms)))

    try:

        indx = np.logical_and(GInh_spks['t']/ms>tmin/ms, GInh_spks['t']/ms<tmax/ms)
        ax.plot(GInh_spks['t'][indx]/ms - peak_center,
                GInh_spks['i'][indx]+nsp['N_e'], marker='.',
                color='red', markersize=.5, linestyle='None')
        
    except AttributeError:

        print(bpath[-4:], "reports: AttributeError. Guess: no inh. spikes from",
              "{:d}s to {:d}s".format(int(tmin/ms),int(tmax/ms)))


    ax.set_xlim(tmin/ms-peak_center, tmax/ms-peak_center)
    ax.set_xlabel('time [ms]')
    ax.set_ylim(0, nsp['N_e'] + nsp['N_i'])
    
    # ax.set_title('T='+str(T/ms)+' s')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    return lookup





def raster_plot_with_peaks(ax, bpath, nsp, tmin, tmax):
    '''
    '''

    with open(bpath+'/raw/gexc_spks.p', 'rb') as pfile:
        GExc_spks = pickle.load(pfile)
    with open(bpath+'/raw/ginh_spks.p', 'rb') as pfile:
        GInh_spks = pickle.load(pfile)

    try:

        t_idx = np.logical_and(GExc_spks['t']/ms>tmin/ms,
                              GExc_spks['t']/ms<tmax/ms)

        times = GExc_spks['t'][t_idx]/second
        n_idx = GExc_spks['i'][t_idx]

        np.testing.assert_array_equal(np.sort(times),
                                      np.array(times))

        ax.plot(times, n_idx_sorted, marker='.', color='blue',
                markersize=.5, linestyle='None')

        
    except AttributeError:

        print(bpath[-4:], "reports: AttributeError. ",
              "Guess: no exc. spikes from",
              "{:d}s to {:d}s".format(int(tmin/second),int(tmax/second)))

    try:
        indx = np.logical_and(GInh_spks['t']/ms>tmin/ms,
                              GInh_spks['t']/ms<tmax/ms)
        
        ax.plot(GInh_spks['t'][indx]/second,
                GInh_spks['i'][indx]+nsp['N_e'], marker='.',
                color='red', markersize=.5, linestyle='None')
        
    except AttributeError:
        print(bpath[-4:], "reports: AttributeError. ",
              "Guess: no inh. spikes from",
              "{:d}s to {:d}s".format(int(tmin/second),int(tmax/second)))


    ax.set_xlim(tmin/second, tmax/second)
    ax.set_xlabel('time [s]')
    ax.set_ylim(0, nsp['N_e'] + nsp['N_i'])
    
    # ax.set_title('T='+str(T/second)+' s')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')





def raster_plot_poisson(ax, bpath, nsp, tmin, tmax):
    '''
    '''

    with open(bpath+'/raw/pinp_spks.p', 'rb') as pfile:
        PInp_spks = pickle.load(pfile)

    try:
        indx = np.logical_and(PInp_spks['t']/ms>tmin/ms, PInp_spks['t']/ms<tmax/ms)
        ax.plot(PInp_spks['t'][indx]/second, PInp_spks['i'][indx], marker='.',
                color='blue', markersize=.5, linestyle='None')
    except AttributeError:
        print(bpath[-4:], "reports: AttributeError. Guess: no PInp. spikes from",
              "{:d}s to {:d}s".format(int(tmin/second),int(tmax/second)))


    ax.set_xlim(tmin/second, tmax/second)
    ax.set_xlabel('time [s]')
    ax.set_ylim(0, nsp['NPInp'])
    
    # ax.set_title('T='+str(T/second)+' s')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    

    
def ge_plot(ax, bpath, nsp, tmin, tmax, i=0):

    with open(bpath+'/raw/gexc_stat.p', 'rb') as pfile:
        GExc_stat = pickle.load(pfile)

    try:
        ge_data=GExc_stat['ge']
        V_data = GExc_stat['V']

        indx = np.logical_and(GExc_stat['t']>tmin, GExc_stat['t']<tmax)

        # for i in range(np.shape(ge_data)[1]):
        ax.plot(GExc_stat['t'][indx]/second, (nsp['Ee']/mV-V_data[:,i][indx]/mV)*ge_data[:,i][indx], color='blue')

        #ax.set_xlim(tmin/second, tmax/second)
        ax.set_xlabel('time [s]')

        #ax.set_ylim(0, nsp['N_e'] + nsp['N_i'])
        # ax.set_title('T='+str(T/second)+' s')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

    except KeyError:
        ax.axis('off')

    
def gi_plot(ax, bpath, nsp, tmin, tmax, i=0):

    with open(bpath+'/raw/gexc_stat.p', 'rb') as pfile:
        GExc_stat = pickle.load(pfile)

    try:
        gi_data=GExc_stat['gi']
        V_data=GExc_stat['V']

        indx = np.logical_and(GExc_stat['t']>tmin, GExc_stat['t']<tmax)

        # for i in range(np.shape(gi_data)[1]):
        #     ax.plot(GExc_stat['t'][indx]/second, gi_data[:,i][indx])

        ax.plot(GExc_stat['t'][indx]/second, (nsp['Ei']/mV-V_data[:,i][indx]/mV)*gi_data[:,i][indx], color='red')

        #ax.set_xlim(tmin/second, tmax/second)
        ax.set_xlabel('time [s]')

        #ax.set_ylim(0, nsp['N_e'] + nsp['N_i'])
        # ax.set_title('T='+str(T/second)+' s')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        
    except KeyError:
        ax.axis('off')



    
def gegi_plot(ax, bpath, nsp, tmin, tmax, i=0):

    with open(bpath+'/raw/gexc_stat.p', 'rb') as pfile:
        GExc_stat = pickle.load(pfile)

    try:
        ge_data=GExc_stat['ge']
        gi_data=GExc_stat['gi']
        V_data=GExc_stat['V']

        indx = np.logical_and(GExc_stat['t']>tmin, GExc_stat['t']<tmax)

        # for i in range(np.shape(gi_data)[1]):
        #     ax.plot(GExc_stat['t'][indx]/second, gi_data[:,i][indx])

        ax.plot(GExc_stat['t'][indx]/second,
                (nsp['Ei']/mV-V_data[:,i][indx]/mV)*gi_data[:,i][indx] +
                (nsp['Ee']/mV-V_data[:,i][indx]/mV)*ge_data[:,i][indx],
                color='grey', alpha=0.7)

        #ax.set_xlim(tmin/second, tmax/second)
        ax.set_xlabel('time [s]')

        #ax.set_ylim(0, nsp['N_e'] + nsp['N_i'])
        # ax.set_title('T='+str(T/second)+' s')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        
    except KeyError:
        ax.axis('off')
        
        



def voltage_traces(ax, bpath, nsp, tmin, tmax):
    
    with open(bpath+'/raw/gexc_stat.p', 'rb') as pfile:
        GExc_stat = pickle.load(pfile)

    try:
        
        if len(GExc_stat['V']) == 0:
            pass

        else:

            indx = np.logical_and(GExc_stat['t']>tmin, GExc_stat['t']<tmax)

            for i in range(len(GExc_stat['V'].T)):
                ax.plot(GExc_stat['t'][indx]/second, GExc_stat['V'][:,i][indx]/mV)

            ax.set_ylim(nsp['Vr_e']/mV-1.5, nsp['Vt_e']/mV+1.5)
            ax.set_title('Membrane Voltage Traces')
            ax.set_xlabel('time [s]')
            ax.set_ylabel('voltage [mV]')

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')


    except KeyError:
        ax.axis('off')


def firing_rates_plot(ax, bpath, group, tbegin, tend, histargs=None):
    if histargs is None:
        histargs = dict()

    try:
        with open(f'{bpath}/raw/g{group}_spks.p', 'rb') as pfile:
            spks = pickle.load(pfile)

        T = tend - tbegin
        indx = np.logical_and(spks['t'] < tend, spks['t'] > tbegin)
        t_exc, id_exc = spks['t'][indx], spks['i'][indx]

        unique = np.unique(spks['i'][indx])

        fr_exc = [np.sum(id_exc == i) / (T / second) for i in unique]

        bins = np.arange(0, np.max(fr_exc), 0.25)
        ax.hist(fr_exc, bins=bins, density=True, label=f"[{tbegin/second}, {tend/second}] s", **histargs)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.set_xlabel('firing rate [Hz]')
        ax.set_ylabel('probability density')

    except (ValueError, KeyError) as e:
        print(e)
        ax.axis('off')
        
def firing_rates_plot_exc(ax, bpath, nsp):

    try:
        with open(bpath+'/raw/gexc_spks.p', 'rb') as pfile:
            GExc_spks = pickle.load(pfile)

        T_prev = nsp['T1']+nsp['T2']+nsp['T3']+nsp['T4']

        indx = GExc_spks['t']>T_prev
        t_exc, id_exc = GExc_spks['t'][indx], GExc_spks['i'][indx]

        fr_exc = [np.sum(id_exc==i)/(nsp['T5']/second) for i in range(nsp['N_e'])]

        ax.hist(fr_exc, bins=35, density=True)
  
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.set_xlabel('firing rate [Hz]')
        ax.set_ylabel('probability density')

    except (ValueError,KeyError) as e:
        ax.axis('off')

    
def firing_rates_plot_inh(ax, bpath, nsp):

    color, alpha = '#d62728', 0.7

    try:

        with open(bpath+'/raw/ginh_spks.p', 'rb') as pfile:
            GInh_spks = pickle.load(pfile)

        T_prev = nsp['T1']+nsp['T2']+nsp['T3']+nsp['T4']

        indx = GInh_spks['t']>T_prev
        t_inh, id_inh = GInh_spks['t'][indx], GInh_spks['i'][indx]

        fr_inh = [np.sum(id_inh==i)/(nsp['T5']/second) for i in range(nsp['N_i'])]

        ax.hist(fr_inh, bins=20, density=True, color=color, alpha=alpha)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.set_xlabel('firing rate [Hz]')
        ax.set_ylabel('probability density')

    except (ValueError,KeyError) as e:
        ax.axis('off')

    



def _translate_type(ztype):
    if ztype=='ins':
        return 'growth'
    if ztype=='prn':
        return 'pruning'
    
        
def syn_turnover_histogram(ax, bpath, nsp, cn='EE', ztype='ins'):

    ax.axis('on')

    if cn=='EE':
        cn_app=''
        color = 'blue'
    elif cn=='EI':
        cn_app='_EI'
        color='red'
    
    try:

        with open(bpath+'/raw/ins-prn_counts%s.p' %cn_app, 'rb') as pfile:
            ip_df = pickle.load(pfile)

        ax.hist(ip_df[ztype+'_c'], bins=35, color=color)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.set_xlabel('number of %s %s events' %(cn, _translate_type(ztype)))
        ax.set_ylabel('occurrence')

    except (ValueError,KeyError) as e:
        ax.axis('off')

    
        
def syn_turnover_EE_EI_correlation(ax, bpath, nsp, xtype='ins', ytype='prn'):

    ax.axis('on')

    try:

        with open(bpath+'/raw/ins-prn_counts.p', 'rb') as pfile:
            EE_df = pickle.load(pfile)

        with open(bpath+'/raw/ins-prn_counts_EI.p', 'rb') as pfile:
            EI_df = pickle.load(pfile)

        xdata, ydata = EE_df[xtype+'_c'], EI_df[ytype+'_c']
   
        ax.scatter(xdata, ydata, 2.)
        
        c,p = scipy.stats.pearsonr(xdata, ydata)

        ax.text(0.75, 0.05, '$c=%.3f$' %c, transform=ax.transAxes)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.set_xlabel('EE %s events' % _translate_type(xtype))
        ax.set_ylabel('EI %s events' % _translate_type(ytype))

    except (ValueError,KeyError) as e:
        ax.axis('off')
    


def syn_turnover_ins_prn_correlation(ax, bpath, nsp, cn='EE'):

    ax.axis('on')

    if cn=='EE':
        cn_app=''
        color = 'blue'
    elif cn=='EI':
        cn_app='_EI'
        color='red'
    
    try:

        with open(bpath+'/raw/ins-prn_counts%s.p' %cn_app, 'rb') as pfile:
            ip_df = pickle.load(pfile)

        ins_c, prn_c = ip_df['ins_c'], ip_df['prn_c']

        ax.scatter(ins_c, prn_c, 2., color=color)
      
        c,p = scipy.stats.pearsonr(ins_c, prn_c)

        ax.text(0.75, 0.05, '$c=%.3f$' %c, transform=ax.transAxes)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        
        ax.set_xlabel('%s growth events' % cn)
        ax.set_ylabel('%s prune events' % cn)

    except (ValueError,KeyError) as e:
        ax.axis('off')
    
