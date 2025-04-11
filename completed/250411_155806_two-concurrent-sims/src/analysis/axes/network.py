
import pickle
import numpy as np
from brian2.units import mV, ms, second, Hz

import mrestimator as mre
import scipy.signal 


def inst_rates(ax, bpath, nsp, tmin, tmax):

    try:
    
        with open(bpath+'/raw/gexc_rate.p', 'rb') as pfile:
            GExc_rate = pickle.load(pfile)
            GExc_smr = pickle.load(pfile)
        with open(bpath+'/raw/ginh_rate.p', 'rb') as pfile:
            GInh_rate = pickle.load(pfile)
            GInh_smr = pickle.load(pfile)
        # with open(bpath+'/raw/pinp_rate.p', 'rb') as pfile:
        #     PInp_rate = pickle.load(pfile)


        
        if len(GExc_rate['rate']) == 0:
            pass

        else:
            indx = np.logical_and(GExc_rate['t']>tmin, GExc_rate['t']<tmax)
            ax.plot(GExc_rate['t'][indx]/second, GExc_smr[indx]/Hz)


        if len(GInh_rate['rate']) == 0:
            pass

        else:
            indx = np.logical_and(GInh_rate['t']>tmin, GInh_rate['t']<tmax)
            ax.plot(GInh_rate['t'][indx]/second, GInh_smr[indx]/Hz,
                    color='red')


        # ax.set_ylim(nsp['Vr_e']/mV-1.5, nsp['Vt_e']/mV+1.5)
        ax.set_title('Rates')
        ax.set_xlabel('time [s]')
        ax.set_ylabel('rate [Hz]')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

    except EOFError:
        ax.axis('off')
    




def inst_rates_with_peaks(ax, bpath, nsp, tmin, tmax, tmax_vis=0):

    peak_times = []

    if tmax_vis == 0:
        tmax_vis = tmax

    try:
    
        with open(bpath+'/raw/gexc_rate.p', 'rb') as pfile:
            GExc_rate = pickle.load(pfile)
            GExc_smr = pickle.load(pfile)
        with open(bpath+'/raw/ginh_rate.p', 'rb') as pfile:
            GInh_rate = pickle.load(pfile)
            GInh_smr = pickle.load(pfile)
        # with open(bpath+'/raw/pinp_rate.p', 'rb') as pfile:
        #     PInp_rate = pickle.load(pfile)


        
        if len(GExc_rate['rate']) == 0:
            pass

        else:
    
            indx = np.logical_and(GExc_rate['t']>tmin,
                                  GExc_rate['t']<tmax)
            
            times = GExc_rate['t'][indx]/second
            rate  = GExc_smr[indx]/Hz
            
            ax.plot(times, rate)

            # distance: at least 75ms (750x0.1ms) between 
            peak_idx, _ = scipy.signal.find_peaks(rate, distance=750,
                                                  height=4.5)

            peak_times = times[peak_idx]*second

            ax.plot(times[peak_idx], rate[peak_idx], 'x',
                    color='black')
                    
               
            

        if len(GInh_rate['rate']) == 0:
            pass

        else:
            indx = np.logical_and(GInh_rate['t']>tmin, GInh_rate['t']<tmax)
            ax.plot(GInh_rate['t'][indx]/second, GInh_smr[indx]/Hz,
                    color='red')


        # ax.set_ylim(nsp['Vr_e']/mV-1.5, nsp['Vt_e']/mV+1.5)
        ax.set_title('Rates')
        ax.set_xlabel('time [s]')
        ax.set_ylabel('rate [Hz]')

        ax.set_xlim(left=tmin/second,right=tmax_vis/second)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

    except EOFError:
        ax.axis('off')


    return peak_times
    
        



def branching_ratio(ax, bpath, nsp, bin_w):

    with open(bpath+'/raw/gexc_spks.p', 'rb') as pfile:
        GExc_spks = pickle.load(pfile)
    # with open(bpath+'/raw/ginh_spks.p', 'rb') as pfile:
    #     GInh_spks = pickle.load(pfile)


    ts = GExc_spks['t']/ms
    ts = ts[ts>(nsp['T1']+nsp['T2']+nsp['T3']+nsp['T4'])/ms]
    ts = ts - (nsp['T1']+nsp['T2']+nsp['T3']+nsp['T4'])/ms

    assert(np.min(ts) >= 0)

    bins = np.arange(0, (nsp['T5']+bin_w)/ms, bin_w/ms)
    counts, bins = np.histogram(ts, bins=bins, density=False)

    print(np.shape(counts))
    # counts = np.reshape(counts, (len(counts),1))
    # print(np.shape(counts))
    
    rk = mre.coefficients(counts, dt=bin_w/ms, dtunit='ms', desc='')
    ft = mre.fit(rk)
    
    return rk, ft, (counts, bins)


def branching_ratio_chunks(ax, bpath, nsp, bin_w, chunk_n):
    with open(bpath + '/raw/gexc_spks.p', 'rb') as pfile:
        GExc_spks = pickle.load(pfile)

    ts = GExc_spks['t'] / ms
    ts = ts[ts > (nsp['T1'] + nsp['T2'] + nsp['T3'] + nsp['T4']) / ms]
    ts = ts - (nsp['T1'] + nsp['T2'] + nsp['T3'] + nsp['T4']) / ms

    assert (np.min(ts) >= 0)

    bins = np.arange(0, (nsp['T5'] + bin_w) / ms, bin_w / ms)
    counts, bins = np.histogram(ts, bins=bins, density=False)

    print(np.shape(counts))
    # counts = np.reshape(counts, (len(counts),1))
    # print(np.shape(counts))

    results = []

    chunk_size = int(len(counts) / chunk_n)
    chunk_bins = np.arange(0, len(counts)+1, chunk_size)
    for i in range(len(chunk_bins)-1):
        rk = mre.coefficients(counts[chunk_bins[i]:chunk_bins[i+1]], dt=bin_w / ms, dtunit='ms', desc='')
        ft = mre.fit(rk)
        complex_fit = mre.fit(rk, fitfunc=mre.f_complex)
        results.append( (rk, ft, complex_fit) )

    return results