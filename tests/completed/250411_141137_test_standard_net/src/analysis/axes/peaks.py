
import pickle
import numpy as np
from brian2.units import mV, ms, second


def get_peak_statistics(bpath, nsp, peak_times, peak_width):

    with open(bpath+'/raw/gexc_spks.p', 'rb') as pfile:
        GExc_spks = pickle.load(pfile)
    with open(bpath+'/raw/ginh_spks.p', 'rb') as pfile:
        GInh_spks = pickle.load(pfile)

    peak_stats = []

    for peak_time in peak_times:

        statistic = {'peak_time': peak_time}

        t_idx = np.logical_and(GExc_spks['t']/ms > (peak_time-(peak_width/2))/ms,
                               GExc_spks['t']/ms < (peak_time+(peak_width/2))/ms)

        times = GExc_spks['t'][t_idx]/ms - peak_time/ms
        n_idx = GExc_spks['i'][t_idx]

        statistic['participating_neurons'] = np.unique(n_idx)

        peak_stats.append(statistic)
        

    return peak_stats


