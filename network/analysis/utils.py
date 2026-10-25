
import numpy as np

def convert_to_EE_adjacency_matrix(syn_array, N):

    assert np.shape(syn_array) == tuple([N*(N-1)]), \
        "Synpase array doesn't have the expected shape"

    # construct i,j arrays
    i = np.repeat(range(N), N)
    j = np.tile(range(N), N)

    idx = (i!=j)
    i,j = i[idx], j[idx]

    adj_matrix = np.zeros((N,N))

    for syn_el, row_id, col_id in zip(syn_array,i,j):
        adj_matrix[row_id, col_id] = syn_el

    return adj_matrix


def convert_to_EI_adjacency_matrix(syn_array, N_E, N_I):

    assert np.shape(syn_array) == tuple([N_I*N_E]), \
        "Synpase array doesn't have the expected shape"

    # construct i,j arrays
    i = np.repeat(range(N_I), N_E)
    j = np.tile(range(N_E), N_I)

    adj_matrix = np.zeros((N_I,N_E))

    for syn_el, row_id, col_id in zip(syn_array,i,j):
        adj_matrix[row_id, col_id] = syn_el

    return adj_matrix

        

def topright_axis_off(ax):

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


if __name__ == "__main__":
    
    convert_to_adjacency_matrix([0,1], 2)


def linear_scale_for_empty_log_axes(fig):
    '''
    matplotlib cannot lay out a log-scaled axis without positive data
    (e.g. no lifetimes in a short simulation); show such axes with a
    linear scale instead
    '''
    for ax in fig.axes:
        if ax.get_xscale() == 'log' and not np.isfinite(ax.dataLim.minposx):
            ax.set_xscale('linear')
        if ax.get_yscale() == 'log' and not np.isfinite(ax.dataLim.minposy):
            ax.set_yscale('linear')


def population_rate(bpath, nsp, population, smooth_width=None):
    '''
    Population rate of the 'exc', 'inh' or 'pinp' (Poisson input) neurons, computed from the
    recorded spikes; identical to Brian2's PopulationRateMonitor (number of spikes per time
    step / (N*dt)) during the periods it recorded: T1, T3 and T5 (Poisson input: T1 and T3).

    :param smooth_width: if given, also return the rate smoothed with a Gaussian of this
                         standard deviation, as PopulationRateMonitor.smooth_rate(width=...)
    :return: dict with 't' and 'rate' (and 'smooth_rate' if smooth_width is given),
             as Brian2 quantities
    '''
    import pickle
    from brian2.units import second, Hz

    fname, n = {'exc': ('gexc_spks', nsp['N_e']), 'inh': ('ginh_spks', nsp['N_i']),
                'pinp': ('pinp_spks', nsp['NPInp'])}[population]
    with open(f'{bpath}/raw/{fname}.p', 'rb') as pfile:
        spikes = pickle.load(pfile)

    dt = float(nsp['dt'] / second)
    ends = np.cumsum([float(nsp[T] / second) for T in ['T1', 'T2', 'T3', 'T4', 'T5']])
    periods = [(0., ends[0]), (ends[1], ends[2]), (ends[3], ends[4])]
    if population == 'pinp':
        periods = periods[:2]

    # time steps of the recorded periods, and the number of spikes in each of them
    steps = np.concatenate([np.arange(int(round(start/dt)), int(round(end/dt)))
                            for start, end in periods])
    spike_steps = np.round(np.asarray(spikes['t'] / second) / dt).astype(np.int64)
    counts = np.zeros(len(steps))
    position = np.searchsorted(steps, spike_steps)
    recorded = (position < len(steps)) & (steps[np.minimum(position, len(steps)-1)] == spike_steps)
    np.add.at(counts, position[recorded], 1)

    rate = counts / dt / n
    result = {'t': steps * dt * second, 'rate': rate * Hz}
    if smooth_width is not None:
        width = float(smooth_width / second) / dt
        width_dt = int(np.round(2 * width))
        window = np.exp(-(np.arange(-width_dt, width_dt + 1) ** 2) * 1.0 / (2 * width ** 2))
        window = window * 1.0 / sum(window)
        result['smooth_rate'] = np.convolve(rate, window, mode='same') * Hz
    return result
