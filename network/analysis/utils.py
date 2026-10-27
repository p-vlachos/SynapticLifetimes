
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

        

class WeightSnapshots:
    '''
    Weight snapshots of the active synapses, as recorded by the network
    (raw/synee_a.p, synei_a.p, syneedynrec.p, syneidynrec.p; see
    net.utils.read_weight_snapshots). Only active synapses are stored,
    inactive synapses have weight 0.
    '''

    def __init__(self, fname):
        import pickle
        with open(fname, 'rb') as pfile:
            data = pickle.load(pfile)
        self.t = data['t']
        self.n_pairs, self.N_pre, self.N_post = data['n_pairs'], data['N_pre'], data['N_post']
        self._start, self._i, self._j, self._a = data['start'], data['i'], data['j'], data['a']
        self._pairs = data['pairs']

    def __len__(self):
        return len(self.t)

    def active(self, k):
        '''
        presynaptic neurons i, postsynaptic neurons j and weights a of the
        synapses active at snapshot k (in the order of the synapses)
        '''
        k = range(len(self))[k]
        s = slice(self._start[k], self._start[k+1])
        return self._i[s], self._j[s], self._a[s]

    def n_active(self):
        ''' number of active synapses at each snapshot '''
        return np.diff(self._start)

    def synapse_index(self, i, j):
        ''' index of the synapses (i, j) within the synapse group '''
        if self._pairs is None:
            # all pairs i != j, ordered by i, then j
            return i*(self.N_post-1) + j - (j > i)
        key = self._pairs[0].astype(np.int64)*self.N_post + self._pairs[1]
        order = np.argsort(key)
        return order[np.searchsorted(key[order], np.asarray(i, dtype=np.int64)*self.N_post + j)]

    def dense(self, k):
        ''' weights and activity (0/1) of all synapses at snapshot k '''
        i, j, a_active = self.active(k)
        idx = self.synapse_index(i, j)
        a, syn_active = np.zeros(self.n_pairs), np.zeros(self.n_pairs, dtype=int)
        a[idx], syn_active[idx] = a_active, 1
        return a, syn_active

    def weight_changes(self, k0, k1):
        '''
        weights at snapshot k0 and their changes until snapshot k1, of the
        synapses active at k0 or k1 (sorted by i, then j); a synapse that was
        inactive at one of the snapshots has weight 0 there, and synapses
        with weight 0 and change 0 are omitted
        '''
        (i0, j0, a0), (i1, j1, a1) = self.active(k0), self.active(k1)
        key0 = i0.astype(np.int64)*self.N_post + j0
        key1 = i1.astype(np.int64)*self.N_post + j1
        keys = np.union1d(key0, key1)
        w0, w1 = np.zeros(len(keys)), np.zeros(len(keys))
        w0[np.searchsorted(keys, key0)] = a0
        w1[np.searchsorted(keys, key1)] = a1
        dw = w1 - w0
        idx = np.logical_or(w0 != 0, dw != 0)
        return w0[idx], dw[idx]


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
