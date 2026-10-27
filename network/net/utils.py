
import os
import numpy as np
from brian2.units import um, meter, second


# record of the binary weight snapshot files written by output_files.cpp
WEIGHT_RECORD = np.dtype([('i', '<i4'), ('j', '<i4'), ('a', '<f8')])


def read_weight_snapshots(fname, syn, final_t=None, all_pairs=False):
    """
        Reads (and deletes) a binary weight snapshot file written by cpp_methods.record_weights:
        for each snapshot a header record (-1, -1, t), followed by one record (i, j, a) per
        active synapse, in the order of the synapses. If final_t is given, the current weights
        of the active synapses of syn are added as a last snapshot at time final_t.

        :param all_pairs: True if syn contains all pairs i != j (in the order of
                          connect(condition='i != j')), otherwise the pre- and postsynaptic
                          neurons of all synapses are stored
        :return: dict with the snapshot times 't' and the 'i', 'j', 'a' of the active synapses of
                 all snapshots concatenated, snapshot k being [start[k], start[k+1]); read with
                 analysis.utils.WeightSnapshots
    """
    records = np.empty(0, dtype=WEIGHT_RECORD)
    if os.path.exists(fname):
        records = np.fromfile(fname, dtype=WEIGHT_RECORD)
        os.remove(fname)
    is_header = records['i'] == -1
    t = records['a'][is_header]
    # number of synapse records before each header
    start = np.flatnonzero(is_header) - np.arange(len(t))
    records = records[~is_header]
    i, j, a = records['i'], records['j'], records['a']

    if final_t is not None:
        active = np.flatnonzero(syn.syn_active[:] == 1)
        t = np.append(t, float(final_t / second))
        start = np.append(start, len(a))
        i = np.concatenate([i, syn.i[:][active]])
        j = np.concatenate([j, syn.j[:][active]])
        a = np.concatenate([a, syn.a[:][active]])

    return {'t': t * second, 'start': np.append(start, len(a)), 'i': i, 'j': j, 'a': a,
            'n_pairs': len(syn), 'N_pre': syn.source.N, 'N_post': syn.target.N,
            'pairs': None if all_pairs else (np.array(syn.i[:]), np.array(syn.j[:]))}


def generate_connections(N_tar, N_src, p, same=False):
    ''' 
    connect source to target with probability p
      - if populations SAME, avoid self-connection
      - if not SAME, connect any to any avoididing multiple
    return list of sources i and targets j
    '''
    nums = np.random.binomial(N_tar-1 if same else N_tar, p, N_src)
    i = np.repeat(np.arange(N_src), nums)
    j = []
    if same:
        for k,n in enumerate(nums):
            # all targets except k itself
            j+=list(np.random.choice([*range(k)]+[*range(k+1,N_tar)],
                                     size=n, replace=False))
    else:
        for k,n in enumerate(nums):
            j+=list(np.random.choice([*range(N_tar)],
                                     size=n, replace=False))

    return i, np.array(j), None, None


def generate_N_connections(N_tar, N_src, N, same=False):
    ''' 
    connect source to target with N connections per target

    return list of sources i and targets N_tar*N
    '''
    if same:
        raise NotImplementedError

    i = np.array([])
    j = np.repeat(range(N_tar), N)

    for k in range(N_tar):
        srcs = np.random.choice(range(N_src), size=N, replace=False)
        i = np.concatenate((i,srcs))

    i,j = i.astype(int), j.astype(int)
    assert len(i)==len(j)

    return i,j



# distance dependent connectivity (neurons on a square sheet), following Miner & Triesch (2016)

def gaussian(x, u, s):
    g = (2 / np.sqrt(2 * np.pi * s * s)) * np.exp(-(x - u) * (x - u) / (2 * s * s))  # gaussian used in Miner paper
    return g


def distances(src_x, src_y, tar_x, tar_y, grid_size, grid_wrap):
    """
        Distances between all source (rows) and target (columns) neurons, without units
        (positions and grid_size in metre). With grid_wrap, the sheet is a torus.
    """
    d = np.abs(np.stack([src_x, src_y], axis=1)[:, None, :] -
               np.stack([tar_x, tar_y], axis=1)[None, :, :])
    if grid_wrap:
        d = np.where(d > grid_size/2, grid_size - d, d)
    return np.sqrt(np.sum(d**2, axis=-1))


def _dd_probability_blocks(src_x, src_y, tar_x, tar_y, sigma, grid_size, grid_wrap, same, block_rows):
    """ Yields (first row, p) for blocks of source rows, p = Gaussian of the distances
        (self-connections set to 0 if source and target pool are the same). """
    for r0 in range(0, len(src_x), block_rows):
        r1 = min(r0 + block_rows, len(src_x))
        dist = distances(src_x[r0:r1], src_y[r0:r1], tar_x, tar_y, grid_size/meter, grid_wrap)
        p = gaussian(dist * meter / um, 0, np.array(sigma / um))
        if same:
            p[np.arange(r1 - r0), np.arange(r0, r1)] = 0.0
        yield r0, p


def generate_dd_connectivity(tar_x, tar_y, src_x, src_y, sigma, grid_size, grid_wrap, same=True, sparseness=1,
                             pair_order=False):
    """
        Generates distance-dependent connectivity: each source-target pair is connected
        independently with probability min(1, c*p(d)), where p(d) is a Gaussian of the distance d
        with standard deviation sigma, and c is chosen such that the expected number of connections
        is sparseness*(number of possible pairs). As long as c*p(d) <= 1 for all pairs, this is the
        same distance dependence as for the insertion of new EE synapses by structural plasticity
        (insert_P*p_distance). If the sparseness is too high for a Gaussian profile with this sigma,
        all close pairs are connected (clipped Gaussian).
        Self-connections are omitted if source and target pool are the same.

        The pairs are processed in blocks of source neurons, so that no full source x target
        matrix is kept in memory.

        :param tar_x, tar_y: target pool coordinates (in metre, without units)
        :param src_x, src_y: source pool coordinates (in metre, without units)
        :param sigma: standard deviation of the Gaussian (with units)
        :param grid_size: side length of the square sheet (with units)
        :param grid_wrap: if true, the sheet is a torus
        :param same: True if source and target pool are the same, False otherwise
        :param sparseness: expected fraction of connected pairs, from [0 1]
        :param pair_order: (only with same=True) instead of the connections, return two arrays over
                           all pairs i != j in the order (0,1), (0,2), ..., (1,0), (1,2), ...:
                           whether each pair is connected (0/1), and p(d) of each pair
        :return: source indices and target indices of the connections (or see pair_order)
    """
    n_src, n_tar = len(src_x), len(tar_x)
    block_rows = max(1, 2**22 // n_tar)
    blocks = lambda: _dd_probability_blocks(src_x, src_y, tar_x, tar_y, sigma, grid_size, grid_wrap,
                                            same, block_rows)
    # keep the probabilities if they are small enough, otherwise recompute them in each pass
    if n_src * n_tar <= 2**25:
        cached = list(blocks())
        blocks = lambda: iter(cached)

    total, p_max, n_nonzero = 0., 0., 0
    for _, p in blocks():
        total += np.sum(p)
        p_max = max(p_max, np.max(p))
        n_nonzero += np.count_nonzero(p)

    n_pairs = (n_src-1) * n_tar if same else n_src * n_tar
    n_expected = sparseness * n_pairs
    c = n_expected / total
    if c * p_max > 1:
        # clipped Gaussian: find c such that sum(min(1, c*p)) is the expected count
        if n_expected > n_nonzero:
            raise ValueError(f"Cannot create {n_expected:.0f} connections from {n_nonzero} pairs")
        expected_count = lambda c: sum(np.minimum(1, c*p).sum() for _, p in blocks())
        lo, hi = c, c
        while expected_count(hi) < n_expected:
            hi *= 2
        for _ in range(100):
            c = (lo + hi) / 2
            if expected_count(c) < n_expected:
                lo = c
            else:
                hi = c
        n_clipped = sum(np.count_nonzero(c*p >= 1) for _, p in blocks())
        print(f"Distance-dependent connectivity (sigma {sigma}, sparseness {sparseness}): "
              f"connection probability clipped to 1 for {n_clipped/(n_src*n_tar):.1%} of the pairs")

    in_src, in_trg, connected_pairs, p_pairs = [], [], [], []
    for r0, p in blocks():
        connected = np.random.uniform(size=p.shape) < np.minimum(1, c*p)
        if pair_order:
            off_diagonal = np.ones(p.shape, dtype=bool)
            off_diagonal[np.arange(p.shape[0]), np.arange(r0, r0 + p.shape[0])] = False
            connected_pairs.append(connected[off_diagonal])
            p_pairs.append(p[off_diagonal])
        else:
            rows, cols = np.nonzero(connected)
            in_src.append(rows + r0)
            in_trg.append(cols)
    if pair_order:
        return np.concatenate(connected_pairs).astype(int), np.concatenate(p_pairs)
    return np.concatenate(in_src), np.concatenate(in_trg)
