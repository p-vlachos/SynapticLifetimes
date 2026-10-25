
import numpy as np
from brian2.units import um, meter


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



def generate_full_connectivity(Nsrc, Ntar=0, same=True):

    if same:
        i = []
        j = []
        for k in range(Nsrc):
            i.extend([k]*(Nsrc-1))
            targets = list(range(Nsrc))
            del targets[k]
            j.extend(targets)

        assert len(i)==len(j)
        return np.array(i), np.array(j)

    else:
        i = []
        j = []
        for k in range(Nsrc):
            i.extend([k]*Ntar)
            targets = list(range(Ntar))
            j.extend(targets)

        assert len(i)==len(j)
        return np.array(i), np.array(j)


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


def generate_dd_connectivity(tar_x, tar_y, src_x, src_y, sigma, grid_size, grid_wrap, same=True, sparseness=1):
    """
        Generates distance-dependent connectivity: each source-target pair is connected
        independently with probability min(1, c*p(d)), where p(d) is a Gaussian of the distance d
        with standard deviation sigma, and c is chosen such that the expected number of connections
        is sparseness*(number of possible pairs). As long as c*p(d) <= 1 for all pairs, this is the
        same distance dependence as for the insertion of new EE synapses by structural plasticity
        (insert_P*p_distance). If the sparseness is too high for a Gaussian profile with this sigma,
        all close pairs are connected (clipped Gaussian).
        Self-connections are omitted if source and target pool are the same.

        :param tar_x, tar_y: target pool coordinates (in metre, without units)
        :param src_x, src_y: source pool coordinates (in metre, without units)
        :param sigma: standard deviation of the Gaussian (with units)
        :param grid_size: side length of the square sheet (with units)
        :param grid_wrap: if true, the sheet is a torus
        :param same: True if source and target pools are the same, False otherwise
        :param sparseness: expected fraction of connected pairs, from [0 1]
        :return: source indices, target indices, matrix of p(d) (sources x targets),
                 matrix of distances in metre (sources x targets)
    """
    dist = distances(src_x, src_y, tar_x, tar_y, grid_size/meter, grid_wrap)
    n_src, n_tar = dist.shape

    p_ = gaussian(dist * meter / um, 0, np.array(sigma / um))
    if same:
        np.fill_diagonal(p_, 0.0)

    n_pairs = (n_src-1) * n_tar if same else n_src * n_tar
    n_expected = sparseness * n_pairs
    c = n_expected / np.sum(p_)
    if c * np.max(p_) > 1:
        # clipped Gaussian: find c such that sum(min(1, c*p)) is the expected count
        if n_expected > np.count_nonzero(p_):
            raise ValueError(f"Cannot create {n_expected:.0f} connections from {np.count_nonzero(p_)} pairs")
        lo, hi = c, c
        while np.minimum(1, hi*p_).sum() < n_expected:
            hi *= 2
        for _ in range(100):
            c = (lo + hi) / 2
            if np.minimum(1, c*p_).sum() < n_expected:
                lo = c
            else:
                hi = c
        print(f"Distance-dependent connectivity (sigma {sigma}, sparseness {sparseness}): "
              f"connection probability clipped to 1 for {np.mean(c*p_ >= 1):.1%} of the pairs")
    p_connect = np.minimum(1, c*p_)

    in_src, in_trg = np.nonzero(np.random.uniform(size=p_.shape) < p_connect)
    return in_src, in_trg, p_, dist
