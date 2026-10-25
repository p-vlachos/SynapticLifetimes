from brian2 import PoissonGroup, Synapses, NeuronGroup, BrianObject
from brian2.units import *
import numpy as np


def synapse_delays(syn_delay: ms, syn_delay_windowsize: ms, Syn: Synapses, shape):
    """
        Configure pre-synaptic delays. Similar to Brunel 2000 these are uniformly distributed.
        The delay is the center of the uniform distribution, while the window size defines the
        total width of the distribution.

        Related parameters:

        - :data:`net.standard_params.synEE_delay`
        - :data:`net.standard_params.synEE_delay_windowsize`
        - :data:`net.standard_params.synEI_delay`
        - :data:`net.standard_params.synEI_delay_windowsize`
    """
    # need to create these for all synapses, not only the initially active ones
    delays = np.random.uniform(low=(syn_delay - syn_delay_windowsize / 2)/second,
                               high=(syn_delay + syn_delay_windowsize / 2)/second,
                               size=shape)
    Syn.delay = delays*second
    return delays  # in seconds


def synapse_dd_delays(dist, src, tar, Syn: Synapses, velocity, rel_sd):
    """
        Configure distance-dependent pre-synaptic delays: the delay of each synapse is
        drawn from a normal distribution with mean distance/velocity and a standard
        deviation proportional to the mean (negative values are set to zero).

        :param dist: distance matrix (in metre, without units) between all source (rows)
                     and target (columns) neurons, as returned by ``generate_dd_connectivity``
        :param src, tar: source and target indices of the synapses of ``Syn``

        Related parameters:

        - :data:`net.standard_params.syn_dd_delay_velocity`
        - :data:`net.standard_params.syn_dd_delay_rel_sd`
    """
    # need to create these for all synapses, not only the initially active ones
    mean = np.asarray(dist)[src, tar]*metre / velocity
    delays = np.random.normal(mean/second, rel_sd*mean/second)
    delays[delays < 0] = 0  # ensure there are non-negative values

    Syn.delay = delays*second
    return delays  # in seconds
