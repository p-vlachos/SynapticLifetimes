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
    delays = np.random.uniform(low=syn_delay - syn_delay_windowsize / 2,
                               high=syn_delay + syn_delay_windowsize / 2,
                               size=shape)
    Syn.delay = delays
    return delays


def synapse_dd_delays(dist: um, Syn: Synapses):
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
    ap_velocity = 150 * um/ms # Action potential velocity (should be adjusted later based on data)
    syn_delay_prop_windowsize = 0.1 # Use to set delay std proportionally to distance (longer->more variance)
    delays = np.random.normal(dist/ap_velocity, np.ones_like(dist)*syn_delay_prop_windowsize)
    delays[delays < 0] = 0 # Ensure there are non-negative values

    Syn.delay = delays
    return delays