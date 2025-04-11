from brian2 import PoissonGroup, Synapses, NeuronGroup, BrianObject
from brian2.units import *
import numpy as np


def strong_mem_noise(tr, GExc: NeuronGroup, GInh: NeuronGroup) -> [BrianObject]:
    """
        Special kind of Poisson membrane noise that is so strong that it immediately enables
        the post-synaptic neuron.

        Related parameters:

        - :data:`net.standard_params.strong_mem_noise_active`
        - :data:`net.standard_params.strong_mem_noise_rate`
    """

    def strong_mem_noise_on_group(G: NeuronGroup):
        GNoise = PoissonGroup(G.N, rates=tr.strong_mem_noise_rate)
        SynNoise = Synapses(source=GNoise, target=G, on_pre="V_post = -40*mV")
        SynNoise.connect(condition="i==j")
        return [GNoise, SynNoise]

    return strong_mem_noise_on_group(GExc) + strong_mem_noise_on_group(GInh)


def cramer_noise(tr, GExc: NeuronGroup, GInh: NeuronGroup) -> [BrianObject]:
    """
        Noise inspired by Cramer et al. 2020 (preprint) where a certain number Kext of incoming
        connections to each neuron is replaced by external noise.

        Related parameters:

        - :data:`net.standard_params.cramer_noise_active`
        - :data:`net.standard_params.cramer_noise_Kext`
        - :data:`net.standard_params.cramer_noise_rate`
    """

    namespace = tr.netw.f_to_dict(short_names=True, fast_access=True)
    Kext = tr.cramer_noise_Kext
    if tr.p_ee_init > 0.0:
        raise NotImplementedError()
    p_ee_target = tr.p_ee/(1-Kext) if Kext < 1 else tr.p_ee
    p_ie_target = tr.p_ie/(1-Kext) if Kext < 1 else tr.p_ie
    conductance_prefix = "" if tr.syn_cond_mode == "exp" else "x"

    GNoise = PoissonGroup(tr.cramer_noise_N, rates=tr.cramer_noise_rate, dt=0.1*ms)
    SynNoiseE = Synapses(source=GNoise, target=GExc, on_pre=f"{conductance_prefix}ge_post += a_ee/norm_f_EE", namespace=namespace)
    SynNoiseE.connect(p=Kext*p_ee_target)
    SynNoiseI = Synapses(source=GNoise, target=GInh, on_pre=f"{conductance_prefix}ge_post += a_ee/norm_f_EE", namespace=namespace)
    SynNoiseI.connect(p=Kext*p_ie_target)

    return [GNoise, SynNoiseE, SynNoiseI]


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