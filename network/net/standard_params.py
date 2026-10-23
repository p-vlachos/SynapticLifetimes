
from brian2.units import *

# Please note: in accordance with Python attribute documentation
# the """ docstrings """ are AFTER the variable they document
# nota bene: this can be extracted by Sphinx

N_e = 400
N_i = int(0.2*N_e)

tau = 20.*ms                 # membrane time constant
""" seems to be used only for the Poisson implementation (not memnoise) """

syn_cond_mode = 'exp'
"""Conductance mode of EE synapses, one of exp, alpha, biexp"""
syn_cond_mode_EI = 'exp'
"""Conductance mode of EI synapses, one of exp, alpha, biexp"""
tau_e = 5.*ms
"""
    EPSP time constant
    when using the biexp conductance model this refers to tau2
"""
tau_i = 10.*ms
"""IPSP time constant"""
tau_e_rise = 0.5*ms
"""
    Used by biexp conductance model
    c.f. https://brian2.readthedocs.io/en/stable/user/converting_from_integrated_form.html
    
    corresponds to tau1
"""
tau_i_rise = 0.15*ms
"""Used by biexp conductance model """
norm_f_EE = 1.0
"""Used by alpha, biexp conductance model"""
norm_f_EI = 1.0
"""Used by alpha, biexp conductance model"""
El = -60.*mV
"""resting value"""
Ee = 0.*mV
"""reversal potential excitation"""
Ei = -80.*mV
"""reversal potential inhibition"""
mu_e = 9.0*mV
"""memnoise μ for excitatory neurons"""
mu_i = 8.5*mV
"""memnoise μ for inhibitory neurons"""
sigma_e = 0.5**0.5*mV
"""memnoise σ for excitatory neurons"""
sigma_i = 0.5**0.5*mV
"""memnoise σ for inhibitory neurons"""

# synaptic delay
syn_delay_active = 0
""" enables pre-synaptic delays """
syn_dd_delay_active = 0
""" enables distance-dependent pre-synaptic delays """
synEE_delay = 4.7*ms
""" for EE synapses, according to biexp peak time with standard params """
synEE_delay_windowsize = 0*ms
""" to use uniform delay distribution, set this to the width of the distribution """
synEI_delay = 4.25*ms
""" for EI synapses, according to biexp peak time with standard params """
synEI_delay_windowsize = 0*ms
""" to use uniform delay distribution, set this to the width of the distribution """
synII_delay = 0*ms
synII_delay_windowsize = 0*ms
synIE_delay = 0*ms
synIE_delay_windowsize = 0*ms

Vr_e = -60.*mV
"""initial V is drawn from Uniform(Vr_e, Vt_e)"""
Vr_i = -60.*mV
"""initial V is drawn from Uniform(Vr_i, Vt_i)"""
Vt_e = -50.*mV
"""initial Vt for excitatory neurons"""
Vt_i = -51.*mV
"""initial Vt for inhibitory neurons"""

a_ee = 0.005
"""initial activity of EE synapses"""
a_ie = 0.005
"""initial activity of IE synapses"""
a_ei = 0.005
"""initial activity of EI synapses"""
a_ii = 0.005
"""initial activity of II synapses"""

p_ee = 0.15
"""initial probability for synapse connection being active"""
p_ie = 0.15
"""initial probability for synapse connection being active"""
p_ei = 0.5
"""initial probability for synapse connection being active"""
p_ii = 0.5
"""initial probability for synapse connection being active"""

taupre = 15*ms
taupost = 30*ms
Aplus = 0.0015
Aminus = -0.00075
amax = 2.0
amin = 0.0
""" Minimal value for synapses, applied on STDP & scaling events. """
amin_i = -1.0
""" Set to something >= 0 if should be different from amin """
amax_i = -1.0
""" Set to something >= 0 if should be different from amax """

tau_r = 15*second
""" Time constant of postsynaptic firing rate average decay. """

external_mode = 'memnoise'
"""Either 'memnoise' or 'poisson' but latter seems to be unimplemented"""

# poisson 
# NOTES: Poissonian input as it is now it is either independent for each neuron (1-to-1) or from a pool of neurons
#        Additionally, in both cases, input to excitatory and inhibitory neurons are independent
PInp_mode = 'pool'
""" either 'pool' or 'indep', only applies if ``external_mode`` is 'poisson' """
NPInp = 1000
"""Number of poisson external input neurons on excitatory neurons"""
NPInp_1n = 10
"""Number of incoming external connections per target"""
NPInp_inh = 1000
"""Number of poisson external input neurons on inhibitory neurons"""
NPInp_inh_1n = 10
"""Number of incoming external connections per inhibitory target"""
PInp_rate = 1270*Hz
"""Firing rate of Poisson input to excitatory neurons"""
PInp_inh_rate = 1250*Hz
"""Firing rate of Poisson input to inhibitory neurons"""
a_EPoi = 0.005
"""Parameter for ``external_mode``=``poisson``"""
a_IPoi = 0.
"""Parameter for ``external_mode``=``poisson``"""
p_EPoi = 0.2  # unused
p_IPoi = 0.1  # unused

refractory_exc = "False"
"""
    Determine the duration or condition of the (absolute) refractory period of excitatory neurons.
    Directly passed to :class:`brian2.NeuronGroup`'s refractory parameter.
    Set to `"False"` to disable.
"""
refractory_inh = "False"
"""
    Determine the duration or condition of the (absolute) refractory period of inhibitory neurons.
    Directly passed to :class:`brian2.NeuronGroup`'s refractory parameter.
    Set to `"False"` to disable.
"""

# synapse noise
syn_noise = 1
"""enable/disable synapse noise"""
syn_noise_type = 'additive'
"""can be either ``additive`` or ``multiplicative``."""
syn_sigma = 1e-09/second
"""synapse noise sigma"""
synEE_mod_dt = 100*ms
"""dt between update of state variables"""

#STDP
stdp_active = 1
synEE_rec = 1
"""enable recording of EE synapse spikes"""
ATotalMax = 0.2
"""the normalization target is drawn from a normal distribution with this expected value"""
sig_ATotalMax = 0.05
"""the normalization target is drawn from a normal distribution with this sigma"""

#iSTDP
istdp_active = 1
taupre_EI = 20*ms
taupost_EI = 20*ms
synEI_rec = 1
"record EI synapse spikes"
iAplus = -1.0
"""potentiation maximum for iSTDP, -1 means same as Aplus"""
iAminus = -1.0
"""depression maximum for iSTDP, -1 means same as Aminus, only used for ``dbexp`` mode"""

# scaling
scl_active = 1
"""EE synaptic scaling"""
dt_synEE_scaling = 25*ms
"""time step for synaptic scaling"""
eta_scaling = 0.25

# iscaling
iscl_active = 1
iATotalMax = 0.7/6
sig_iATotalMax = 0.025
syn_iscl_rec = 0
"""record inhibitory synaptic scaling (via CPP methods)"""
eta_iscaling = -1.0
""" Learning rate for normalization of EI connections. If `-1.0` it's set to same as :member:`eta_scaling`. """

scaling_kappa = 2.5 * Hz
scaling_eta = 0.1
scaling_dt = 10*second

# structural plasticity
strct_active = 1
"""enable/disable structural plasticity"""
insert_P = 0.0002
"""synapse creation probability"""
strct_dt = 1000*ms
"""structural plasticity timestep"""
a_insert = 0.
"""initial activity of synapse on activation"""
p_inactivate = 0.01
"""
    if mode is ``zero`` this is the probability for a synapse to become inactive if it doesn't have enough activity
"""
strct_c = 0.002
"""threshold for ``zero`` mode"""

T1 = 1*second
"""initial recording phase with all recorders active"""
T2 = 10*second
"""main simulation, active recorders: turnover, C_stat, SynEE_a"""
T3 = 1*second
"""all recorders active"""
T4 = 1*second
"""record STDP and weight scaling mechanisms"""
T5 = 5*second
"""freeze network and record exc spikes for cross correlations"""
dt = 0.1*ms
n_threads = 1

# distance-dependent connectivity
"""
Implemented as described in Miner et al (2016). ddcon can be enabled/disabled for all model,
there is no possibility to enable it only for specific synapse type. 
"""
ddcon_active = 0
"""enable/disable distance-dependent connectivity"""
grid_size = 1000*um
"""side length of the square-shaped plane"""
half_width = 200*um
"""half-width in gaussian probability density function, see Miner et al (2016) for 
an explanation why 200um is a good value"""
grid_wrap = 0
""" If 1, wrap the grid around the edges to the opposite edge. """

# recording
memtraces_rec = 1
vttraces_rec = 1
getraces_rec = 1
gitraces_rec = 1
gfwdtraces_rec = 1
rates_rec = 1
"""additionally to recording averaged rates from EE and EI groups each per time step, record smoothed rates over 25ms"""
anormtar_rec = 0
"""record ATotalMax, only really makes sense when being used with `scl_mode` = 'proportional'"""

nrec_GExc_stat = 3
"""how many of excitatory neurons to record"""
nrec_GInh_stat = 3
"""how many of inhibitory neurons to record"""
GExc_stat_dt = 2.*ms
"""time step of recording excitatory neurons"""
GInh_stat_dt = 2.*ms
"""time step of recording inhibitory neurons"""

synee_atraces_rec = 1
synee_activetraces_rec = 0
synee_Apretraces_rec = 1
synee_Aposttraces_rec = 1
n_synee_traces_rec = 1000
"""number of EE synaptic traces to record"""
synEE_stat_dt = 2.*ms
"""time step of EE recording of synaptic traces"""

synei_atraces_rec = 1
synei_activetraces_rec = 1
synei_Apretraces_rec = 1
synei_Aposttraces_rec = 1
n_synei_traces_rec = 1000
"""number of EI synaptic traces to record"""
synEI_stat_dt = 2.*ms
"""time step of EI recording of synaptic traces"""

syn_scl_rec = 1
"""record synaptic scaling (via CPP methods)"""
stdp_rec_T = 1.*second
scl_rec_T = 0.1*second
"""additional time to record synaptic scaling after T3 (so first x seconds of T4)"""

synEEdynrec = 1
"""enable/disable recording of EE synaptic dynamics (``a`` variable)"""
synEIdynrec = 1
"""enable/disable recording of EI synaptic dynamics (``a`` variable)"""
syndynrec_dt = 1*second
syndynrec_npts = 10
"""desired number of recordings of synaptic dynamics"""

turnover_rec = 1
"""recording of synaptic turnover"""
T2_spks_rec = 0
"""recording of excitatory and inhibitory spikes during phase T2"""
synee_a_nrecpoints = 10
"""desired magnitude of EE synaptic recordings"""
synei_a_nrecpoints = 10
"""desired magnitude of EI synaptic recordings"""

crs_crrs_rec = 1
"""record and calculate cross-correlations between pre-synaptic and post-synaptic spike trains"""

adjust_insertP = 0
"""enable/disable homeostasis mechanism for EE synapse insertion probability"""
adjust_insertP_mode = "adjust_rate"
"""
modes for homeostatic adjustment of growth rate:

  - ```adjust_rate```
  - ```constant_count``` adjust growth rate so that number of synapses stays roughly constant
"""
csample_dt = 10*second
"""dt for insertP homeostasis mechanism"""

# post processing
pp_tcut = 1*second
"""synapse turnover is calculated starting at this time point"""

population_binned_rec = 0
"""efficiently records binned activity with GExc/GInh.dt resolution per population"""

random_seed = 578

