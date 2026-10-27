"""
Parameters of a run (./run.sh). Each entry of input_dict is a list of values: lists of length n
give n simulations (simulation k uses the k-th value of each list), lists of length 1 are used
for all simulations. Parameters not listed here have the values of standard_params.py.

Notation: XY means "X <- Y" (target <- source), e.g. EI = from inhibitory to excitatory
neurons. Synaptic weights a are the jumps of the synaptic conductance per presynaptic spike,
in units of the leak conductance.
"""
from brian2.units import ms, mV, second, ksecond, Hz, um
from .param_tools import *
import numpy as np

sigv = 1. / second      # (unused)

a_min_insert = [0.0363] # minimal EE weight: new synapses start with it (a_insert), weaker ones
                        # can be pruned (strct_c), and STDP/normalization keep weights above it (amin)
stdp_eta = 0.01         # scale of the STDP amplitudes (Aplus, Aminus, iAplus, iAminus)
ifactor = 1.75          # scale of the inhibitory weights and iSTDP amplitudes

input_dict = {# Simulation phases (run one after the other)
              'T1': [10 * second],      # initial phase: everything recorded (spikes, traces)
              'T2': [50000 * second],   # main phase: plasticity and synapse turnover (only turnover and weight snapshots recorded)
              'T3': [5 * second],       # everything recorded again
              'T4': [5 * second],       # recording of single STDP and normalization events (see *_rec_T)
              'T5': [15000 * second],   # plasticity frozen, spikes recorded
              'dt': [0.1 * ms],         # simulation time step

              'random_seed': [7910],#, 13278, 37459, 9827, 8179],   # seed of all random numbers
              'pp_tcut': [500 * second],    # synapse lifetimes/survival are computed from this time on

              # Network parameters
              'N_e': [1600],            # number of excitatory neurons
              'N_i': [ 320],            # number of inhibitory neurons

              'p_ee': [0.08],           # fraction of active EE synapses (initially, and target of the growth homeostasis)
              'p_ei': [0.24],           # connection probabilities of the static E<-I, I<-E, I<-I connections
              'p_ie': [0.24],
              'p_ii': [0.24],

              'a_ee': [0.0263*2.65],    # initial weights (EE and EI change by plasticity, IE and II are fixed)
              'a_ie': [0.0263*2.5],
              'a_ei': [0.0438*ifactor],
              'a_ii': [0.0438*ifactor*2.5],

              'ddcon_active': [1],      # 1: neurons on a square sheet (1000 um), connection probability falls
                                        #    with distance (Gaussian); 0: random connectivity
              'sigma_dd': [150*um],     # standard deviation of the Gaussian distance dependence
              'grid_wrap': [1],         # 1: the sheet is a torus (no edges)

              'syn_delay_active': [1],      # 1: synaptic transmission delays (otherwise none)
              'syn_dd_delay_active' : [0],  # 1: delays = distance / velocity (requires ddcon_active)
              'synEE_delay': [3.0*ms],              # otherwise uniform delays: mean ...
              'synEE_delay_windowsize': [1.5*ms],   # ... and width of the uniform distribution
              'synIE_delay': [1.0*ms],
              'synIE_delay_windowsize': [0.5*ms],
              'synII_delay': [1.0*ms],
              'synII_delay_windowsize': [0.5*ms],
              'synEI_delay': [0.5*ms],
              'synEI_delay_windowsize': [0.25*ms],

              # Neuronal parameters (conductance-based leaky integrate-and-fire neurons)
              'external_mode': ['memnoise'],    # external input: 'memnoise' (noise on the membrane potential)
                                                # or 'poisson' (Poisson spike trains)
              'mu_e': [0.0*mV],                 # mean of the external input (exc./inh. neurons)
              'mu_i': [0.0*mV],
              'sigma_e': [4.5*mV, 6.1*mV],      # strength of the membrane noise: 6.1 mV gives the
              'sigma_i': [4.5*mV, 6.1*mV],      # subcritical (noise-driven), 4.5 mV the reverberating regime
              'Vt_e': [-50. * mV],      # spike threshold
              'Vt_i': [-50. * mV],
              'Vr_e': [-70. * mV],      # reset potential after a spike (initial V: uniform between Vr and Vt)
              'Vr_i': [-70. * mV],
              'refractory_exc': ['2*ms'],   # refractory period
              'refractory_inh': ['1*ms'],

              # Synaptic parameters
              'syn_cond_mode': ['exp'],     # time course of the excitatory conductance: 'exp' or 'biexp'
              'syn_cond_mode_EI': ['exp'],  # the same for the inhibitory conductance
              'tau_e': [5*ms],              # decay time constant of the excitatory conductance
              'tau_i': [10*ms],             # decay time constant of the inhibitory conductance

              'syn_noise': [1],                 # 1: multiplicative noise on the EE and EI weights
              'syn_sigma_EE': [1e-9/second],    # its strength (1e-9/s: negligible)
              'syn_sigma_EI': [1e-9/second],

              # Plasticity parameters
              # --- STDP (EE synapses, all-to-all spike pairs) ---
              'stdp_active': [1],
              'Aplus': [1.6253*stdp_eta],       # weight change of a pre-before-post pair (at zero delay)
              'Aminus': [-0.8127*stdp_eta],     # weight change of a post-before-pre pair

              # --- inhibitory STDP (EI synapses) ---
              'istdp_active': [1],
              'iAplus': [1.6253*ifactor*stdp_eta],
              'iAminus': [-0.5*1.6253*ifactor*stdp_eta],
              'taupre_EI': [20*ms],             # time constants of the iSTDP window
              'taupost_EI': [20*ms],

              # --- Structural plasticity (EE synapses are created and pruned) ---
              'strct_active': [1],
              'strct_dt': [1000 * ms],          # interval of the structural plasticity steps
              'strct_c': a_min_insert,          # synapses with weight <= strct_c ...
              'p_inactivate': [0.1],            # ... are pruned with this probability per step
              'a_insert': a_min_insert,         # weight of a newly created synapse
              'insert_P': [0.05],               # creation probability per step (times the distance factor);
                                                # not used with adjust_insertP_mode 'constant_count'
              'adjust_insertP': [1],            # 1: growth homeostasis sets insert_P
              'adjust_insertP_mode': ['constant_count'],  # 'constant_count': number of active synapses stays
                                                          # constant; 'adjust_rate': fraction approaches p_ee
              'csample_dt': [1 * second],       # interval of the growth homeostasis

              # --- Homeostatic normalization of the input weights of each excitatory neuron ---
              'dt_synEE_scaling': [100 * ms],   # interval of the normalization
              'eta_scaling': [1.0],             # fraction of the deviation from the target corrected per step
                                                # (1: the sum is set exactly to the target)

              'scl_active': [1],                # 1: normalization of the EE input weights (sum -> ANormTar)
              'scl_scaling_kappa': [4.5*Hz],    # target firing rate of the adaptive target ANormTar
              'scl_scaling_eta': [0.0001],      # change of ANormTar per scl_scaling_dt: eta*(kappa - rate)/kappa
              'scl_scaling_dt': [1 * second],   # interval of the adaptation of ANormTar

              'iscl_active': [1],               # 1: normalization of the EI input weights (sum -> iATotalMax)

              # --- General ---
              'tau_r': [15 * second],   # time constant of the firing rate estimate of the excitatory neurons
              'amin': a_min_insert,     # bounds of the EE weights
              'amax': [0.1662*2],
              'amin_i' : [0.005],       # bounds of the EI weights
              'amax_i' : [0.320],
              'ATotalMax': [3.4*2.65],  # initial normalization target (sum of the EE input weights) ...
              'sig_ATotalMax': [0.05],  # ... drawn per neuron with this standard deviation
              'iATotalMax': [3.4*ifactor],  # normalization target of the EI input weights (sum per neuron)

              # Recording parameters (T1, T3: traces of the first neurons and synapses)
              'memtraces_rec': [1],         # membrane potential traces
              'getraces_rec': [1],          # excitatory conductance traces
              'gitraces_rec': [1],          # inhibitory conductance traces
              'nrec_GExc_stat': [400],      # ... of this many excitatory ...
              'nrec_GInh_stat': [3],        # ... and inhibitory neurons
              'GExc_stat_dt': [2 * ms],     # ... with this time step
              'GInh_stat_dt': [2 * ms],
              'synee_atraces_rec': [1],         # EE traces: weight a ...
              'synee_activetraces_rec': [0],    # ... active/inactive ...
              'synee_Apretraces_rec': [0],      # ... STDP traces ...
              'synee_Aposttraces_rec': [0],
              'n_synee_traces_rec': [1000],     # ... of this many EE synapses (potential pairs)
              'synEE_stat_dt': [2 * ms],        # ... with this time step
              'synei_atraces_rec': [1],         # the same for EI synapses
              'synei_activetraces_rec': [0],
              'synei_Apretraces_rec': [0],
              'synei_Aposttraces_rec': [0],
              'n_synei_traces_rec': [1000],
              'synEI_stat_dt': [2 * ms],
              'synee_a_nrecpoints': [10],   # snapshots of all active EE weights every T2/synee_a_nrecpoints (+ one at the end)
              'synei_a_nrecpoints': [10],   # the same for EI weights
              'synEEdynrec': [1],           # EE weight snapshots every syndynrec_dt ...
              'synEIdynrec': [0],           # (EI)
              'syndynrec_dt': [1 * second],
              'syndynrec_npts': [10],       # ... this many at the beginning and at the end of T2
              'turnover_rec': [1],          # record every creation and pruning of a synapse
              'T2_spks_rec': [0],           # also record the spikes in T2 (large)
              'anormtar_rec': [1],          # record the normalization targets ANormTar with the neuron traces
              'syn_scl_rec': [0],           # record the single normalization changes ...
              'syn_iscl_rec': [0],          # (EI)
              'scl_rec_T': [1 * second],    # ... during the first scl_rec_T of T4
              'synEE_rec': [0],             # record the single STDP events of EE ...
              'synEI_rec': [0],             # ... and EI synapses ...
              'stdp_rec_T': [1 * second],   # ... during the first stdp_rec_T of T4
              'population_binned_rec': [0], # number of spikes per time step of each population (whole run)

}


name = 'hdf5_data'      # name of the run: run folder <timestamp>_<name>, data/<name>.hdf5
explore_dict = n_list(input_dict)

if __name__ == "__main__":
    print(name)
