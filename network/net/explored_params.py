from brian2.units import ms, mV, second, ksecond, Hz, um
from .param_tools import *
import numpy as np

sigv = 1. / second

a_min_insert = [0.0363] # 
stdp_eta = 0.01         # 
ifactor = 1.75          # 

input_dict = {'T1': [10 * second],
              'T2': [50000 * second],   # 50_000
              'T3': [5 * second],
              'T4': [5 * second],
              'T5': [15000 * second],     # 15_000
              'dt': [0.1 * ms],
              
              'random_seed': [7910], #, 13278, 37459, 7910, 13278, 37459]
              'pp_tcut': [500 * second],# Synapse turnover calculation onset

              # Network parameters
              'N_e': [1600],
              'N_i': [ 320],

              'p_ee': [0.08],
              'p_ei': [0.24],
              'p_ie': [0.24],
              'p_ii': [0.24],

              'a_ee': [0.0263*2.65], # Initial
              'a_ie': [0.0263*2.5],
              'a_ei': [0.0438*ifactor],
              'a_ii': [0.0438*ifactor*2.5],

              'ddcon_active': [1],
              'half_width': [150*um],
              'grid_wrap': [1],

              'syn_delay_active': [1],
              'syn_dd_delay_active' : [0],  # I implemented this (keep at zero)
              'synEE_delay': [3.0*ms],
              'synEE_delay_windowsize': [1.5*ms],
              'synIE_delay': [1.0*ms],
              'synIE_delay_windowsize': [0.5*ms],
              'synII_delay': [1.0*ms],
              'synII_delay_windowsize': [0.5*ms],
              'synEI_delay': [0.5*ms],
              'synEI_delay_windowsize': [0.25*ms],

              # Neuronal parameters
              'external_mode': ['memnoise'],
              'mu_e': [0.0*mV],
              'mu_i': [0.0*mV],
              'sigma_e': [4.5*mV], # 6.1 (sub-crit - noise), 4.5 (reverb - rec)
              'sigma_i': [4.5*mV],
              'Vt_e': [-50. * mV],
              'Vt_i': [-50. * mV],
              'Vr_e': [-70. * mV],
              'Vr_i': [-70. * mV],
              'refractory_exc': ['2*ms'],
              'refractory_inh': ['1*ms'],

              'strong_mem_noise_active': [0],
              'strong_mem_noise_rate': [0.1*Hz],

              'cramer_noise_active': [0],
              'cramer_noise_rate': [3.0 * Hz],
              'cramer_noise_Kext': [1.0],
              'cramer_noise_N': [int(2.5* 1600)],

              # Synaptic parameters
              'syn_cond_mode': ['exp'],
              'syn_cond_mode_EI': ['exp'],
              'tau_e': [5*ms],
              'tau_i': [10*ms],

              'syn_noise': [1],
              'syn_noise_type': ['additive'],   # 'kesten'
              'syn_kesten_mu_epsilon_1': [-0.062/second],
              'syn_kesten_mu_eta': [0.0058/second],
              'syn_kesten_var_epsilon_1': [0.001/second],
              'syn_kesten_var_eta': [0.00006/second],
              'syn_kesten_inh': [0],            # If Kesten is used
              'syn_kesten_mu_epsilon_1_i': [-0.1/second],
              'syn_kesten_mu_eta_i': [0.006/second],
              'syn_kesten_var_epsilon_1_i': [0.001/second],
              'syn_kesten_var_eta_i': [0.00001/second],

              # Plasticity parameters
              # --- STDP ---
              'stdp_active': [1],
              'stdp_ee_mode': ['song'],
              'Aplus': [1.6253*stdp_eta],
              'Aminus': [-0.8127*stdp_eta],

              'istdp_active': [1],
              'istdp_type': ['dbexp'],      # or 'dbexp'. Jan told me he used the dbexp
              'iAplus': [1.6253*ifactor*stdp_eta],
              'iAminus': [-0.5*1.6253*ifactor*stdp_eta],    # This is for assymetric
              # 'LTD_a': [0.8*ifactor*stdp_eta*0.1],  # for symmetric (why like this)

              # --- Structural ---
              'strct_active': [1],
              'strct_mode': ['zero'],
              'strct_dt': [1000 * ms],
              'strct_c': a_min_insert,
              'a_insert': a_min_insert,
              'insert_P': [0.05],
              'p_inactivate': [0.1],
              'adjust_insertP': [1],
              'adjust_insertP_mode': ['constant_count'],
              'csample_dt': [1 * second],

              # --- Homeostatic ---
              'dt_synEE_scaling': [100 * ms],   # Normalization timestep (how often to apply it)
              'eta_scaling': [1.0],             # Strength of normalization (if set on 1, then its instantaneous per dt)

              'scl_active': [1],
              'scl_mode': ["scaling"],
              'scl_scaling_kappa': [4.5*Hz],    # Target firing rate
              'scl_scaling_eta': [0.0001], # 0.1         # Scaling "learning rate" (1/τ, where τ the effector)
              'scl_scaling_dt': [1 * second],   # Scaling timestep (how often to apply it)
              
              'iscl_active': [1],

              # --- General ---
              'tau_r': [15 * second],   # Timescale of activity sensor
              'amin': a_min_insert,
              'amax': [0.1662*2],
              'amin_i' : [0.005],
              'amax_i' : [0.320],
              'ATotalMax': [3.4*2.65],
              'sig_ATotalMax': [0.05],
              'iATotalMax': [3.4*ifactor],

              # Recording parameters
              'crs_crrs_rec': [0],      # Calc and record correlation
              'memtraces_rec': [1],
              'vttraces_rec': [1],
              'getraces_rec': [1],
              'gitraces_rec': [1],
              'nrec_GExc_stat': [400],
              'nrec_GInh_stat': [3],
              'GExc_stat_dt': [2 * ms],
              'GInh_stat_dt': [2 * ms],
              'synee_atraces_rec': [1],
              'synee_activetraces_rec': [0],
              'synee_Apretraces_rec': [0],
              'synee_Aposttraces_rec': [0],
              'n_synee_traces_rec': [1000],
              'synEE_stat_dt': [2 * ms],
              'synei_atraces_rec': [1],
              'synei_activetraces_rec': [0],
              'synei_Apretraces_rec': [0],
              'synei_Aposttraces_rec': [0],
              'n_synei_traces_rec': [1000],
              'synEI_stat_dt': [2 * ms],
              'synee_a_nrecpoints': [10],
              'synei_a_nrecpoints': [10],
              'synEEdynrec': [1],
              'synEIdynrec': [0],
              'syndynrec_dt': [1 * second],
              'syndynrec_npts': [10],
              'turnover_rec': [1],
              'spks_rec': [1],
              'T2_spks_rec': [0],
              'rates_rec': [1],
              'anormtar_rec': [1],
              'syn_scl_rec': [0],
              'syn_iscl_rec': [0],
              'scl_rec_T': [1 * second],
              'synEE_rec': [0],         # Record synaptic spikes
              'synEI_rec': [0],
              'stdp_rec_T': [1 * second],
              'population_binned_rec': [0],

              # Unused settings (could be deleted)
              'ip_active': [0],
              'h_IP_e': [25 * Hz],
              'h_IP_i': [-80 * Hz],  # disable
              'eta_IP': [0.01 * mV],

              'synEE_std_rec': [0],
              'std_active': [0],  # use default parameters
              'tau_std': [200*ms],

              'sra_active': [0],
              'Dgsra': [0.1]
}


name = 'hdf5_data'
explore_dict = n_list(input_dict)

if __name__ == "__main__":
    print(name)
