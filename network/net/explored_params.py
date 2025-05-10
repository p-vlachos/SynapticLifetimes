from brian2.units import ms, mV, second, ksecond, Hz, um
from .param_tools import *
import numpy as np

sigv = 1. / second

input_dict = {'T1': [10 * second],
              'T2': [500 * second], #5000
              'pp_tcut': [100 * second], # 500
              'T3': [5 * second],
              'T4': [5 * second],
              'syn_scl_rec': [0],
              'syn_iscl_rec': [0],
              'scl_rec_T': [1 * second],
              'synEE_rec': [1],
              'synEI_rec': [0],
              'stdp_rec_T': [1 * second],
              'T5': [200 * second], #2000
              'crs_crrs_rec': [0],
              'dt': [0.1 * ms],
              'N_e': [1600],
              'N_i': [ 320],
              'syn_cond_mode': ['exp'],
              'syn_cond_mode_EI': ['exp'],
              'tau_e': [5*ms],
              'tau_i': [10*ms],
              'refractory_exc': ['2*ms'],
              'refractory_inh': ['1*ms'],
              'external_mode': ['memnoise'],
              'mu_e': [0.0*mV],
              'mu_i': [0.0*mV],
              'sigma_e': [6.1*mV], #6.1, 4.1, 4.25 is the last value Jan used
              'sigma_i': [6.1*mV],
            #   'PInp_mode' : ['indep'],
            #   'PInp_rate' : [8000*Hz],
            #   'PInp_inh_rate' : [6000*Hz],
              #           ??                              ->  2.26 Hz  ....
              'p_ee': [0.08],
              'p_ei': [0.24],
              'p_ie': [0.24],
              'p_ii': [0.24],

              'syn_delay_active': [1],
              'syn_dd_delay_active' : [1],
              'synEE_delay': [3.0*ms],
              'synEE_delay_windowsize': [1.5*ms], #1.5
              'synIE_delay': [1.0*ms],
              'synIE_delay_windowsize': [0.5*ms], #0.5
              'synII_delay': [1.0*ms],
              'synII_delay_windowsize': [0.5*ms], #0.5
              'synEI_delay': [0.5*ms],
              'synEI_delay_windowsize': [0.25*ms], #0.25

              'stdp_active': [1],
              'Aplus': [0.015],
              'Aminus': [-0.5*0.015],

              'istdp_active': [1],
              'istdp_type': ['dbexp'],
              'iAplus': [0.030],
              'iAminus': [-0.5*0.030],
              # 'LTD_a': [0.8*ifactor*stdp_eta*0.1],  # original 0.1

              'strct_active': [1],
              'strct_mode': ['zero'],
              'strct_dt': [1000 * ms],
              'strct_c': [0.04],
              'a_insert': [0.04],
              'insert_P': [0.05],
              'p_inactivate': [0.1],
              'adjust_insertP': [1],
              'adjust_insertP_mode': ['constant_count'],
              'csample_dt': [1 * second],

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

              'dt_synEE_scaling': [100 * ms],           # Normalization timestep (how often to apply it)
              'eta_scaling': [1.0],                     # Strength of normalization (if set on 1, then its instantaneous per dt)

              'scl_active': [1],
              'scl_mode': ["scaling"],
              'scl_scaling_kappa': [2.5*Hz],            # Target firing rate
              'scl_scaling_eta': [0.001], #XXX          # Scaling "learning rate" (1/τ)
              'scl_scaling_dt': [100 * ms],               # Scaling timestep (how often to apply it)
              'anormtar_rec': [1],
              'amin': [0.040],
              'amax': [0.320],
              'a_ee': [0.130], #XXX  16.8/(0.08*1599)
              'a_ie': [0.080],
              'a_ei': [0.100],      #XXX
              'a_ii': [0.190],
              'ATotalMax': [0.130*0.08*1599], # 16.6296
              'sig_ATotalMax': [0.05],               #XXX

              'iscl_active': [1],
              'iATotalMax': [0.100*0.24*320], # 7.68
              'amin_i' : [0.005],
              'amax_i' : [0.320],

              'Vt_e': [-50. * mV],
              'Vt_i': [-50. * mV],
              'Vr_e': [-70. * mV],
              'Vr_i': [-70. * mV],
              'ip_active': [0],
              'h_IP_e': [25 * Hz],
              'h_IP_i': [-80 * Hz],  # disable
              'eta_IP': [0.01 * mV],

              'syn_noise': [1],
              'syn_noise_type': ['kesten'],
              'syn_kesten_mu_eta':       [0.003/second],
              'syn_kesten_var_eta':      [1.0*0.00001/second],
              'syn_kesten_inh': [1],
              'syn_kesten_mu_eta_i':       [0.003/second],
              'syn_kesten_var_eta_i':      [1.0*0.00001/second],

              'strong_mem_noise_active': [0],
              'strong_mem_noise_rate': [0.1*Hz],

              'cramer_noise_active': [0],
              'cramer_noise_rate': [3.0 * Hz],
              'cramer_noise_Kext': [1.0],
              'cramer_noise_N': [int(2.5* 1600)],

              'synEE_std_rec': [0],
              'std_active': [0],  # use default parameters
              'tau_std': [200*ms],

              'sra_active': [0],
              'Dgsra': [0.1],

              'stdp_ee_mode': ['song'],

              'tau_r': [15 * second],

              'ddcon_active': [1],
              'half_width': [150*um],
              'grid_wrap': [1],

              'population_binned_rec': [1], 
              'random_seed': [8301]#, 3, 929]
}


name = 'hdf5_data'
explore_dict = n_list(input_dict)

if __name__ == "__main__":
    print(name)
