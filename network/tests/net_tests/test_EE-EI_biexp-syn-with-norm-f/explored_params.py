from brian2.units import ms, mV, second, Hz
from pypet import cartesian_product


input_dict = {'crs_crrs_rec': [0],
              'syn_scl_rec' : [0],
              'syn_iscl_rec' : [0],
              'synEE_rec': [0],
              'synEI_rec': [0],
              'syn_noise': [0],
              'synee_activetraces_rec' : [0],
              'synee_Apretraces_rec': [0],
              'synee_Aposttraces_rec': [0],
              'synee_a_nrecpoints': [1],
              'synei_a_nrecpoints': [1],
              'synEEdynrec': [1],
              'synEIdynrec': [1],
              'turnover_rec': [0],
              'syn_cond_mode': ['biexp'],
              'syn_cond_mode_EI': ['biexp'],
              'tau_e_rise': [3.5*ms],
              'tau_i_rise': [0.75*ms],
              'norm_f_EE' : [2.1],
              'norm_f_EI' : [1.0,2.1]}


name = 'test_standard_net'

explore_dict = cartesian_product(input_dict)

