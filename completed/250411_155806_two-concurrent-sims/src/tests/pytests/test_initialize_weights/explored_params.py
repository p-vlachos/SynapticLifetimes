from brian2.units import ms, mV, second, Hz
from pypet import cartesian_product


input_dict = {'T1': [1*second],
              'T2': [0.*second],
              'T3': [0*second],
              'T4': [0*second],
              'syn_scl_rec' : [0],
              'syn_iscl_rec' : [0],
              'synEE_rec': [0],
              'synEI_rec': [0],
              'T5': [0*second],
              'crs_crrs_rec' : [0],
              'stdp_active': [0],
              'istdp_active': [0],
              'strct_active': [0],
              'istrct_active': [0],
              'synee_activetraces_rec' : [0],
              'synee_Apretraces_rec': [0],
              'synee_Aposttraces_rec': [0],
              'synei_atraces_rec' : [0],
              'synei_activetraces_rec' : [0],
              'synei_Apretraces_rec': [0],
              'synei_Aposttraces_rec': [0],
              'synee_a_nrecpoints': [10],
              'synei_a_nrecpoints': [10],
              'synEEdynrec': [0],
              'synEIdynrec': [0],
              'syndynrec_dt': [1*second],
              'syndynrec_npts': [10],
              'turnover_rec': [0],
              'strct_dt' : [1000*ms],
              'spks_rec' : [1],
              'adjust_insertP': [0],
              'adjust_EI_insertP': [0],
              'csample_dt' : [10*second],
              'scl_active': [0],
              'iscl_active': [0],
              'a_ee' : [0.005],
              'a_ie' : [0.005],
              'a_ei' : [0.005],
              'a_ii' : [0.005]
}

name = 'test_initialize_weights'

explore_dict = cartesian_product(input_dict)
