from brian2.units import ms, mV, second, Hz
from pypet import cartesian_product


input_dict = {'iscl_active' : [0],
              'crs_crrs_rec': [0],
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
              'turnover_rec': [0]}


name = 'test_standard_net'

explore_dict = cartesian_product(input_dict)

