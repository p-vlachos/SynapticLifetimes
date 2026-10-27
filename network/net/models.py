
# --- neurons ---
# conductance-based LIF neurons (used for both populations); the external input is
# either membrane noise (memnoise) or Poisson spike trains through the conductance gext
neuron_memnoise = '''
              dV/dt = (El-V + ge*(Ee-V) + gi*(Ei-V))/tau +  mu/tau + (sigma * xi) / (tau **.5) : volt (unless refractory)
              Vt : volt
              sigma: volt (constant)
              mu : volt (constant)
              '''

neuron_poisson = '''
              dV/dt = (El-V + (ge + gext)*(Ee-V) + gi*(Ei-V))/tau : volt (unless refractory)
              dgext/dt = -gext/tau_e : 1
              Vt : volt
              '''

# variables of excitatory neurons for the growth homeostasis: number of active input EE synapses
# and sum of p_distance of the inactive ones
neuron_exc_growth = '''
              n_active_in : 1
              pinactive_in : 1
              '''

# Gaussian distance factor of the insertion probability of EE synapses (the same function as
# utils.gaussian), computed from the positions of the two neurons on the (wrapped) sheet
p_distance_wrap = '''
            dist_x = abs(x_pre - x_post) : metre
            dist_y = abs(y_pre - y_post) : metre
            dist_wrapped = sqrt(((grid_size - dist_x)*int(dist_x > 0.5*grid_size) + dist_x*int(dist_x <= 0.5*grid_size))**2 + ((grid_size - dist_y)*int(dist_y > 0.5*grid_size) + dist_y*int(dist_y <= 0.5*grid_size))**2) : metre
            p_distance = 2/sqrt(2*pi*(sigma_dd/um)**2) * exp(-(dist_wrapped/um)**2/(2*(sigma_dd/um)**2)) : 1
            '''
p_distance_nowrap = '''
            p_distance = 2/sqrt(2*pi*(sigma_dd/um)**2) * exp(-(sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2)/um)**2/(2*(sigma_dd/um)**2)) : 1
            '''
p_distance_uniform = 'p_distance = 1 : 1'

# summed per postsynaptic neuron, for the growth homeostasis
synEE_growth_mod = '''
            n_active_in_post = 1.0*syn_active : 1 (summed)
            pinactive_in_post = p_distance*(1 - syn_active) : 1 (summed)
            '''

# variables of excitatory neurons for the normalization of their input weights
neuron_exc_normalization = '''
              AsumEE : 1
              AsumEI : 1
              ANormTar : 1
              iANormTar : 1
              min_ANormTar : 1
              max_ANormTar : 1
              '''

neuron_threshold = 'V > Vt'
neuron_reset_exc = 'V = Vr_e'
neuron_reset_inh = 'V = Vr_i'

# firing rate sensor, used by the homeostatic adaptation of the normalization target
condlif_rate_sensor = "dr/dt = -r/tau_r : Hz"
reset_rate_sensor = "r += 1/tau_r"

syn_cond_EE_exp = '''
                  dge /dt = -ge/tau_e : 1
                  '''

syn_cond_EI_exp = '''
                  dgi /dt = -gi/tau_i : 1
                  '''

# refer to https://brian2.readthedocs.io/en/stable/user/converting_from_integrated_form.html
syn_cond_EE_biexp = '''
                    dge/dt = (invpeakEE*xge-ge)/tau_e_rise : 1
                    dxge/dt = -xge/tau_e                   : 1
                    '''

syn_cond_EI_biexp = '''
                    dgi/dt = (invpeakEI*xgi-gi)/tau_i_rise : 1
                    dxgi/dt = -xgi/tau_i                   : 1
                    '''




synEE_static = 'a : 1'
# multiplicative weight noise, used for EE and EI synapses
syn_noise_vars = '''a : 1
                  syn_sigma : 1/second (shared)'''
# exact update of the (Stratonovich) multiplicative weight noise da/dt = a*sqrt(syn_sigma)*xi
# over an interval of synEE_mod_dt, followed by clipping to the weight bounds
syn_noise_update = '''a = a*exp(syn_noise_active*syn_active*sqrt(syn_sigma*synEE_mod_dt)*randn())
                      a = syn_active*clip(a, amin, amax)'''


synEE_mod = '''            
            syn_active : integer

            taupre : second (shared) 
            taupost : second (shared) 

            dApre  /dt = -Apre/taupre  : 1 (event-driven)
            dApost /dt = -Apost/taupost : 1 (event-driven)
            
            Aplus : 1 (shared)
            Aminus : 1 (shared)

            insert_P : 1 (shared) 
            p_inactivate : 1 (shared)
            stdp_active : integer (shared)
            syn_noise_active : integer (shared)

            scl_rec_start : second (shared)
            scl_rec_max   : second (shared)

            stdp_rec_start : second (shared)
            stdp_rec_max   : second (shared)
            

            amin : 1 (shared)
            amax : 1 (shared)
            '''

synEE_scl_mod = 'AsumEE_post = a : 1 (summed)'
synEI_scl_mod = 'AsumEI_post = a : 1 (summed)'
synEE_pre_exp   = '''
                  ge_post += syn_active*a
                  '''

synEE_pre_biexp = '''
                  xge_post += syn_active*a/norm_f_EE
                  '''
# STDP with all-to-all spike interactions: every spike adds to its trace
# (Song, Miller & Abbott 2000); the weight changes are applied in syn_pre_STDP / syn_post_STDP
synEE_pre_STDP = "Apre += syn_active*Aplus"


synEI_pre_exp   = '''
                  gi_post += syn_active*a
                  Apre += syn_active*Aplus
                  '''

synEI_pre_biexp = '''
                  xgi_post += syn_active*a/norm_f_EI
                  Apre += syn_active*Aplus
                  '''


syn_pre_STDP = '''
                 a = syn_active*clip(a+Apost*stdp_active, amin, amax)
                 '''


synEE_pre_rec = '''
                dummy = record_spk(t, i, j, a, Apre, Apost, syn_active, 0, stdp_rec_start, stdp_rec_max)
                '''

synEI_pre_rec = '''
                dummy = record_spk_EI(t, i, j, a, Apre, Apost, syn_active, 0, stdp_rec_start, stdp_rec_max)
                '''


syn_post = '''
           Apost += syn_active*Aminus
           '''

syn_post_STDP = '''
                a = syn_active*clip(a+Apre*stdp_active, amin, amax)
                '''

synEE_post_rec = '''
                 dummy = record_spk(t, i, j, a, Apre, Apost, syn_active, 1, stdp_rec_start, stdp_rec_max)
                 '''

synEI_post_rec = '''
                 dummy = record_spk_EI(t, i, j, a, Apre, Apost, syn_active, 1, stdp_rec_start, stdp_rec_max)
                 '''


synEE_scaling = '''
                a = syn_active*clip(syn_scale(a, ANormTar, AsumEE_post, eta_scaling, t, syn_active, scl_rec_start, scl_rec_max, i, j), amin, amax)
                '''

synEI_scaling = '''
                a = syn_active*clip(syn_EI_scale(a, iANormTar, AsumEI_post, eta_iscaling, t, syn_active, scl_rec_start, scl_rec_max, i, j), amin, amax)
                '''

synEE_target_scaling = 'ANormTar = clip(ANormTar + (scl_scaling_eta) * ((scl_scaling_kappa-r)/scl_scaling_kappa), min_ANormTar, max_ANormTar)'

turnover_rec_mod = '''
                   dummy = record_turnover(t, was_active_before, should_become_active, should_stay_active, syn_active, i, j)
                   '''

# zero mode
strct_mod = '''
            r_inact = rand()
            should_stay_active = int(a > strct_c) + int(a<= strct_c)*int(r_inact>p_inactivate)
            s = rand()
            should_become_active = int(s < insert_P*p_distance)
            was_active_before = syn_active
            syn_active = int(syn_active==1) * int(should_stay_active) \
                     + int(syn_active==0) * int(should_become_active)
            a = a*int(was_active_before==1)*int(syn_active==1) \
                + a_insert*int(was_active_before==0)*int(syn_active==1)
            '''

# 2D plane for distance-dependent connectivity
ddcon = '''
            x : metre 
            y : metre
        '''
