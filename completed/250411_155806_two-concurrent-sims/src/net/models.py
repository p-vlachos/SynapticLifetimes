
# condlif_poisson = '''
#               dV/dt = (El-V + (gfwd+ge)*(Ee-V) + gi*(Ei-V))/tau : volt
#               Vt : volt 
#               dge /dt = -ge/tau_e : 1
#               dgfwd /dt = -gfwd/tau_e : 1
#               dgi /dt = -gi/tau_i : 1

#               AsumEE : 1
#               AsumEI : 1

#               ANormTar : 1
#               iANormTar : 1
#               '''

condlif_poisson = '''
              dV/dt = (El-V + gsra*(Esra-V) + (ge + gext)*(Ee-V) + gi*(Ei-V))/tau : volt (unless refractory)
              dgext/dt = -gext/tau_e : 1

              AsumEE : 1
              AsumEI : 1

              ANormTar : 1
              iANormTar : 1
              min_ANormTar : 1
              max_ANormTar : 1
              '''


condlif_memnoise = '''
              dV/dt = (El-V + gsra*(Esra-V) + ge*(Ee-V) + gi*(Ei-V))/tau +  mu/tau + (sigma * xi) / (tau **.5) : volt (unless refractory)

              AsumEE : 1
              AsumEI : 1

              sigma: volt (constant)
              mu : volt (constant)

              ANormTar : 1
              iANormTar : 1
              min_ANormTar : 1
              max_ANormTar : 1
              '''

condlif_sra = 'dgsra/dt = -gsra/tau_sra : 1'
condlif_nosra = 'gsra : 1'

condlif_noIP = 'Vt : volt'
condlif_IP = '''
    dVt/dt = -IP_active*eta_IP*h_IP : volt
    h_IP : Hz
    IP_active : 1
'''
reset_IP = 'Vt += IP_active*eta_IP'
reset_sra = 'gsra += Dgsra'

condlif_triplet = "dr/dt = -r/tau_r : Hz"
reset_triplet = "r += 1/tau_r"

condlif_jedlicka = condlif_triplet
reset_jedlicka = reset_triplet

syn_cond_EE_exp = '''
                  dge /dt = -ge/tau_e : 1
                  '''

syn_cond_EI_exp = '''
                  dgi /dt = -gi/tau_i : 1
                  '''

syn_cond_EE_alpha = '''
                    dge /dt = (xge-ge)/tau_e : 1
                    dxge /dt = -xge/tau_e : 1
                    '''

# refer to https://brian2.readthedocs.io/en/stable/user/converting_from_integrated_form.html
syn_cond_EE_biexp = '''
                    dge/dt = (invpeakEE*xge-ge)/tau_e_rise : 1
                    dxge/dt = -xge/tau_e                   : 1
                    '''

syn_cond_EI_alpha = '''
                    dgi /dt = (xgi-gi)/tau_i : 1
                    dxgi /dt = -xgi/tau_i : 1
                    '''

syn_cond_EI_biexp = '''
                    dgi/dt = (invpeakEI*xgi-gi)/tau_i_rise : 1
                    dxgi/dt = -xgi/tau_i                   : 1
                    '''



# refractory period???

nrnEE_thrshld = 'V > Vt'

nrnEE_reset = 'V = Vr_e'

poisson_mod = 'PInp_a : 1'

# !--- add event-driven for efficiency ---!
# dApre  /dt = -Apre/taupre  : 1 (event-driven)
# dApost /dt = -Apost/taupost : 1 (event-driven)

synEE_static = 'a : 1'
synEE_noise_add   = '''da/dt = syn_noise_active*syn_active*syn_sigma**0.5*xi : 1
                       syn_sigma : 1/second (shared)'''

synEE_noise_mult  = '''da/dt = syn_noise_active*syn_active*a*syn_sigma**0.5*xi : 1
                       syn_sigma : 1/second (shared)'''
synEE_noise_kesten = '''da/dt = syn_noise_active*syn_active * ( (syn_kesten_mu_epsilon_1 * (syn_kesten_factor*a) + syn_kesten_mu_eta) + (syn_kesten_var_epsilon_1 * (syn_kesten_factor*a)**2 + syn_kesten_var_eta)**0.5 * xi_kesten) / syn_kesten_factor : 1
                        '''
synEI_noise_kesten = '''da/dt = syn_noise_active*syn_active * ( (syn_kesten_mu_epsilon_1_i * (syn_kesten_factor*a) + syn_kesten_mu_eta_i) + (syn_kesten_var_epsilon_1_i * (syn_kesten_factor*a)**2 + syn_kesten_var_eta_i)**0.5 * xi_kesten) / syn_kesten_factor : 1
                        '''

synEE_noise_decay = '''da/dt = syn_noise_active*(amin - a)/tau_adecay : 1'''
synEI_noise_decay = '''da/dt = syn_noise_active*(amin - a)/tau_adecay : 1'''


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
            
            p_distance : 1

            amin : 1 (shared)
            amax : 1 (shared)
            '''

synEE_scl_mod = 'AsumEE_post = a : 1 (summed)'
synEI_scl_mod = 'AsumEI_post = a : 1 (summed)'
synEE_scl_prop_mod = 'ANormTar_post = syn_active*ATotalMaxSingle : 1 (summed)'
synEI_scl_prop_mod = 'iANormTar_post = syn_active*iATotalMaxSingle : 1 (summed)'
synEE_nostd_mod = 'D : 1'
synEE_std_mod = 'dD/dt = (1 - D)/tau_std : 1 (event-driven)'
synEE_triplet_mod = 'dAslow/dt = -Aslow/tau_slow : 1 (event-driven)'


synEE_p_activate = '''
                   r_ee = rand()
                   syn_active = int(r_ee < p_ee)
                   a = syn_active*a
                   '''

synEE_pre_exp   = '''
                  ge_post += D*syn_active*a
                  '''

synEE_pre_alpha = '''
                  xge_post += D*syn_active*a/norm_f_EE
                  '''

synEE_pre_biexp = '''
                  xge_post += D*syn_active*a/norm_f_EE
                  '''
synEE_pre_STDP = "Apre = syn_active*Aplus"
synEE_pre_triplet = "Apre += syn_active*Aplus"
synEE_pre_jedlicka = "Apre = syn_active*Aplus/(r_post/jedlicka_kappa)"

synEE_pre_std = '''D *= std_d'''


synEI_pre_exp   = '''
                  gi_post += syn_active*a
                  Apre = syn_active*Aplus
                  '''

synEI_pre_alpha = '''
                  xgi_post += syn_active*a/norm_f_EI
                  Apre = syn_active*Aplus
                  '''

synEI_pre_biexp = '''
                  xgi_post += syn_active*a/norm_f_EI
                  Apre = syn_active*Aplus
                  '''


synEI_pre_sym_exp   = '''
                      gi_post += syn_active*a
                      Apre = syn_active*Aplus
                      a = a - stdp_active*LTD_a
                      '''

synEI_pre_sym_alpha = '''
                       xgi_post += syn_active*a/norm_f_EI
                       Apre = syn_active*Aplus
                       a = a - stdp_active*LTD_a
                       '''

synEI_pre_sym_biexp = '''
                       xgi_post += syn_active*a/norm_f_EI
                       Apre = syn_active*Aplus
                       a = a - stdp_active*LTD_a
                       '''


syn_pre_STDP = '''
                 a = syn_active*clip(a+Apost*stdp_active, amin, amax)
                 '''

syn_pre_STDP_triplet = '''
                        Aminus_t = (Aplus*taupre*tau_slow)/(taupost*triplet_kappa) * r_post**2
                        a = syn_active*clip(a - stdp_active*(Aminus_t * Apost), amin, amax)
                       '''


synEE_pre_rec = '''
                dummy = record_spk(t, i, j, a, Apre, Apost, syn_active, 0, stdp_rec_start, stdp_rec_max)
                '''

synEI_pre_rec = '''
                dummy = record_spk_EI(t, i, j, a, Apre, Apost, syn_active, 0, stdp_rec_start, stdp_rec_max)
                '''


syn_post = '''
           Apost = syn_active*Aminus
           '''
synEE_post_jedlicka = '''Apost = syn_active*Aminus*(r_post/jedlicka_kappa)'''

syn_post_triplet_before = ''' Apost += 1 '''
syn_post_triplet_after = ''' Aslow += 1 '''

synEI_post_sym = '''
                 Apost = syn_active*Aplus
                 '''


syn_post_STDP = '''
                a = syn_active*clip(a+Apre*stdp_active, amin, amax)
                '''

syn_post_triplet_STDP = '''
                        a = syn_active*clip(a+Apre*Aslow*stdp_active, amin, amax)
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

synEE_target_scaling_nonadaptive = 'ANormTar = clip(ANormTar + (scl_scaling_eta) * sign(scl_scaling_kappa-r), min_ANormTar, max_ANormTar)'
synEE_target_scaling_saturated = 'ANormTar = clip(ANormTar + clip((scl_scaling_eta) * ((scl_scaling_kappa-r)/scl_scaling_kappa), min_Ascaling, max_Ascaling), min_ANormTar, max_ANormTar)'
synEE_target_scaling = 'ANormTar = clip(ANormTar + (scl_scaling_eta) * ((scl_scaling_kappa-r)/scl_scaling_kappa), min_ANormTar, max_ANormTar)'

# rand() == uniform(0,1)
#strct_mod = ''

strct_mod_thrs = '''
                 r_ins = rand()
                 should_stay_active = int(a > prn_thrshld)
                 should_become_active = int(r_ins < insert_P)
                 was_active_before = syn_active
                 syn_active = int(syn_active==1) * int(should_stay_active) \
                    + int(syn_active==0) * int(should_become_active)
                 a = a*int(was_active_before==1)*int(syn_active==1) \
                    + a_insert*int(was_active_before==0)*int(syn_active==1)
                 '''

turnover_rec_mod = '''
                   dummy = record_turnover(t, was_active_before, should_become_active, should_stay_active, syn_active, i, j)
                   '''

turnoverEI_rec_mod = '''
                   dummy = record_turnover_EI(t, was_active_before, should_become_active, should_stay_active, syn_active, i, j)
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
