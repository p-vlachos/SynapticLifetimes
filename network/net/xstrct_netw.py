
import os, shutil, pickle, logging, neo

from . import models as mod, network_features
from .brian2objects.binnedactivitymonitor import BinnedActivityMonitor
from .utils import generate_connections, generate_full_connectivity, \
                   generate_N_connections, generate_dd_connectivity2

import numpy as np
import pypet

from brian2.units import ms,mV,second,metre

from brian2 import NeuronGroup, StateMonitor, SpikeMonitor, \
                   PoissonGroup, Synapses, set_device, device, Clock, \
                   defaultclock, prefs, Network, PopulationRateMonitor, seed

import elephant.utils
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import cch
import quantities as pq

from .run_routines import run_phase, run_T2_syndynrec

from .cpp_methods import syn_scale, syn_EI_scale, \
                         record_turnover, record_spk, record_spk_EI

def run_net(tr):

    # prefs.codegen.target = 'numpy'
    # prefs.codegen.target = 'cython'
    if tr.n_threads > 1:
        prefs.devices.cpp_standalone.openmp_threads = tr.n_threads

    set_device('cpp_standalone', directory='./builds/%.4d'%(tr.v_idx),
               build_on_run=False)


    # set brian 2 and numpy random seeds
    seed(tr.random_seed)
    np.random.seed(tr.random_seed+11)

    print("Started process with id ", str(tr.v_idx))

    T = tr.T1 + tr.T2 + tr.T3 + tr.T4 + tr.T5

    # variables which will be used by the equations, ``short_names`` converts PyPet's namespaced
    # variable names to simple variable names without namespace (ValueError if not unique)
    namespace = tr.netw.f_to_dict(short_names=True, fast_access=True)
    if tr.eta_iscaling < 0:
        namespace['eta_iscaling'] = tr.eta_scaling

    defaultclock.dt = tr.netw.sim.dt

    # collect all network components dependent on configuration (e.g. poisson vs. memnoise)
    # and add them to the Brian 2 network object later
    netw_objects = []

    if tr.external_mode=='memnoise':
        neuron_model = tr.condlif_memnoise
    elif tr.external_mode=='poisson':
        # raise NotImplementedError
        neuron_model = tr.condlif_poisson
    else:
        raise NotImplementedError

    neuronE_reset, neuronI_reset = tr.nrnEE_reset, 'V=Vr_i'

    if tr.scl_active:
        neuron_model += f"\n{tr.condlif_rate_sensor}"
        neuronE_reset += f"\n{tr.reset_rate_sensor}"

    if tr.syn_cond_mode=='exp':
        neuron_model += tr.syn_cond_EE_exp
        print("Using EE exp mode")
    elif tr.syn_cond_mode=='biexp':
        neuron_model += tr.syn_cond_EE_biexp
        namespace['invpeakEE'] = (tr.tau_e / tr.tau_e_rise) ** \
            (tr.tau_e_rise / (tr.tau_e - tr.tau_e_rise))
        print("Using EE biexp mode")
    else:
        raise ValueError(f"Invalid syn_cond_mode {tr.syn_cond_mode}")

    if tr.syn_cond_mode_EI=='exp':
        neuron_model += tr.syn_cond_EI_exp
        print("Using EI exp mode")
    elif tr.syn_cond_mode_EI=='biexp':
        neuron_model += tr.syn_cond_EI_biexp
        namespace['invpeakEI'] = (tr.tau_i / tr.tau_i_rise) ** \
            (tr.tau_i_rise / (tr.tau_i - tr.tau_i_rise))
        print("Using EI biexp mode")
    else:
        raise ValueError(f"Invalid syn_cond_mode_EI {tr.syn_cond_mode_EI}")

    # space is added to both inh and exc populations
    if tr.ddcon_active:
        neuron_model += tr.ddcon

    GExc = NeuronGroup(N=tr.N_e, model=neuron_model,
                       threshold=tr.nrnEE_thrshld,
                       reset=neuronE_reset, #method=tr.neuron_method,
                       name='GExc', namespace=namespace,
                       refractory=tr.refractory_exc)
    GInh = NeuronGroup(N=tr.N_i, model=neuron_model,
                       threshold ='V > Vt',
                       reset=neuronI_reset, #method=tr.neuron_method,
                       name='GInh', namespace=namespace,
                       refractory=tr.refractory_inh)

    if tr.external_mode=='memnoise':
        # GExc.mu, GInh.mu = [0.*mV] + (tr.N_e-1)*[tr.mu_e], tr.mu_i
        # GExc.sigma, GInh.sigma = [0.*mV] + (tr.N_e-1)*[tr.sigma_e], tr.sigma_i
        GExc.mu, GInh.mu = tr.mu_e, tr.mu_i
        GExc.sigma, GInh.sigma = tr.sigma_e, tr.sigma_i

    GExc.Vt, GInh.Vt = tr.Vt_e, tr.Vt_i
    GExc.V , GInh.V  = np.random.uniform(tr.Vr_e/mV, tr.Vt_e/mV,
                                         size=tr.N_e)*mV, \
                       np.random.uniform(tr.Vr_i/mV, tr.Vt_i/mV,
                                         size=tr.N_i)*mV

    if tr.scl_active:
        GExc.r = tr.scl_scaling_kappa
        GInh.r = tr.scl_scaling_kappa

    # ddcon init
    if tr.ddcon_active:
        GExc.x, GExc.y = np.random.uniform(high=tr.grid_size, size=tr.N_e)*metre, \
                         np.random.uniform(high=tr.grid_size, size=tr.N_e)*metre
        GInh.x, GInh.y = np.random.uniform(high=tr.grid_size, size=tr.N_i)*metre, \
                         np.random.uniform(high=tr.grid_size, size=tr.N_i)*metre,
        # save neuron positions for debug
        tr.f_add_result('GExc_x', np.array(GExc.x))
        tr.f_add_result('GExc_y', np.array(GExc.y))
        tr.f_add_result('GInh_x', np.array(GInh.x))
        tr.f_add_result('GInh_y', np.array(GInh.y))

    netw_objects.extend([GExc,GInh])


    if tr.external_mode=='poisson':
    
        if tr.PInp_mode == 'pool':
            PInp = PoissonGroup(tr.NPInp, rates=tr.PInp_rate,
                                namespace=namespace, name='poissongroup_exc')
            
            sPN = Synapses(target=GExc, source=PInp, model=tr.poisson_mod,
                           on_pre='gext_post += a_EPoi',
                           namespace=namespace, name='synPInpExc')
            
            sPN_src, sPN_tar = generate_N_connections(N_tar=tr.N_e,
                                                      N_src=tr.NPInp,
                                                      N=tr.NPInp_1n)

        elif tr.PInp_mode == 'indep':
            PInp = PoissonGroup(tr.N_e, rates=tr.PInp_rate,
                                namespace=namespace)
            sPN = Synapses(target=GExc, source=PInp, model=tr.poisson_mod,
                           on_pre='gext_post += a_EPoi',
                           namespace=namespace, name='synPInp_inhInh')
            sPN_src, sPN_tar = range(tr.N_e), range(tr.N_e)

        sPN.connect(i=sPN_src, j=sPN_tar)

        if tr.PInp_mode == 'pool':

            PInp_inh = PoissonGroup(tr.NPInp_inh, rates=tr.PInp_inh_rate,
                                    namespace=namespace, name='poissongroup_inh')
            
            sPNInh = Synapses(target=GInh, source=PInp_inh,
                              model=tr.poisson_mod,
                              on_pre='gext_post += a_EPoi',
                              namespace=namespace)
            
            sPNInh_src, sPNInh_tar = generate_N_connections(N_tar=tr.N_i,
                                                            N_src=tr.NPInp_inh,
                                                            N=tr.NPInp_inh_1n)

        elif tr.PInp_mode == 'indep':

            PInp_inh = PoissonGroup(tr.N_i, rates=tr.PInp_inh_rate,
                                    namespace=namespace)

            sPNInh = Synapses(target=GInh, source=PInp_inh,
                              model=tr.poisson_mod,
                              on_pre='gext_post += a_EPoi',
                              namespace=namespace)

            sPNInh_src, sPNInh_tar = range(tr.N_i), range(tr.N_i)

        sPNInh.connect(i=sPNInh_src, j=sPNInh_tar)

        netw_objects.extend([PInp, sPN, PInp_inh, sPNInh])
    

        
    if tr.syn_noise:

        if tr.syn_noise_type=='additive':
            synEE_mod = '''%s 
                           %s''' %(tr.synEE_noise_add, tr.synEE_mod)

            synEI_mod = '''%s 
                           %s''' %(tr.synEE_noise_add, tr.synEE_mod)

        elif tr.syn_noise_type=='multiplicative':
            synEE_mod = '''%s 
                           %s''' %(tr.synEE_noise_mult, tr.synEE_mod)

            synEI_mod = '''%s 
                           %s''' %(tr.synEE_noise_mult, tr.synEE_mod)

        else:
            raise ValueError(f"Invalid syn_noise_type {tr.syn_noise_type}")
    else:
        synEE_mod = '''%s 
                       %s''' %(tr.synEE_static, tr.synEE_mod)

        synEI_mod = '''%s 
                       %s''' %(tr.synEE_static, tr.synEE_mod)

    if tr.scl_active:
        synEE_mod = '''%s
                       %s''' %(synEE_mod, tr.synEE_scl_mod)


    if tr.iscl_active:
        synEI_mod = '''%s
                       %s''' %(synEI_mod, tr.synEI_scl_mod)
        
        
    if tr.syn_cond_mode=='exp':
        synEE_pre_mod = mod.synEE_pre_exp
    elif tr.syn_cond_mode=='biexp':
        synEE_pre_mod = mod.synEE_pre_biexp
    else:
        raise Exception("synaptic conductance mode is invalid")


    synEE_post_mod = ""

    if tr.stdp_active:
        synEE_post_mod = mod.syn_post

        synEE_pre_mod = '''%s 
                            %s
                            %s''' % (synEE_pre_mod, tr.synEE_pre_STDP, mod.syn_pre_STDP)
        synEE_post_mod = '''%s 
                            %s''' % (synEE_post_mod, mod.syn_post_STDP)

    if tr.synEE_rec:
        synEE_pre_mod  = '''%s 
                            %s''' %(synEE_pre_mod, mod.synEE_pre_rec)
        synEE_post_mod = '''%s 
                            %s''' %(synEE_post_mod, mod.synEE_post_rec)

    # E<-E advanced synapse model
    SynEE = Synapses(target=GExc, source=GExc, model=synEE_mod,
                     on_pre=synEE_pre_mod, on_post=synEE_post_mod,
                     namespace=namespace, dt=tr.synEE_mod_dt)


    

    if tr.istdp_active:

        if tr.syn_cond_mode_EI=='exp':
            EI_pre_mod = mod.synEI_pre_exp
        elif tr.syn_cond_mode_EI=='biexp':
            EI_pre_mod = mod.synEI_pre_biexp

        synEI_pre_mod  = '''%s 
                            %s''' %(EI_pre_mod, mod.syn_pre_STDP)
        synEI_post_mod = '''%s 
                            %s''' %(mod.syn_post, mod.syn_post_STDP)

    if tr.istdp_active and tr.synEI_rec:

        synEI_pre_mod  = '''%s 
                            %s''' %(synEI_pre_mod, mod.synEI_pre_rec)
        synEI_post_mod = '''%s 
                            %s''' %(synEI_post_mod, mod.synEI_post_rec)
            
    if tr.istdp_active:        
        SynEI = Synapses(target=GExc, source=GInh, model=synEI_mod,
                         on_pre=synEI_pre_mod, on_post=synEI_post_mod,
                         namespace=namespace, dt=tr.synEE_mod_dt)
    else:
        model = '''a : 1
                   syn_active : 1'''
        SynEI = Synapses(target=GExc, source=GInh, model=model,
                         on_pre='gi_post += a',
                         namespace=namespace)

    #other simple
    conductance_prefix = "" if tr.syn_cond_mode == "exp" else "x"
    SynIE = Synapses(target=GInh, source=GExc, on_pre=f'{conductance_prefix}ge_post += a_ie', namespace=namespace)
    SynII = Synapses(target=GInh, source=GInh, on_pre=f'{conductance_prefix}gi_post += a_ii', namespace=namespace)

    # all possible EE synapses exist, structural plasticity (de)activates them
    sEE_src, sEE_tar = generate_full_connectivity(tr.N_e, same=True)

    SynEE.connect(i=sEE_src, j=sEE_tar)
    SynEE.syn_active = 0
    SynEE.taupre, SynEE.taupost = tr.taupre, tr.taupost
    SynEE.Aplus = tr.Aplus
    SynEE.Aminus = tr.Aminus

    if tr.ddcon_active:
        sEI_src, sEI_tar, _, dist_EI = generate_dd_connectivity2(np.array(GExc.x), np.array(GExc.y),
                                                        np.array(GInh.x), np.array(GInh.y),
                                                        tr.half_width, tr.grid_size, tr.grid_wrap,
                                                        same=False, sparseness=tr.p_ei)
        SynEI.connect(i=sEI_src, j=sEI_tar)
        if tr.istdp_active:
            SynEI.Aplus = tr.Aplus if tr.iAplus < 0 else tr.iAplus
            SynEI.Aminus = tr.Aminus if tr.iAminus < 0 else tr.iAminus
    else:
        sEI_src, sEI_tar, _, _ = generate_connections(tr.N_e, tr.N_i, tr.p_ei)
        SynEI.connect(i=sEI_src, j=sEI_tar)

    if tr.istdp_active:        
        SynEI.taupre, SynEI.taupost = tr.taupre_EI, tr.taupost_EI

    sIE_src, sIE_tar, _, dist_IE = generate_dd_connectivity2(np.array(GInh.x), np.array(GInh.y),
                                                    np.array(GExc.x), np.array(GExc.y),
                                                    tr.half_width, tr.grid_size, tr.grid_wrap,
                                                    same=False, sparseness=tr.p_ie) if tr.ddcon_active \
        else generate_connections(tr.N_i, tr.N_e, tr.p_ie)
    sII_src, sII_tar, _, dist_II = generate_dd_connectivity2(np.array(GInh.x), np.array(GInh.y),
                                                    np.array(GInh.x), np.array(GInh.y),
                                                    tr.half_width,  tr.grid_size, tr.grid_wrap,
                                                    sparseness=tr.p_ii) if tr.ddcon_active \
        else generate_connections(tr.N_i, tr.N_i, tr.p_ii, same=True)

    SynIE.connect(i=sIE_src, j=sIE_tar)
    SynII.connect(i=sII_src, j=sII_tar)

    tr.f_add_result('sEE_src', sEE_src)
    tr.f_add_result('sEE_tar', sEE_tar)
    tr.f_add_result('sIE_src', sIE_src)
    tr.f_add_result('sIE_tar', sIE_tar)
    tr.f_add_result('sEI_src', sEI_src)
    tr.f_add_result('sEI_tar', sEI_tar)
    tr.f_add_result('sII_src', sII_src)
    tr.f_add_result('sII_tar', sII_tar)


    if tr.syn_noise:
        SynEE.syn_sigma = tr.syn_sigma
        SynEE.run_regularly('a = syn_active*clip(a,amin,amax)', when='after_groups',
                            name='SynEE_noise_clipper')

    if tr.syn_noise and tr.istdp_active:
        SynEI.syn_sigma = tr.syn_sigma
        SynEI.run_regularly('a = syn_active*clip(a,amin_i,amax_i)', when='after_groups',
                            name='SynEI_noise_clipper') 

    SynEE.insert_P = tr.insert_P
    SynEE.p_inactivate = tr.p_inactivate
    SynEE.stdp_active = 1
    SynEE.syn_noise_active = 1
    print('Setting maximum EE weight threshold to ', tr.amax)

    if tr.stdp_active:
        SynEE.amax = tr.amax
        SynEE.amin = tr.amin

    if tr.istdp_active:
        SynEI.stdp_active=1
        SynEI.amin = tr.amin_i if tr.amin_i >= 0 else tr.amin
        SynEI.amax = tr.amax_i if tr.amax_i >= 0 else tr.amax
        # EI synapses have the noise equations, but the noise is switched off
        SynEI.syn_noise_active = 0

    # initially active EE synapses
    if tr.ddcon_active:
        sEE_src_dd, sEE_tar_dd, sEE_p, dist_EE = generate_dd_connectivity2(np.array(GExc.x), np.array(GExc.y),
                                                                  np.array(GExc.x), np.array(GExc.y),
                                                                  tr.half_width, tr.grid_size, tr.grid_wrap,
                                                                  sparseness=tr.p_ee)
        # sEE_p is the full N_e x N_e matrix (incl. the diagonal), pick
        # the entries of the synapses in SynEE's order
        SynEE.p_distance = sEE_p[sEE_src, sEE_tar]  # TODO record this in trajectory

        EEactive = np.zeros(shape=(tr.N_e, tr.N_e))
        EEactive[sEE_src_dd, sEE_tar_dd] = 1
        np.fill_diagonal(EEactive, np.nan)
        syn_EE_active_init = EEactive[~np.isnan(EEactive)].flatten()
        SynEE.syn_active = syn_EE_active_init

        EEa = np.zeros(shape=EEactive.shape)
        EEa[sEE_src_dd, sEE_tar_dd] = tr.a_ee
        syn_EE_weights_init = EEa[~np.isnan(EEactive)].flatten()
        SynEE.a = syn_EE_weights_init
    else:
        # make randomly chosen synapses active at beginning
        rs = np.random.uniform(size=len(sEE_src))
        syn_EE_active_init = (rs < tr.p_ee).astype('int')
        SynEE.syn_active, SynEE.a = syn_EE_active_init, syn_EE_active_init * tr.a_ee
        SynEE.p_distance = 1

    # all EI synapses are active
    SynEI.syn_active, SynEI.a = 1, tr.a_ei

    if tr.syn_delay_active:

        if tr.syn_dd_delay_active:
            ee_delays = network_features.synapse_dd_delays(dist_EE, SynEE)
            ei_delays = network_features.synapse_dd_delays(dist_EI, SynEI)
            ii_delays = network_features.synapse_dd_delays(dist_II, SynII)
            ie_delays = network_features.synapse_dd_delays(dist_IE, SynIE)
        else:
            shapeEE, shapeEI, shapeII, shapeIE = syn_EE_active_init.shape, len(sEI_src), len(sII_src), len(sIE_src)
            ee_delays = network_features.synapse_delays(tr.synEE_delay, tr.synEE_delay_windowsize, SynEE, shapeEE)
            ei_delays = network_features.synapse_delays(tr.synEI_delay, tr.synEI_delay_windowsize, SynEI, shapeEI)
            ii_delays = network_features.synapse_delays(tr.synII_delay, tr.synII_delay_windowsize, SynII, shapeII)
            ie_delays = network_features.synapse_delays(tr.synIE_delay, tr.synIE_delay_windowsize, SynIE, shapeIE)

        tr.f_add_result("sEE_delays", ee_delays)
        tr.f_add_result("sEI_delays", ei_delays)
        tr.f_add_result("sII_delays", ii_delays)
        tr.f_add_result("sIE_delays", ie_delays)

    # recording of stdp in T4
    SynEE.stdp_rec_start = tr.T1+tr.T2+tr.T3
    SynEE.stdp_rec_max = tr.T1+tr.T2+tr.T3 + tr.stdp_rec_T

    if tr.istdp_active:
        SynEI.stdp_rec_start = tr.T1+tr.T2+tr.T3
        SynEI.stdp_rec_max = tr.T1+tr.T2+tr.T3 + tr.stdp_rec_T

  
       
    # synaptic scaling
    if tr.netw.config.scl_active:

        if tr.syn_scl_rec:
            SynEE.scl_rec_start = tr.T1+tr.T2+tr.T3
            SynEE.scl_rec_max = tr.T1+tr.T2+tr.T3 + tr.scl_rec_T
        else:
            SynEE.scl_rec_start = T+10*second
            SynEE.scl_rec_max = T

        GExc.min_ANormTar = tr.amin*tr.p_ee*tr.N_e
        GExc.max_ANormTar = tr.amax*tr.p_ee*tr.N_e
        if tr.sig_ATotalMax==0.:
            GExc.ANormTar = tr.ATotalMax
        else:
            GExc.ANormTar = np.random.normal(loc=tr.ATotalMax,
                                             scale=tr.sig_ATotalMax,
                                             size=tr.N_e)
        # homeostatic adaptation of the normalization target
        GExc.run_regularly(
            mod.synEE_target_scaling,
            dt=tr.scl_scaling_dt,
            when='end',
            name='synEE_target_scaling'
        )

        SynEE.summed_updaters['AsumEE_post']._clock = Clock(dt=tr.dt_synEE_scaling)
        synee_scaling = SynEE.run_regularly(tr.synEE_scaling,
                                            dt=tr.dt_synEE_scaling,
                                            when='end',
                                            name='synEE_scaling')

    if tr.istdp_active and tr.netw.config.iscl_active:

        if tr.syn_iscl_rec:
            SynEI.scl_rec_start = tr.T1+tr.T2+tr.T3
            SynEI.scl_rec_max = tr.T1+tr.T2+tr.T3 + tr.scl_rec_T
        else:
            SynEI.scl_rec_start = T+10*second
            SynEI.scl_rec_max = T

        if tr.sig_iATotalMax==0.:
            GExc.iANormTar = tr.iATotalMax
        else:
            GExc.iANormTar = np.random.normal(loc=tr.iATotalMax,
                                              scale=tr.sig_iATotalMax,
                                              size=tr.N_e)

        SynEI.summed_updaters['AsumEI_post']._clock = Clock(
            dt=tr.dt_synEE_scaling)

        synei_scaling = SynEI.run_regularly(tr.synEI_scaling,
                                            dt=tr.dt_synEE_scaling,
                                            when='end',
                                            name='synEI_scaling')


    # --- Structural plasticity ---
    if tr.netw.config.strct_active:
        if tr.turnover_rec:
            strct_mod  = '''%s 
                            %s''' %(tr.strct_mod, tr.turnover_rec_mod)
        else:
            strct_mod = tr.strct_mod

        strctplst = SynEE.run_regularly(strct_mod, dt=tr.strct_dt,
                                        when='start', name='strct_plst_zero')

    netw_objects.extend([SynEE, SynEI, SynIE, SynII])


    if tr.adjust_insertP:

        # keep track of the number of active synapses
        # this is more of a hack, where we have one neuron which
        # calculate the ratio of active synapses c of the total count NSyn
        sum_target = NeuronGroup(1, '''
            c : 1 (shared)
            pinactive : 1 (shared)
        ''', dt=tr.csample_dt)

        sum_model = '''NSyn : 1 (constant)
                       c_post = (1.0*syn_active_pre)/NSyn : 1 (summed)
                       pinactive_post = p_distance_pre*int(syn_active_pre == 0) : 1 (summed)
                    '''
        sum_connection = Synapses(target=sum_target, source=SynEE,
                                  model=sum_model, dt=tr.csample_dt,
                                  name='get_active_synapse_count')
        sum_connection.connect()
        sum_connection.NSyn = len(sEE_src)  # maximum number of EE synapses

        # homeostatically adjust growth rate
        # a similar hack as above, but here we update SynEE's insert_P
        # depending on the ratio of active synapses
        growth_updater = Synapses(sum_target, SynEE, model="""
                synapse_target_count : 1 (shared)
                synapse_max_count : 1 (shared)
                synapse_current_count : 1 (shared)
                synapse_needed_count : 1 (shared)
                synapse_inactive_count : 1 (shared)
        """)
        if tr.adjust_insertP_mode == "adjust_rate":
            growth_updater.run_regularly('insert_P_post *= 0.1/c_pre',
                                         when='after_groups', dt=tr.csample_dt,
                                         name='update_insP')
        elif tr.adjust_insertP_mode == "constant_count":
            growth_updater.run_regularly('''
                synapse_current_count = int(c_pre*synapse_max_count)
                synapse_needed_count = synapse_target_count - synapse_current_count
                synapse_inactive_count = synapse_max_count - synapse_current_count
                insert_P_post = (1/pinactive_pre)*synapse_needed_count
                ''',
                when='after_groups', dt=tr.csample_dt,
                name='update_insP')
        else:
            raise Exception(f"bad value '{tr.adjust_insertP_mode}' for 'adjust_insertP_mode'")
        growth_updater.connect(j='0')  # SynEE acts as one single target neuron
        growth_updater.synapse_target_count = np.sum(syn_EE_active_init)
        growth_updater.synapse_max_count = tr.N_e * (tr.N_e - 1)
        netw_objects.extend([sum_target, sum_connection, growth_updater])


    ##########################################################################
    ############### -------------- Recording ------------------ ##############
    ##########################################################################

    GExc_recvars = []
    if tr.memtraces_rec:
        GExc_recvars.append('V')
    if tr.vttraces_rec:
        GExc_recvars.append('Vt')
    if tr.getraces_rec:
        GExc_recvars.append('ge')
    if tr.gitraces_rec:
        GExc_recvars.append('gi')
    if tr.gfwdtraces_rec and tr.external_mode=='poisson':
        GExc_recvars.append('gext')
    if tr.anormtar_rec:
        if tr.scl_active == 1:
            GExc_recvars.append('ANormTar')
        if tr.iscl_active == 1:
            GExc_recvars.append('iANormTar')

    GInh_recvars = GExc_recvars
    
    GExc_stat = StateMonitor(GExc, GExc_recvars,
                             record=list(range(tr.nrec_GExc_stat)),
                             dt=tr.GExc_stat_dt)
    GInh_stat = StateMonitor(GInh, GInh_recvars,
                             record=list(range(tr.nrec_GInh_stat)),
                             dt=tr.GInh_stat_dt)
    
    # SynEE stat
    SynEE_recvars = []
    if tr.synee_atraces_rec:
        SynEE_recvars.append('a')
    if tr.synee_activetraces_rec:
        SynEE_recvars.append('syn_active')
    if tr.synee_Apretraces_rec:
        SynEE_recvars.append('Apre')
    if tr.synee_Aposttraces_rec:
        SynEE_recvars.append('Apost')

    SynEE_stat = StateMonitor(SynEE, SynEE_recvars,
                              record=range(tr.n_synee_traces_rec),
                              when='end', dt=tr.synEE_stat_dt)

    if tr.istdp_active:
        # SynEI stat
        SynEI_recvars = []
        if tr.synei_atraces_rec:
            SynEI_recvars.append('a')
        if tr.synei_activetraces_rec:
            SynEI_recvars.append('syn_active')
        if tr.synei_Apretraces_rec:
            SynEI_recvars.append('Apre')
        if tr.synei_Aposttraces_rec:
            SynEI_recvars.append('Apost')

        SynEI_stat = StateMonitor(SynEI, SynEI_recvars,
                                  record=range(tr.n_synei_traces_rec),
                                  when='end', dt=tr.synEI_stat_dt)
        netw_objects.append(SynEI_stat)
        

    GExc_spks = SpikeMonitor(GExc)
    GInh_spks = SpikeMonitor(GInh)

    GExc_rate = PopulationRateMonitor(GExc)
    GInh_rate = PopulationRateMonitor(GInh)

    if tr.population_binned_rec:
        GExc_binned = BinnedActivityMonitor(GExc, dtype=np.uint16)
        GInh_binned = BinnedActivityMonitor(GInh, dtype=np.uint16)
        netw_objects.extend([GExc_binned, GInh_binned])

    if tr.external_mode=='poisson':
        PInp_spks = SpikeMonitor(PInp)
        PInp_rate = PopulationRateMonitor(PInp)
        netw_objects.extend([PInp_spks,PInp_rate])


    if tr.synee_a_nrecpoints==0 or tr.sim.T2==0*second:
        SynEE_a_dt = 2*(tr.T1+tr.T2+tr.T3+tr.T4+tr.T5)
    else:
        SynEE_a_dt = tr.sim.T2/tr.synee_a_nrecpoints

        # make sure that choice of SynEE_a_dt does lead
        # to execessively many recordings - this can
        # happen if t1 >> t2.
        estm_nrecs = int(T/SynEE_a_dt)
        if estm_nrecs > 3*tr.synee_a_nrecpoints:
            print('''Estimated number of EE weight recordings (%d)
            exceeds desired number (%d), increasing 
            SynEE_a_dt''' % (estm_nrecs, tr.synee_a_nrecpoints))

            SynEE_a_dt = T/tr.synee_a_nrecpoints
        
    record_range_ee = range(len(sEE_src))

    SynEE_a = StateMonitor(SynEE, ['a','syn_active'],
                           record=record_range_ee,
                           dt=SynEE_a_dt,
                           when='end', order=100)

    record_range = range(len(sEI_src))

    if tr.synei_a_nrecpoints>0 and tr.sim.T2>0*second:
        SynEI_a_dt = tr.sim.T2/tr.synei_a_nrecpoints

        estm_nrecs = int(T/SynEI_a_dt)
        if estm_nrecs > 3*tr.synei_a_nrecpoints:
            print('''Estimated number of EI weight recordings
            (%d) exceeds desired number (%d), increasing 
            SynEI_a_dt''' % (estm_nrecs, tr.synei_a_nrecpoints))

            SynEI_a_dt = T/tr.synei_a_nrecpoints
  
        SynEI_a = StateMonitor(SynEI, ['a','syn_active'],
                               record=record_range,
                               dt=SynEI_a_dt,
                               when='end', order=100)

        netw_objects.append(SynEI_a)


    netw_objects.extend([GExc_stat, GInh_stat,
                         SynEE_stat, SynEE_a, 
                         GExc_spks, GInh_spks,
                         GExc_rate, GInh_rate])

    if (tr.synEEdynrec and
        (2*tr.syndynrec_npts*tr.syndynrec_dt < tr.sim.T2) ):
        SynEE_dynrec = StateMonitor(SynEE, ['a'],
                                    record=record_range_ee,
                                    dt=tr.syndynrec_dt,
                                    name='SynEE_dynrec',
                                    when='end', order=100)
        SynEE_dynrec.active=False
        netw_objects.extend([SynEE_dynrec])

            
    if (tr.synEIdynrec and
        (2*tr.syndynrec_npts*tr.syndynrec_dt < tr.sim.T2) ):
        SynEI_dynrec = StateMonitor(SynEI, ['a'],
                                    record=record_range,
                                    dt=tr.syndynrec_dt,
                                    name='SynEI_dynrec',
                                    when='end', order=100)
        SynEI_dynrec.active=False
        netw_objects.extend([SynEI_dynrec])  

    net = Network(*netw_objects)

    
    def set_active(*argv):
        for net_object in argv:
            net_object.active=True

    def set_inactive(*argv):
        for net_object in argv:
            net_object.active=False

    ##########################################################################
    ############### ---------- Simulation periods ------------- ##############
    ##########################################################################
    
    # --------- T1 ---------
    # initial recording period, all recorders active

    T1T3_recorders = [GExc_spks, GInh_spks, 
                      SynEE_stat, 
                      GExc_stat, GInh_stat,
                      GExc_rate, GInh_rate]

    if tr.istdp_active:
        T1T3_recorders.append(SynEI_stat)
    

    set_active(*T1T3_recorders)

    if tr.external_mode=='poisson':
        set_active(PInp_spks, PInp_rate)
       
    run_phase(net, tr.sim.T1, 'T1')


    # --------- T2 ---------
    # main simulation period, only active recordings are:
    #   1) turnover 2) C_stat 3) SynEE_a

    set_inactive(*T1T3_recorders)

    if tr.T2_spks_rec:
        set_active(GExc_spks, GInh_spks)
    
    if tr.external_mode=='poisson':
        set_inactive(PInp_spks, PInp_rate)

    run_T2_syndynrec(net, tr)


    # --------- T3 ---------
    # second recording period, all recorders active

    set_active(*T1T3_recorders)
    
    if tr.external_mode=='poisson':
        set_active(PInp_spks, PInp_rate)
    
    run_phase(net, tr.sim.T3, 'T3')
    
    # --------- T4 ---------
    # record STDP and scaling weight changes to file through the cpp models
    
    set_inactive(*T1T3_recorders)

    if tr.external_mode=='poisson':
        set_inactive(PInp_spks, PInp_rate)

    run_phase(net, tr.sim.T4, 'T4')

    # --------- T5 ---------
    # freeze network and record Exc spikes for cross correlations

    if tr.scl_active:
        synee_scaling.active=False
    if tr.istdp_active and tr.netw.config.iscl_active:
        synei_scaling.active=False
    if tr.strct_active:
        strctplst.active=False
    SynEE.stdp_active=0
    if tr.istdp_active:
        SynEI.stdp_active=0
    SynEE.syn_noise_active = 0
    if tr.istdp_active:
        # otherwise we use a simplified model and don't have this parameter
        SynEI.syn_noise_active = 0

    set_active(GExc_rate, GInh_rate)
    set_active(GExc_spks, GInh_spks)

    run_phase(net, tr.sim.T5, 'T5')
    
    SynEE_a.record_single_timestep()
    if tr.synei_a_nrecpoints>0 and tr.sim.T2 > 0.*second:
        SynEI_a.record_single_timestep()

    ##########################################################################
    ############## -------------- Build & Run ------------------ #############
    ##########################################################################
    
    build_directory = 'builds/%.4d'%(tr.v_idx)

    # Copy C++ files over to build directory
    src_dir = os.path.dirname(os.path.realpath(__file__))
    os.makedirs(build_directory, exist_ok=True)
    shutil.copy(f"{src_dir}/output_files.cpp", build_directory)
    shutil.copy(f"{src_dir}/output_files.h", build_directory)

    device.build(directory=build_directory, clean=True,
                 compile=True, run=True, debug=False,
                 additional_source_files=["output_files.cpp"])

    # -----------------------------------------

    print("start dumping")

    # save monitors as raws in build directory
    raw_dir = 'builds/%.4d/raw/'%(tr.v_idx)
    
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    with open(raw_dir+'namespace.p','wb') as pfile:
        pickle.dump(namespace,pfile)   

    with open(raw_dir+'gexc_stat.p','wb') as pfile:
        pickle.dump(GExc_stat.get_states(),pfile)   
    with open(raw_dir+'ginh_stat.p','wb') as pfile:
        pickle.dump(GInh_stat.get_states(),pfile)   
        
    with open(raw_dir+'synee_stat.p','wb') as pfile:
        pickle.dump(SynEE_stat.get_states(),pfile)

    if tr.istdp_active:
        with open(raw_dir+'synei_stat.p','wb') as pfile:
            pickle.dump(SynEI_stat.get_states(),pfile)

    if ( (tr.synEEdynrec or tr.synEIdynrec) and
         (2*tr.syndynrec_npts*tr.syndynrec_dt < tr.sim.T2) ):

        if tr.synEEdynrec:
            with open(raw_dir+'syneedynrec.p','wb') as pfile:
                pickle.dump(SynEE_dynrec.get_states(),
                            pfile)
        if tr.synEIdynrec:
            with open(raw_dir+'syneidynrec.p','wb') as pfile:
                pickle.dump(SynEI_dynrec.get_states(),
                            pfile)            
        
    with open(raw_dir+'synee_a.p','wb') as pfile:
        SynEE_a_states = SynEE_a.get_states()
        if tr.crs_crrs_rec:
            SynEE_a_states['i'] = list(SynEE.i)
            SynEE_a_states['j'] = list(SynEE.j)
        pickle.dump(SynEE_a_states,pfile)

    if tr.synei_a_nrecpoints>0 and tr.sim.T2 > 0.*second:
        with open(raw_dir+'synei_a.p','wb') as pfile:
            SynEI_a_states = SynEI_a.get_states()
            if tr.crs_crrs_rec:
                SynEI_a_states['i'] = list(SynEI.i)
                SynEI_a_states['j'] = list(SynEI.j)
            pickle.dump(SynEI_a_states,pfile)
        

    with open(raw_dir+'gexc_spks.p','wb') as pfile:
        pickle.dump(GExc_spks.get_states(),pfile)   
    with open(raw_dir+'ginh_spks.p','wb') as pfile:
        pickle.dump(GInh_spks.get_states(),pfile)

    if tr.external_mode=='poisson':
        with open(raw_dir+'pinp_spks.p','wb') as pfile:
            pickle.dump(PInp_spks.get_states(),pfile)

    with open(raw_dir+'gexc_rate.p','wb') as pfile:
        pickle.dump(GExc_rate.get_states(),pfile)
        if tr.rates_rec:
            pickle.dump(GExc_rate.smooth_rate(width=25*ms),pfile)   
    with open(raw_dir+'ginh_rate.p','wb') as pfile:
        pickle.dump(GInh_rate.get_states(),pfile)
        if tr.rates_rec:
            pickle.dump(GInh_rate.smooth_rate(width=25*ms),pfile)

    if tr.population_binned_rec:
        with open(raw_dir+'gexc_binned.p', 'wb') as pfile:
            pickle.dump(GExc_binned.get_states(),pfile)
        with open(raw_dir+'ginh_binned.p', 'wb') as pfile:
            pickle.dump(GInh_binned.get_states(),pfile)

    if tr.external_mode=='poisson':
        with open(raw_dir+'pinp_rate.p','wb') as pfile:
            pickle.dump(PInp_rate.get_states(),pfile)
            if tr.rates_rec:
                pickle.dump(PInp_rate.smooth_rate(width=25*ms),pfile)   


    # ----------------- add raw data ------------------------
    fpath = 'builds/%.4d/'%(tr.v_idx)

    # files written by output_files.cpp, with their number of columns
    # (see cpp_methods.py)
    for fname, ncols in [('turnover', 4),
                         ('spk_register', 7), ('spk_register_EI', 7),
                         ('scaling_deltas', 5), ('scaling_deltas_EI', 5)]:

        # the file might not exist or be empty if nothing was recorded;
        # reshape so that a file with zero or one rows is still 2D
        data = np.empty((0, ncols))
        if os.path.exists(fpath+fname):
            if os.path.getsize(fpath+fname) > 0:
                data = np.loadtxt(fpath+fname, delimiter=',', ndmin=2)
            os.remove(fpath+fname)

        with open(raw_dir+fname+'.p','wb') as pfile:
            pickle.dump(data,pfile)



    # --------------- cross-correlations ---------------------

    if tr.crs_crrs_rec:

        # spike times lie on the dt grid, i.e. often exactly on bin edges;
        # elephant correctly shifts them into the next bin but logs a
        # warning for every single spike train
        logging.getLogger(elephant.utils.__file__).addFilter(
            lambda record: 'rounding errors' not in record.getMessage())

        GExc_spks = GExc_spks.get_states()
        synee_a = SynEE_a_states
        wsize = 100*pq.ms

        for binsize in [1*pq.ms, 2*pq.ms, 5*pq.ms]: 

            wlen = int(wsize/binsize)

            ts, idxs = GExc_spks['t'], GExc_spks['i']
            idxs = idxs[ts>tr.T1+tr.T2+tr.T3+tr.T4]
            ts = ts[ts>tr.T1+tr.T2+tr.T3+tr.T4]
            ts = ts - (tr.T1+tr.T2+tr.T3+tr.T4)

            sts = [neo.SpikeTrain(ts[idxs==i]/second*pq.s,
                                  t_stop=tr.T5/second*pq.s) for i in
                   range(tr.N_e)]

            crs_crrs, syn_a = [], []

            for f,(i,j) in enumerate(zip(synee_a['i'], synee_a['j'])):
                if synee_a['syn_active'][-1][f]==1:

                    crs_crr, cbin = cch(BinnedSpikeTrain(sts[i],
                                                         bin_size=binsize),
                                        BinnedSpikeTrain(sts[j],
                                                         bin_size=binsize),
                                        cross_correlation_coefficient=True,
                                        border_correction=True,
                                        window=(-1*wlen,wlen))

                    crs_crrs.append(list(np.array(crs_crr).T[0]))
                    syn_a.append(synee_a['a'][-1][f])


            fname = 'crs_crrs_wsize%dms_binsize%fms_full' %(wsize/pq.ms,
                                                            binsize/pq.ms)

            df = {'cbin': cbin, 'crs_crrs': np.array(crs_crrs),
                  'syn_a': np.array(syn_a), 'binsize': binsize,
                  'wsize': wsize, 'wlen': wlen}


            with open('builds/%.4d/raw/'%(tr.v_idx)+fname+'.p', 'wb') as pfile:
                pickle.dump(df, pfile)


        GInh_spks = GInh_spks.get_states()
        synei_a = SynEI_a_states
        wsize = 100*pq.ms

        for binsize in [1*pq.ms, 2*pq.ms, 5*pq.ms]: 

            wlen = int(wsize/binsize)

            ts_E, idxs_E = GExc_spks['t'], GExc_spks['i']
            idxs_E = idxs_E[ts_E>tr.T1+tr.T2+tr.T3+tr.T4]
            ts_E = ts_E[ts_E>tr.T1+tr.T2+tr.T3+tr.T4]
            ts_E = ts_E - (tr.T1+tr.T2+tr.T3+tr.T4)

            ts_I, idxs_I = GInh_spks['t'], GInh_spks['i']
            idxs_I = idxs_I[ts_I>tr.T1+tr.T2+tr.T3+tr.T4]
            ts_I = ts_I[ts_I>tr.T1+tr.T2+tr.T3+tr.T4]
            ts_I = ts_I - (tr.T1+tr.T2+tr.T3+tr.T4)

            sts_E = [neo.SpikeTrain(ts_E[idxs_E==i]/second*pq.s,
                                    t_stop=tr.T5/second*pq.s) for i in
                     range(tr.N_e)]

            sts_I = [neo.SpikeTrain(ts_I[idxs_I==i]/second*pq.s,
                                    t_stop=tr.T5/second*pq.s) for i in
                     range(tr.N_i)]

            crs_crrs, syn_a = [], []

            for f,(i,j) in enumerate(zip(synei_a['i'], synei_a['j'])):
                if synei_a['syn_active'][-1][f]==1:

                    crs_crr, cbin = cch(BinnedSpikeTrain(sts_I[i],
                                                         bin_size=binsize),
                                        BinnedSpikeTrain(sts_E[j],
                                                         bin_size=binsize),
                                        cross_correlation_coefficient=True,
                                        border_correction=True,
                                        window=(-1*wlen,wlen))

                    crs_crrs.append(list(np.array(crs_crr).T[0]))
                    syn_a.append(synei_a['a'][-1][f])


            fname = 'EI_crrs_wsize%dms_binsize%fms_full' %(wsize/pq.ms,
                                                            binsize/pq.ms)

            df = {'cbin': cbin, 'crs_crrs': np.array(crs_crrs),
                  'syn_a': np.array(syn_a), 'binsize': binsize,
                  'wsize': wsize, 'wlen': wlen}


            with open('builds/%.4d/raw/'%(tr.v_idx)+fname+'.p', 'wb') as pfile:
                pickle.dump(df, pfile)


    # -----------------  clean up  ---------------------------
    shutil.rmtree('builds/%.4d/results/'%(tr.v_idx))
    shutil.rmtree('builds/%.4d/static_arrays/'%(tr.v_idx))
    shutil.rmtree('builds/%.4d/brianlib/'%(tr.v_idx))
    shutil.rmtree('builds/%.4d/code_objects/'%(tr.v_idx))
            

    # ---------------- plot results --------------------------

    #os.chdir('./analysis/file_based/')

    from analysis.overview_winh import overview_figure
    overview_figure('builds/%.4d'%(tr.v_idx), namespace)

    from analysis.frequencies import frequencies_figure
    frequencies_figure('builds/%.4d'%(tr.v_idx))

    from analysis.isi import isi_figure
    isi_figure('builds/%.4d'%(tr.v_idx))

    from analysis.synw_fb import synw_figure
    synw_figure('builds/%.4d'%(tr.v_idx), namespace)
    if tr.istdp_active:
        synw_figure('builds/%.4d'%(tr.v_idx), namespace, connections='EI')

    from analysis.synw_log_fb import synw_log_figure
    synw_log_figure('builds/%.4d'%(tr.v_idx), namespace)
    if tr.istdp_active:
        synw_log_figure('builds/%.4d'%(tr.v_idx), namespace, connections='EI')
    
    from analysis.turnover_fb import turnover_figure
    turnover_figure('builds/%.4d'%(tr.v_idx), namespace, fit=False)

    from analysis.turnover_fb import turnover_figure
    turnover_figure('builds/%.4d'%(tr.v_idx), namespace, fit=True)

          
