
from pypet.brian2.parameter import Brian2Parameter

from . import standard_params as prm
from . import models as mod

def add_params(tr):

    tr.v_standard_parameter=Brian2Parameter
    tr.v_fast_access=True

    tr.f_add_parameter('netw.N_e', prm.N_e)
    tr.f_add_parameter('netw.N_i', prm.N_i)

    tr.f_add_parameter('netw.config.syn_cond_mode',   prm.syn_cond_mode)
    tr.f_add_parameter('netw.config.syn_cond_mode_EI',   prm.syn_cond_mode_EI)
    tr.f_add_parameter('netw.tau',   prm.tau)
    tr.f_add_parameter('netw.tau_e', prm.tau_e)
    tr.f_add_parameter('netw.tau_i', prm.tau_i)
    tr.f_add_parameter('netw.tau_e_rise', prm.tau_e_rise)
    tr.f_add_parameter('netw.tau_i_rise', prm.tau_i_rise)
    tr.f_add_parameter('netw.norm_f_EE', prm.norm_f_EE)
    tr.f_add_parameter('netw.norm_f_EI', prm.norm_f_EI)

    # synaptic delays
    tr.f_add_parameter("netw.config.syn_delay_active", prm.syn_delay_active)
    tr.f_add_parameter("netw.config.syn_dd_delay_active", prm.syn_dd_delay_active)
    tr.f_add_parameter("netw.syn_dd_delay_velocity", prm.syn_dd_delay_velocity)
    tr.f_add_parameter("netw.syn_dd_delay_rel_sd", prm.syn_dd_delay_rel_sd)
    tr.f_add_parameter("netw.synEE_delay", prm.synEE_delay)
    tr.f_add_parameter("netw.synEE_delay_windowsize", prm.synEE_delay_windowsize)
    tr.f_add_parameter("netw.synEI_delay", prm.synEI_delay)
    tr.f_add_parameter("netw.synEI_delay_windowsize", prm.synEI_delay_windowsize)
    tr.f_add_parameter("netw.synII_delay", prm.synII_delay)
    tr.f_add_parameter("netw.synII_delay_windowsize", prm.synII_delay_windowsize)
    tr.f_add_parameter("netw.synIE_delay", prm.synIE_delay)
    tr.f_add_parameter("netw.synIE_delay_windowsize", prm.synIE_delay_windowsize)
    
    tr.f_add_parameter('netw.El',    prm.El)
    tr.f_add_parameter('netw.Ee',    prm.Ee)
    tr.f_add_parameter('netw.Ei',    prm.Ei)
    
    tr.f_add_parameter('netw.Vr_e',  prm.Vr_e)
    tr.f_add_parameter('netw.Vr_i',  prm.Vr_i)
    tr.f_add_parameter('netw.Vt_e',  prm.Vt_e)
    tr.f_add_parameter('netw.Vt_i',  prm.Vt_i)

    tr.f_add_parameter('netw.a_ee',  prm.a_ee)
    tr.f_add_parameter('netw.a_ie',  prm.a_ie)
    tr.f_add_parameter('netw.a_ei',  prm.a_ei)
    tr.f_add_parameter('netw.a_ii',  prm.a_ii)

    tr.f_add_parameter('netw.p_ee',  prm.p_ee)
    tr.f_add_parameter('netw.p_ie',  prm.p_ie)
    tr.f_add_parameter('netw.p_ei',  prm.p_ei)
    tr.f_add_parameter('netw.p_ii',  prm.p_ii)

    # Poisson Input
    tr.f_add_parameter('netw.external_mode', prm.external_mode)
    tr.f_add_parameter('netw.mu_e', prm.mu_e)
    tr.f_add_parameter('netw.mu_i', prm.mu_i)
    tr.f_add_parameter('netw.sigma_e', prm.sigma_e)
    tr.f_add_parameter('netw.sigma_i', prm.sigma_i)

    tr.f_add_parameter('netw.PInp_mode',  prm.PInp_mode)
    tr.f_add_parameter('netw.NPInp',  prm.NPInp)
    tr.f_add_parameter('netw.NPInp_1n',  prm.NPInp_1n)
    tr.f_add_parameter('netw.NPInp_inh',  prm.NPInp_inh)
    tr.f_add_parameter('netw.NPInp_inh_1n',  prm.NPInp_inh_1n)    
    tr.f_add_parameter('netw.a_EPoi',  prm.a_EPoi)
    tr.f_add_parameter('netw.a_IPoi',  prm.a_IPoi)
    tr.f_add_parameter('netw.PInp_rate',  prm.PInp_rate)
    tr.f_add_parameter('netw.PInp_inh_rate',  prm.PInp_inh_rate)

    tr.f_add_parameter('netw.refractory_exc', prm.refractory_exc)
    tr.f_add_parameter('netw.refractory_inh', prm.refractory_inh)

    # synapse noise
    tr.f_add_parameter('netw.syn_noise',  prm.syn_noise)
    tr.f_add_parameter('netw.syn_sigma_EE',  prm.syn_sigma_EE)
    tr.f_add_parameter('netw.syn_sigma_EI',  prm.syn_sigma_EI)
    tr.f_add_parameter('netw.synEE_mod_dt',  prm.synEE_mod_dt)

    tr.f_add_parameter('netw.synEE_static',  mod.synEE_static)
    tr.f_add_parameter('netw.syn_noise_mod',  mod.syn_noise_mod)
    tr.f_add_parameter('netw.tau_r', prm.tau_r)
    tr.f_add_parameter('netw.synEE_scl_mod',  mod.synEE_scl_mod)
    tr.f_add_parameter('netw.synEI_scl_mod',  mod.synEI_scl_mod)

    tr.f_add_parameter('netw.synEE_pre_STDP', mod.synEE_pre_STDP)
    tr.f_add_parameter('netw.condlif_rate_sensor', mod.condlif_rate_sensor)
    tr.f_add_parameter('netw.reset_rate_sensor', mod.reset_rate_sensor)
    tr.f_add_parameter('netw.scl_scaling_kappa', prm.scaling_kappa)
    tr.f_add_parameter('netw.scl_scaling_eta', prm.scaling_eta)
    tr.f_add_parameter('netw.scl_scaling_dt', prm.scaling_dt)

    # STDP
    tr.f_add_parameter('netw.config.stdp_active', prm.stdp_active)
    tr.f_add_parameter('netw.taupre',    prm.taupre)
    tr.f_add_parameter('netw.taupost',   prm.taupost)
    tr.f_add_parameter('netw.Aplus',     prm.Aplus)
    tr.f_add_parameter('netw.Aminus',    prm.Aminus)
    tr.f_add_parameter('netw.amax',      prm.amax)
    tr.f_add_parameter('netw.amin',      prm.amin)
    tr.f_add_parameter('netw.amin_i',    prm.amin_i)
    tr.f_add_parameter('netw.amax_i',    prm.amax_i)
    tr.f_add_parameter('netw.synEE_rec',      prm.synEE_rec)

    # iSTDP
    tr.f_add_parameter('netw.config.istdp_active', prm.istdp_active)
    tr.f_add_parameter('netw.taupre_EI',    prm.taupre_EI)
    tr.f_add_parameter('netw.taupost_EI',   prm.taupost_EI)
    tr.f_add_parameter('netw.synEI_rec',      prm.synEI_rec)
    tr.f_add_parameter('netw.iAplus', prm.iAplus)
    tr.f_add_parameter('netw.iAminus', prm.iAminus)

    # scaling
    tr.f_add_parameter('netw.config.scl_active', prm.scl_active)
    tr.f_add_parameter('netw.ATotalMax',         prm.ATotalMax)
    tr.f_add_parameter('netw.sig_ATotalMax',     prm.sig_ATotalMax)
    tr.f_add_parameter('netw.dt_synEE_scaling',  prm.dt_synEE_scaling)
    tr.f_add_parameter('netw.eta_scaling',       prm.eta_scaling)
    tr.f_add_parameter('netw.mod.synEE_scaling', mod.synEE_scaling)

    # iscaling
    tr.f_add_parameter('netw.config.iscl_active', prm.iscl_active)
    tr.f_add_parameter('netw.mod.synEI_scaling', mod.synEI_scaling)
    tr.f_add_parameter('netw.iATotalMax',        prm.iATotalMax)
    tr.f_add_parameter('netw.sig_iATotalMax',    prm.sig_iATotalMax)
    tr.f_add_parameter('netw.syn_iscl_rec',        prm.syn_iscl_rec)
    tr.f_add_parameter('netw.eta_iscaling',       prm.eta_iscaling)

    # structural plasticity
    tr.f_add_parameter('netw.insert_P',    prm.insert_P)
    tr.f_add_parameter('netw.a_insert',    prm.a_insert)
    tr.f_add_parameter('netw.strct_dt',    prm.strct_dt)
    tr.f_add_parameter('netw.p_inactivate',    prm.p_inactivate)
    tr.f_add_parameter('netw.strct_c',    prm.strct_c)

    tr.f_add_parameter('netw.mod.neuron_memnoise',   mod.neuron_memnoise)
    tr.f_add_parameter('netw.mod.neuron_poisson',   mod.neuron_poisson)
    tr.f_add_parameter('netw.mod.neuron_exc_normalization',   mod.neuron_exc_normalization)
    tr.f_add_parameter('netw.mod.syn_cond_EE_exp',   mod.syn_cond_EE_exp)
    tr.f_add_parameter('netw.mod.syn_cond_EE_biexp',   mod.syn_cond_EE_biexp)
    tr.f_add_parameter('netw.mod.syn_cond_EI_exp',   mod.syn_cond_EI_exp)
    tr.f_add_parameter('netw.mod.syn_cond_EI_biexp',   mod.syn_cond_EI_biexp)
    tr.f_add_parameter('netw.mod.neuron_threshold', mod.neuron_threshold)
    tr.f_add_parameter('netw.mod.neuron_reset_exc', mod.neuron_reset_exc)
    tr.f_add_parameter('netw.mod.neuron_reset_inh', mod.neuron_reset_inh)
    tr.f_add_parameter('netw.mod.synEE_mod',     mod.synEE_mod)

    tr.f_add_parameter('netw.mod.strct_mod',     mod.strct_mod)
    tr.f_add_parameter('netw.mod.turnover_rec_mod',     mod.turnover_rec_mod)
    

    tr.f_add_parameter('netw.sim.T1',  prm.T1)
    tr.f_add_parameter('netw.sim.T2',  prm.T2)
    tr.f_add_parameter('netw.sim.T3',  prm.T3)
    tr.f_add_parameter('netw.sim.T4',  prm.T4)
    tr.f_add_parameter('netw.sim.T5',  prm.T5)
    tr.f_add_parameter('netw.sim.dt', prm.dt)
    tr.f_add_parameter('netw.sim.n_threads', prm.n_threads)

    tr.f_add_parameter('netw.config.strct_active', prm.strct_active)
    tr.f_add_parameter('netw.rec.turnover_rec', prm.turnover_rec)

    # distance-dependent connectivity
    tr.f_add_parameter('netw.config.ddcon_active', prm.ddcon_active)
    tr.f_add_parameter('netw.grid_size', prm.grid_size)
    tr.f_add_parameter('netw.sigma_dd', prm.sigma_dd)
    tr.f_add_parameter('netw.sigma_dd_inh', prm.sigma_dd_inh)
    tr.f_add_parameter('netw.mod.ddcon', mod.ddcon)
    tr.f_add_parameter('netw.grid_wrap', prm.grid_wrap)

    # recording
    tr.f_add_parameter('netw.rec.memtraces_rec', prm.memtraces_rec)
    tr.f_add_parameter('netw.rec.vttraces_rec', prm.vttraces_rec)
    tr.f_add_parameter('netw.rec.getraces_rec', prm.getraces_rec)
    tr.f_add_parameter('netw.rec.gitraces_rec', prm.gitraces_rec)
    tr.f_add_parameter('netw.rec.gfwdtraces_rec', prm.gfwdtraces_rec)
    tr.f_add_parameter('netw.rec.rates_rec', prm.rates_rec)
    tr.f_add_parameter('netw.rec.anormtar_rec', prm.anormtar_rec)

    tr.f_add_parameter('netw.rec.nrec_GExc_stat', prm.nrec_GExc_stat)
    tr.f_add_parameter('netw.rec.nrec_GInh_stat', prm.nrec_GInh_stat)
    tr.f_add_parameter('netw.rec.GExc_stat_dt', prm.GExc_stat_dt)
    tr.f_add_parameter('netw.rec.GInh_stat_dt', prm.GInh_stat_dt)

    tr.f_add_parameter('netw.rec.syn_scl_rec', prm.syn_scl_rec)
    tr.f_add_parameter('netw.rec.stdp_rec_T', prm.stdp_rec_T)
    tr.f_add_parameter('netw.rec.scl_rec_T', prm.scl_rec_T)

    tr.f_add_parameter('netw.rec.synEEdynrec', prm.synEEdynrec)
    tr.f_add_parameter('netw.rec.synEIdynrec', prm.synEIdynrec)
    tr.f_add_parameter('netw.rec.syndynrec_dt', prm.syndynrec_dt)
    tr.f_add_parameter('netw.rec.syndynrec_npts', prm.syndynrec_npts)

    tr.f_add_parameter('netw.rec.synee_atraces_rec',
                       prm.synee_atraces_rec)
    tr.f_add_parameter('netw.rec.synee_activetraces_rec',
                       prm.synee_activetraces_rec)
    tr.f_add_parameter('netw.rec.synee_Apretraces_rec',
                       prm.synee_Apretraces_rec)
    tr.f_add_parameter('netw.rec.synee_Aposttraces_rec',
                       prm.synee_Aposttraces_rec)
    tr.f_add_parameter('netw.rec.n_synee_traces_rec',
                       prm.n_synee_traces_rec)
    tr.f_add_parameter('netw.rec.synEE_stat_dt', prm.synEE_stat_dt)

    tr.f_add_parameter('netw.rec.synei_atraces_rec',
                       prm.synei_atraces_rec)
    tr.f_add_parameter('netw.rec.synei_activetraces_rec',
                       prm.synei_activetraces_rec)
    tr.f_add_parameter('netw.rec.synei_Apretraces_rec',
                       prm.synei_Apretraces_rec)
    tr.f_add_parameter('netw.rec.synei_Aposttraces_rec',
                       prm.synei_Aposttraces_rec)
    tr.f_add_parameter('netw.rec.n_synei_traces_rec',
                       prm.n_synei_traces_rec)
    tr.f_add_parameter('netw.rec.synEI_stat_dt', prm.synEI_stat_dt)

    tr.f_add_parameter('netw.rec.T2_spks_rec', prm.T2_spks_rec)
    tr.f_add_parameter('netw.synee_a_nrecpoints', prm.synee_a_nrecpoints)
    tr.f_add_parameter('netw.synei_a_nrecpoints', prm.synei_a_nrecpoints)
    
    tr.f_add_parameter('netw.crs_crrs_rec', prm.crs_crrs_rec)

    tr.f_add_parameter('netw.adjust_insertP', prm.adjust_insertP)
    tr.f_add_parameter('netw.adjust_insertP_mode', prm.adjust_insertP_mode)
    tr.f_add_parameter('netw.csample_dt', prm.csample_dt)
    

    # post processing
    tr.f_add_parameter('netw.pp_tcut', prm.pp_tcut)

    tr.f_add_parameter('netw.population_binned_rec', prm.population_binned_rec)

    # seed
    tr.f_add_parameter('netw.random_seed', prm.random_seed)
