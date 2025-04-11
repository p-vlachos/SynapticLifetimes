
import pickle
import numpy as np

from brian2.units import ms,mV,second,Hz

from analysis.methods.process_survival import extract_survival
from analysis.methods.process_turnover_pd import extract_lifetimes
from analysis.methods.process_turnover_statistics import get_insert_and_prune_counts

from analysis.srvprb_all import srvprb_all_figure
from analysis.srvprb_EE import srvprb_EE_figure
from analysis.srvprb_EI import srvprb_EI_figure
from analysis.branching_ratio import branching_ratio_figure

from analysis.methods.resample_dA import resample_spk_register, \
                                         resample_scl_deltas


def post_process_turnover(tr, connections='EE', binning=True, from_numpy=False, turnover=None):

    if connections=='EE':
        cn=''
    elif connections=='EI':
        cn='_EI'

    t_cut = tr.pp_tcut
    t_split = (tr.T1+tr.T2+tr.T3+tr.T4-t_cut)/2.

    if t_split/second > 0:

        # E<-E connections

        if turnover is None:
            if from_numpy:
                turnover = np.load('builds/%.4d/raw/turnover%s.npz'%(tr.v_idx,cn))['turnover']
            else:
                with open('builds/%.4d/raw/turnover%s.p' %(tr.v_idx,cn), 'rb') as pfile:
                    turnover = pickle.load(pfile)

        # abort if turnover empty
        if len(turnover)==0:
            return
        
        # tr.N_e only used for s_id creation,
        # and since tr.N_i > tr.N_e,
        # tr.NE is correct for both 'EE' & 'EI'
        print("starting survival times")
        full_t, ex_ids = extract_survival(turnover,
                                          tr.N_e,
                                          t_split=t_split,
                                          t_cut=t_cut)
        print("done with survival, saving now")

        fpath = 'builds/%.4d/raw/survival%s_full_t.p' %(tr.v_idx, cn)
        with open(fpath, 'wb') as pfile:
            out = {'t_split': t_split, 't_cut': t_cut,
                   'full_t': full_t, 'excluded_ids': ex_ids}
            pickle.dump(out, pfile)

    if not binning:
        return

    Tmax = tr.T1 + tr.T2 + tr.T3 + tr.T4

    lts_wthsrv, lts_dthonly, ex_ids = extract_lifetimes(turnover,
                                                        tr.N_e,
                                                        t_cut=t_cut,
                                                        Tmax=Tmax)

    bpath = 'builds/%.4d' %(tr.v_idx)
    for bin_w in [0.1*second, 0.5*second, 1*second, 10*second, 100*second]:

        bins = np.arange(bin_w/second,
                         Tmax/second+2*bin_w/second,
                         bin_w/second)

        f_add = 'bin%dcs' %(int(bin_w/second*10.))

        counts, edges = np.histogram(lts_wthsrv, bins=bins,
                                     density=True)
        centers = (edges[:-1] + edges[1:])/2.            
        
        with open(bpath+'/raw/lts%s_wthsrv_' %cn +f_add+'.p', 'wb') as pfile:
            out = {'Tmax': Tmax, 't_cut': t_cut,
                   'counts': counts, 'excluded_ids': ex_ids,
                   'bins': bins, 'centers': centers, 'bin_w': bin_w}
            pickle.dump(out, pfile)


        counts, edges = np.histogram(lts_dthonly, bins=bins,
                                     density=True)
        centers = (edges[:-1] + edges[1:])/2.            

        with open(bpath+'/raw/lts%s_dthonly_' %cn +f_add+'.p', 'wb') as pfile:
            out = {'Tmax': Tmax, 't_cut': t_cut,
                   'counts': counts, 'excluded_ids': ex_ids,
                   'bins': bins, 'centers': centers, 'bin_w': bin_w}
            pickle.dump(out, pfile)


    for nbins in [25,50,100,250,500,1000,2500,5000]:

        bins = np.logspace(np.log10(1),
                           np.log10((Tmax-t_cut)/second+0.5),
                           num=nbins)

        f_add = 'lognbin%d' %(nbins)

        counts, edges = np.histogram(lts_wthsrv, bins=bins,
                                     density=True)
        centers = (edges[:-1] + edges[1:])/2.            

        with open(bpath+'/raw/lts%s_wthsrv_' %cn +f_add+'.p', 'wb') as pfile:
            out = {'Tmax': Tmax, 't_cut': t_cut,
                   'counts': counts, 'excluded_ids': ex_ids,
                   'bins': bins, 'centers': centers, 'nbins': nbins}
            pickle.dump(out, pfile)


        counts, edges = np.histogram(lts_dthonly, bins=bins,
                                     density=True)
        centers = (edges[:-1] + edges[1:])/2.            

        with open(bpath+'/raw/lts%s_dthonly_' %cn +f_add+'.p', 'wb') as pfile:
            out = {'Tmax': Tmax, 't_cut': t_cut,
                   'counts': counts, 'excluded_ids': ex_ids,
                   'bins': bins, 'centers': centers, 'nbins': nbins}
            pickle.dump(out, pfile)



def post_process_per_neuron_turnover_counts(tr, connections='EE'):

    if connections=='EE':
        cn=''
    elif connections=='EI':
        cn='_EI'

    t_cut = tr.pp_tcut
    t_max = tr.T1+tr.T2+tr.T3+tr.T4

    with open('builds/%.4d/raw/turnover%s.p' %(tr.v_idx,cn), 'rb') as pfile:
        turnover = pickle.load(pfile)
        
    ins_c, prn_c = get_insert_and_prune_counts(turnover, tr.N_e, t_cut, t_max)

    bpath = 'builds/%.4d' %(tr.v_idx)
    
    with open(bpath+'/raw/ins-prn_counts%s' %cn + '.p', 'wb') as pfile:
        out = {'t_max': t_max, 't_cut': t_cut,
               'ins_c': ins_c, 'prn_c': prn_c}
        pickle.dump(out, pfile)
    



def post_process_stdp_rec(tr):

    bpath = 'builds/%.4d' %(tr.v_idx)

    if tr.synEE_rec:
    
        with open(bpath+'/raw/spk_register.p', 'rb') as pfile:
            spkr = pickle.load(pfile)

        for resamp_dt in [1*second, 2*second, 5*second, 10*second]:

            dA_v, a_v = resample_spk_register(spkr, tr, resamp_dt)

            fname = 'stdp_dA_T%ds_bin%dms.p' %(int(tr.stdp_rec_T/second),
                                               int(resamp_dt/second*1000))

            with open(bpath+'/raw/'+fname, 'wb') as pfile:
                pickle.dump((dA_v, a_v), pfile)

    if tr.synEI_rec:
    
        with open(bpath+'/raw/spk_register_EI.p', 'rb') as pfile:
            spkr = pickle.load(pfile)

        for resamp_dt in [1*second, 2*second, 5*second, 10*second]:

            dA_v, a_v = resample_spk_register(spkr, tr, resamp_dt)

            fname = 'stdp_EI_dA_T%ds_bin%dms.p' %(int(tr.stdp_rec_T/second),
                                                  int(resamp_dt/second*1000))

            with open(bpath+'/raw/'+fname, 'wb') as pfile:
                pickle.dump((dA_v, a_v), pfile)


def post_process_scl_rec(tr):

    bpath = 'builds/%.4d' %(tr.v_idx)
    
    if tr.syn_scl_rec:
        
        with open(bpath+'/raw/scaling_deltas.p', 'rb') as pfile:
            scl_deltas = pickle.load(pfile)

        for resamp_dt in [1*second, 2*second, 5*second, 10*second]:

            dA_v, a_v = resample_scl_deltas(scl_deltas, tr, resamp_dt)

            fname = 'scl_dA_T%ds_bin%dms.p' %(int(tr.scl_rec_T/second),
                                               int(resamp_dt/second*1000))

            with open(bpath+'/raw/'+fname, 'wb') as pfile:
                pickle.dump((dA_v, a_v), pfile)

    if tr.syn_iscl_rec:

        with open(bpath+'/raw/scaling_deltas_EI.p', 'rb') as pfile:
            scl_deltas = pickle.load(pfile)

        for resamp_dt in [1*second, 2*second, 5*second, 10*second]:

            dA_v, a_v = resample_scl_deltas(scl_deltas, tr, resamp_dt)

            fname = 'scl_EI_dA_T%ds_bin%dms.p' %(int(tr.scl_rec_T/second),
                                               int(resamp_dt/second*1000))

            with open(bpath+'/raw/'+fname, 'wb') as pfile:
                pickle.dump((dA_v, a_v), pfile)


            


def post_process(tr):

    want_postprocess = tr.T5 > 50*second

    if want_postprocess:
        branching_ratio_figure(f'builds/{tr.v_idx:04d}')
    post_process_stdp_rec(tr)
    post_process_scl_rec(tr)

    if tr.turnover_rec and want_postprocess:

        if tr.strct_active:
            post_process_turnover(tr, 'EE')
        if tr.istdp_active and tr.istrct_active:
            post_process_turnover(tr, 'EI')

        if tr.strct_active:
            post_process_per_neuron_turnover_counts(tr, 'EE')
        if tr.istdp_active and tr.istrct_active:
            post_process_per_neuron_turnover_counts(tr, 'EI')

        
        srvprb_EE_figure('builds/%.4d'%(tr.v_idx))
        srvprb_EI_figure('builds/%.4d'%(tr.v_idx))
        srvprb_all_figure('builds/%.4d'%(tr.v_idx))


