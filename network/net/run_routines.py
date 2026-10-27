
from brian2.units import second

# the functions used in the model equations are looked up in the namespace
# of the module that calls net.run
from .cpp_methods import syn_scale, syn_EI_scale, \
                         record_turnover, record_spk, record_spk_EI, record_weights


def run_phase(net, duration, name):
    if duration == 0*second:
        print(f"{name} is zero, skipping")
        return
    net.run(duration, report='text', report_period=300*second)


def run_T2_syndynrec(net, tr):
    '''
    main simulation period, only turnover and SynEE_a are recorded;
    optionally record the weight dynamics (syndynrec) in a short time
    interval at the beginning and at the end of the period
    '''

    if not ((tr.synEEdynrec or tr.synEIdynrec) and
            (2*tr.syndynrec_npts*tr.syndynrec_dt < tr.sim.T2)):
        run_phase(net, tr.sim.T2, 'T2')
        return

    print('Doing syndynrec!')

    def set_dynrec_active(active):
        if tr.synEEdynrec:
            net['SynEE_dynrec'].active = active
        if tr.synEIdynrec:
            net['SynEI_dynrec'].active = active

    set_dynrec_active(True)
    run_phase(net, tr.syndynrec_npts*tr.syndynrec_dt, 'T2 (begin)')
    set_dynrec_active(False)
    run_phase(net, tr.sim.T2 - 2*tr.syndynrec_npts*tr.syndynrec_dt, 'T2')
    set_dynrec_active(True)
    run_phase(net, tr.syndynrec_npts*tr.syndynrec_dt, 'T2 (end)')
    set_dynrec_active(False)
