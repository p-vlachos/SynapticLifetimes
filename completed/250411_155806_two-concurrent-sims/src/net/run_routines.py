
from brian2.units import ms,mV,second,Hz

from .cpp_methods import syn_scale, syn_EI_scale, \
                         record_turnover, record_turnover_EI, \
                         record_spk, record_spk_EI

# from .cuda_methods import syn_scale, syn_EI_scale, \
                        #  record_turnover, record_turnover_EI, \
                        #  record_spk, record_spk_EI


def run_T2_syndynrec(net, tr, netw_objects):

    # main simulation period
    # only active recordings are:
    #   1) turnover 2) C_stat 3) SynEE_a

    # stop early if T2 should not be simulated
    if tr.sim.T2 == 0*second:
        print("T2 is zero, returning!!!")
        return
    
    # record the change of weights in a short time interval at
    # beginning and end of simulation
    if ( (tr.synEEdynrec or tr.synEIdynrec) and
         (2*tr.syndynrec_npts*tr.syndynrec_dt < tr.sim.T2) ):

        print('Doing syndynrec!')

        if tr.synEEdynrec:
            net['SynEE_dynrec'].active=True
        if tr.synEIdynrec:
            net['SynEI_dynrec'].active=True
        
        net.run(tr.syndynrec_npts*tr.syndynrec_dt, report='text',
                report_period=300*second, profile=tr.netw.profiling)

        if tr.synEEdynrec:
            net['SynEE_dynrec'].active=False
        if tr.synEIdynrec:
            net['SynEI_dynrec'].active=False

        net.run(tr.sim.T2 - 2*tr.syndynrec_npts*tr.syndynrec_dt,
                report='text', report_period=300*second,
                profile=tr.netw.profiling)

        if tr.synEEdynrec:
            net['SynEE_dynrec'].active=True
        if tr.synEIdynrec:
            net['SynEI_dynrec'].active=True
             
        net.run(tr.syndynrec_npts*tr.syndynrec_dt, report='text',
                report_period=300*second, profile=tr.netw.profiling)

        if tr.synEEdynrec:
            net['SynEE_dynrec'].active=False
        if tr.synEIdynrec:
            net['SynEI_dynrec'].active=False



    else:
        # not recording simulate, normally
        net.run(tr.sim.T2, report='text',
                report_period=300*second, profile=tr.netw.profiling)



def run_T3_split(net, tr):

    if tr.sim.T3 == 0*second:
        print("T3 is zero, returning!!!")
        return

    else:
        
        net.run(tr.sim.T3/2, report='text',
                report_period=300*second, profile=tr.netw.profiling)
        
        # net['GInh'].mu=tr.mu_i+0.75*mV

        net.run(tr.sim.T3/2, report='text',
                report_period=300*second, profile=tr.netw.profiling)
        

def run_T4(net, tr):

    if tr.sim.T4 == 0*second:
        print("T4 is zero, returning!!!")
        return

    else:
        net.run(tr.sim.T4, report='text',
                report_period=300*second, profile=tr.netw.profiling)


def run_T5(net, tr):

    if tr.sim.T5 == 0*second:
        print("T5 is zero, returning!!!")
        return

    else:
        net.run(tr.sim.T5, report='text',
                report_period=300*second, profile=tr.netw.profiling)

