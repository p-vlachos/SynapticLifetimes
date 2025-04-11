
import argparse, sys, os, itertools, pickle, time
import numpy as np
from brian2.units import mV, ms, second

from .methods.process_turnover_pd import extract_lifetimes


# return a list of each build (simulations run)
# e.g. build_dirs = ['builds/0003', 'builds/0007', ...]
# sorted to ensure expected order
build_dirs = sorted(['builds/'+pth for pth in next(os.walk("builds/"))[1]])

bin_w = 1*second
fit = False


for bpath in build_dirs:

    try:

        print('Found ', bpath)

        with open(bpath+'/raw/namespace.p', 'rb') as pfile:
            nsp=pickle.load(pfile)

        t_cut = 100*second
        Tmax = nsp['T1']+nsp['T2']+nsp['T3']

        print('started loading data')
        a = time.time()
        with open(bpath+'/raw/turnover.p', 'rb') as pfile:
            turnover = pickle.load(pfile)
        b=time.time()
        print('finished loading data, took %.2f seconds' %(b-a))

        a=time.time()
        print('\n started survival extractation')
        lts_wthsrv, lts_dthonly, ex_ids = extract_lifetimes(turnover,
                                                            nsp['N_e'],
                                                            t_cut=t_cut,
                                                            Tmax=Tmax)

        b = time.time()
        print('finished lifetimes extraction, took %.2f seconds' %(b-a))



        for bin_w in [0.1*second, 0.5*second, 1*second, 10*second, 100*second]:

            bins = np.arange(bin_w/second,
                             Tmax/second+2*bin_w/second,
                             bin_w/second)

            f_add = 'bin%dcs' %(int(bin_w/second*10.))

            counts, edges = np.histogram(lts_wthsrv, bins=bins,
                                         density=True)
            centers = (edges[:-1] + edges[1:])/2.            

            with open(bpath+'/raw/lts_wthsrv_'+f_add+'.p', 'wb') as pfile:
                out = {'Tmax': Tmax, 't_cut': t_cut,
                       'counts': counts, 'excluded_ids': ex_ids,
                       'bins': bins, 'centers': centers, 'bin_w': bin_w}
                pickle.dump(out, pfile)


            counts, edges = np.histogram(lts_dthonly, bins=bins,
                                         density=True)
            centers = (edges[:-1] + edges[1:])/2.            

            with open(bpath+'/raw/lts_dthonly_'+f_add+'.p', 'wb') as pfile:
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

            with open(bpath+'/raw/lts_wthsrv_'+f_add+'.p', 'wb') as pfile:
                out = {'Tmax': Tmax, 't_cut': t_cut,
                       'counts': counts, 'excluded_ids': ex_ids,
                       'bins': bins, 'centers': centers, 'nbins': nbins}
                pickle.dump(out, pfile)


            counts, edges = np.histogram(lts_dthonly, bins=bins,
                                         density=True)
            centers = (edges[:-1] + edges[1:])/2.            

            with open(bpath+'/raw/lts_dthonly_'+f_add+'.p', 'wb') as pfile:
                out = {'Tmax': Tmax, 't_cut': t_cut,
                       'counts': counts, 'excluded_ids': ex_ids,
                       'bins': bins, 'centers': centers, 'nbins': nbins}
                pickle.dump(out, pfile)




    except FileNotFoundError:
        print(bpath[-4:], "reports: No namespace data. Skipping.")
