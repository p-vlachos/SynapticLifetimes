
import argparse, sys, os, itertools, pickle, time
import numpy as np
from brian2.units import mV, ms, second

from .methods.process_survival import extract_survival


    
if __name__ == "__main__":

    
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
            t_split = (nsp['T2']-t_cut)/2.


            print('started loading data')
            a = time.time()
            with open(bpath+'/raw/turnover.p', 'rb') as pfile:
                turnover = pickle.load(pfile)
            b=time.time()
            print('finished loading data, took %.2f seconds' %(b-a))

            a=time.time()
            print('\n started survival extractation')
            full_t, ex_ids = extract_survival(turnover,
                                              nsp['N_e'],
                                              t_split=t_split,
                                              t_cut=t_cut)

            
            b = time.time()
            print('finished survival extraction, took %.2f seconds' %(b-a))

            with open(bpath+'/raw/survival_full_t.p', 'wb') as pfile:
                out = {'t_split': t_split, 't_cut': t_cut,
                       'full_t': full_t, 'excluded_ids': ex_ids}
                pickle.dump(out, pfile)
                    


        except FileNotFoundError:
            print(bpath[-4:], "reports: No namespace data. Skipping.")


