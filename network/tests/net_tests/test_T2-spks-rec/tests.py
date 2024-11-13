
import unittest, os, pickle
from brian2.units import second
import numpy as np


class Test_spks_rec(unittest.TestCase):

    with open('builds/0000/raw/namespace.p', 'rb') as pfile:
        nsp = pickle.load(pfile)

    
    with open('builds/0000/raw/gexc_spks.p', 'rb') as pfile:
        exc_spks_00 = pickle.load(pfile)

    with open('builds/0001/raw/gexc_spks.p', 'rb') as pfile:
        exc_spks_01 = pickle.load(pfile)

    with open('builds/0000/raw/ginh_spks.p', 'rb') as pfile:
        inh_spks_00 = pickle.load(pfile)

    with open('builds/0001/raw/ginh_spks.p', 'rb') as pfile:
        inh_spks_01 = pickle.load(pfile)

   
    def test_exc_spks_not_recorded_T2(self): 

        tmin1, tmax1 = 0*second, self.nsp['T1']
    
        tmin3 = self.nsp['T1']+self.nsp['T2'] 
        tmax3 = self.nsp['T1']+self.nsp['T2']+self.nsp['T3']

        self.assertEqual(np.sum(np.logical_and(
                                self.exc_spks_00['t']>tmax1, 
                                self.exc_spks_00['t']<=tmin3)), 0)


    def test_exc_spks_recorded_in_T2(self): 

        tmin1, tmax1 = 0*second, self.nsp['T1']
    
        tmin3 = self.nsp['T1']+self.nsp['T2'] 
        tmax3 = self.nsp['T1']+self.nsp['T2']+self.nsp['T3']

        self.assertGreater(len(np.logical_and(
                             self.exc_spks_01['t']>tmax1, 
                             self.exc_spks_01['t']<=tmin3)), 0)

    def test_inh_spks_not_recorded_T2(self): 

        tmin1, tmax1 = 0*second, self.nsp['T1']
    
        tmin3 = self.nsp['T1']+self.nsp['T2'] 
        tmax3 = self.nsp['T1']+self.nsp['T2']+self.nsp['T3']

        self.assertEqual(np.sum(np.logical_and(
                                self.inh_spks_00['t']>tmax1, 
                                self.inh_spks_00['t']<=tmin3)), 0)


    def test_inh_spks_recorded_in_T2(self): 

        tmin1, tmax1 = 0*second, self.nsp['T1']
    
        tmin3 = self.nsp['T1']+self.nsp['T2'] 
        tmax3 = self.nsp['T1']+self.nsp['T2']+self.nsp['T3']

        self.assertGreater(len(np.logical_and(
                             self.inh_spks_01['t']>tmax1, 
                             self.inh_spks_01['t']<=tmin3)), 0)


            
        

if __name__ == '__main__':
    unittest.main()
