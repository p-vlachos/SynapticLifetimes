
import unittest, os, pickle

import numpy as np


class Test_varying_sig_normalization_changes_results(unittest.TestCase):
    '''
    '''

    def setUp(self):

        self.bpath_0 = "builds/0000"
        self.bpath_1 = "builds/0001"
        self.bpath_2 = "builds/0002"
        self.bpath_3 = "builds/0003"
        

    def test_membrane_potential_as_expected(self):

        with open(self.bpath_0+'/raw/gexc_stat.p', 'rb') as pfile:
            gexc_0=pickle.load(pfile)

        with open(self.bpath_1+'/raw/gexc_stat.p', 'rb') as pfile:
            gexc_1=pickle.load(pfile)

        with open(self.bpath_2+'/raw/gexc_stat.p', 'rb') as pfile:
            gexc_2=pickle.load(pfile)

        with open(self.bpath_3+'/raw/gexc_stat.p', 'rb') as pfile:
            gexc_3=pickle.load(pfile)
        
        np.testing.assert_equal(gexc_0['t'], gexc_1['t'])
        
        np.testing.assert_equal(gexc_0['ge'], gexc_1['ge'])
        np.testing.assert_equal(gexc_0['gi'], gexc_1['gi'])
        np.testing.assert_equal(gexc_0['V'], gexc_1['V'])

        # although differing normalization should create different
        # dynamics, initial 'V' should match
        np.testing.assert_equal(gexc_1['V'][0], gexc_2['V'][0])
        np.testing.assert_equal(gexc_1['V'][0], gexc_3['V'][0])

        self.assertFalse(np.array_equal(gexc_1['ge'], gexc_2['ge']))
        self.assertFalse(np.array_equal(gexc_1['gi'], gexc_2['gi']))
        self.assertFalse(np.array_equal(gexc_1['V'], gexc_2['V']))

        self.assertFalse(np.array_equal(gexc_1['ge'], gexc_3['ge']))
        self.assertFalse(np.array_equal(gexc_1['gi'], gexc_3['gi']))
        self.assertFalse(np.array_equal(gexc_1['V'], gexc_3['V']))

       

if __name__ == '__main__':
    unittest.main()
