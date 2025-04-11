
import unittest, os, pickle

import numpy as np


class Test_same_seed_produces_same_results(unittest.TestCase):
    '''
    '''

    def setUp(self):

        self.bpath_0 = "builds/0000"
        self.bpath_1 = "builds/0001"
        self.bpath_2 = "builds/0002"

        
    def test_builds_have_expected_seeds(self):

        with open(self.bpath_0+'/raw/namespace.p', 'rb') as pfile:
            nsp_0=pickle.load(pfile)

        with open(self.bpath_1+'/raw/namespace.p', 'rb') as pfile:
            nsp_1=pickle.load(pfile)

        with open(self.bpath_2+'/raw/namespace.p', 'rb') as pfile:
            nsp_2=pickle.load(pfile)

        self.assertEqual(nsp_0['random_seed'], nsp_1['random_seed'])

        self.assertNotEqual(nsp_1['random_seed'],
                            nsp_2['random_seed'])


    def test_membrane_potential_as_expected(self):

        with open(self.bpath_0+'/raw/gexc_stat.p', 'rb') as pfile:
            gexc_0=pickle.load(pfile)

        with open(self.bpath_1+'/raw/gexc_stat.p', 'rb') as pfile:
            gexc_1=pickle.load(pfile)

        with open(self.bpath_2+'/raw/gexc_stat.p', 'rb') as pfile:
            gexc_2=pickle.load(pfile)
        
        np.testing.assert_equal(gexc_0['t'], gexc_1['t'])
        
        np.testing.assert_equal(gexc_0['ge'], gexc_1['ge'])
        np.testing.assert_equal(gexc_0['gi'], gexc_1['gi'])
        np.testing.assert_equal(gexc_0['V'], gexc_1['V'])

        self.assertFalse(np.array_equal(gexc_1['ge'], gexc_2['ge']))
        self.assertFalse(np.array_equal(gexc_1['gi'], gexc_2['gi']))
        self.assertFalse(np.array_equal(gexc_1['V'], gexc_2['V']))
        

    def test_exc_spikes_as_expected(self):

        with open(self.bpath_0+'/raw/gexc_spks.p', 'rb') as pfile:
            gexc_spks_0=pickle.load(pfile)

        with open(self.bpath_1+'/raw/gexc_spks.p', 'rb') as pfile:
            gexc_spks_1=pickle.load(pfile)

        with open(self.bpath_2+'/raw/gexc_spks.p', 'rb') as pfile:
            gexc_spks_2=pickle.load(pfile)

        np.testing.assert_equal(gexc_spks_0['t'], gexc_spks_1['t'])
        np.testing.assert_equal(gexc_spks_0['i'], gexc_spks_1['i'])
        
        self.assertFalse(np.array_equal(gexc_spks_1['t'],
                                        gexc_spks_2['t']))
        self.assertFalse(np.array_equal(gexc_spks_1['i'],
                                        gexc_spks_2['i']))
        

    def test_inh_spikes_as_expected(self):

        with open(self.bpath_0+'/raw/ginh_spks.p', 'rb') as pfile:
            ginh_spks_0=pickle.load(pfile)

        with open(self.bpath_1+'/raw/ginh_spks.p', 'rb') as pfile:
            ginh_spks_1=pickle.load(pfile)

        with open(self.bpath_2+'/raw/ginh_spks.p', 'rb') as pfile:
            ginh_spks_2=pickle.load(pfile)

        np.testing.assert_equal(ginh_spks_0['t'], ginh_spks_1['t'])
        np.testing.assert_equal(ginh_spks_0['i'], ginh_spks_1['i'])
        
        self.assertFalse(np.array_equal(ginh_spks_1['t'],
                                        ginh_spks_2['t']))
        self.assertFalse(np.array_equal(ginh_spks_1['i'],
                                        ginh_spks_2['i']))


    def test_EE_synapse_traces_as_expected(self):

        with open(self.bpath_0+'/raw/synee_stat.p', 'rb') as pfile:
            synee_stat_0=pickle.load(pfile)

        with open(self.bpath_1+'/raw/synee_stat.p', 'rb') as pfile:
            synee_stat_1=pickle.load(pfile)

        with open(self.bpath_2+'/raw/synee_stat.p', 'rb') as pfile:
            synee_stat_2=pickle.load(pfile)

        np.testing.assert_equal(synee_stat_0['a'], synee_stat_1['a'])
        np.testing.assert_equal(synee_stat_0['Apre'],
                                synee_stat_1['Apre'])
        np.testing.assert_equal(synee_stat_0['Apost'],
                                synee_stat_1['Apost'])

        self.assertFalse(np.array_equal(synee_stat_1['a'],
                                        synee_stat_2['a']))
        self.assertFalse(np.array_equal(synee_stat_1['Apre'],
                                        synee_stat_2['Apre']))
        self.assertFalse(np.array_equal(synee_stat_1['Apost'],
                                        synee_stat_2['Apost']))

        
    def test_EI_synapse_traces_as_expected(self):

        with open(self.bpath_0+'/raw/synei_stat.p', 'rb') as pfile:
            synei_stat_0=pickle.load(pfile)

        with open(self.bpath_1+'/raw/synei_stat.p', 'rb') as pfile:
            synei_stat_1=pickle.load(pfile)

        with open(self.bpath_2+'/raw/synei_stat.p', 'rb') as pfile:
            synei_stat_2=pickle.load(pfile)
  
        np.testing.assert_equal(synei_stat_0['a'], synei_stat_1['a'])
        np.testing.assert_equal(synei_stat_0['Apre'],
                                synei_stat_1['Apre'])
        np.testing.assert_equal(synei_stat_0['Apost'],
                                synei_stat_1['Apost'])
       
        self.assertFalse(np.array_equal(synei_stat_1['a'],
                                        synei_stat_2['a']))
        self.assertFalse(np.array_equal(synei_stat_1['Apre'],
                                        synei_stat_2['Apre']))
        self.assertFalse(np.array_equal(synei_stat_1['Apost'],
                                        synei_stat_2['Apost']))



    def test_EE_weights_as_expected(self):

        with open(self.bpath_0+'/raw/synee_a.p', 'rb') as pfile:
            synee_a_0=pickle.load(pfile)

        with open(self.bpath_1+'/raw/synee_a.p', 'rb') as pfile:
            synee_a_1=pickle.load(pfile)

        with open(self.bpath_2+'/raw/synee_a.p', 'rb') as pfile:
            synee_a_2=pickle.load(pfile)

        np.testing.assert_equal(synee_a_0['t'], synee_a_1['t'])
        np.testing.assert_equal(synee_a_0['a'], synee_a_1['a'])
        np.testing.assert_equal(synee_a_0['syn_active'],
                                synee_a_1['syn_active'])
        
        self.assertFalse(np.array_equal(synee_a_1['a'],
                                        synee_a_2['a']))     
        self.assertFalse(np.array_equal(synee_a_1['syn_active'],
                                        synee_a_2['syn_active']))        

        
    def test_EI_weights_as_expected(self):

        with open(self.bpath_0+'/raw/synei_a.p', 'rb') as pfile:
            synei_a_0=pickle.load(pfile)

        with open(self.bpath_1+'/raw/synei_a.p', 'rb') as pfile:
            synei_a_1=pickle.load(pfile)

        with open(self.bpath_2+'/raw/synei_a.p', 'rb') as pfile:
            synei_a_2=pickle.load(pfile)

        np.testing.assert_equal(synei_a_0['t'], synei_a_1['t'])
        np.testing.assert_equal(synei_a_0['a'], synei_a_1['a'])
        np.testing.assert_equal(synei_a_0['syn_active'],
                                synei_a_1['syn_active'])
        
        self.assertFalse(np.array_equal(synei_a_1['a'],
                                        synei_a_2['a']))     
        # self.assertFalse(np.array_equal(synei_a_1['syn_active'],
        #                                 synei_a_2['syn_active']))        
    

    def test_EE_turnover_as_expected(self):

        with open(self.bpath_0+'/raw/turnover.p', 'rb') as pfile:
            turnover_0=pickle.load(pfile)

        with open(self.bpath_1+'/raw/turnover.p', 'rb') as pfile:
            turnover_1=pickle.load(pfile)

        with open(self.bpath_2+'/raw/turnover.p', 'rb') as pfile:
            turnover_2=pickle.load(pfile)

        np.testing.assert_equal(turnover_0, turnover_1)
        
        self.assertFalse(np.array_equal(turnover_1,
                                        turnover_2))

    def test_EI_turnover_as_expected(self):
        # not enabled by default and not tested here
        pass
        

if __name__ == '__main__':
    unittest.main()
