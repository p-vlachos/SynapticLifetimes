
import unittest, os, pickle

import numpy as np


class Test_EE_weights_equal_loaded_weights(unittest.TestCase):
    '''
    STDP/iSTDP and other weight changes are disabled,
    weight matrix should equal the initial weights
    '''

    def setUp(self):

        self.bpath = "builds/0000"

        with open(self.bpath+'/raw/namespace.p', 'rb') as pfile:
            self.nsp=pickle.load(pfile)

        with open(self.bpath+'/raw/synee_a.p', 'rb') as pfile:
            self.synee_a = pickle.load(pfile)

        fpath = os.path.join(self.nsp['basepath'],
                             self.nsp['weight_path'])

        with open(fpath+'synee_a.p', 'rb') as pfile:
            self.synee_a_init = pickle.load(pfile)


    def test_shapes_EE_weights_match_shapes_loaded(self):

        self.assertEqual(np.shape(self.synee_a_init['a'][-1,:]),
                         np.shape(self.synee_a['a'][-1,:]))
            

    def test_EE_weights_equal_loaded_EE_weights(self):

        np.testing.assert_array_equal(self.synee_a_init['a'][-1,:],
                                      self.synee_a['a'][-1,:])


    def test_EE_syn_active_equal_loaded_EE_syn_active(self):

        np.testing.assert_array_equal(
            self.synee_a_init['syn_active'][-1,:],
            self.synee_a['syn_active'][-1,:])


        

class Test_EI_weights_equal_loaded_weights(unittest.TestCase):
    '''
    STDP/iSTDP and other weight changes are disabled,
    weight matrix should equal the initial weights
    '''

    def setUp(self):

        self.bpath = "builds/0000"

        with open(self.bpath+'/raw/namespace.p', 'rb') as pfile:
            self.nsp=pickle.load(pfile)

        with open(self.bpath+'/raw/synei_a.p', 'rb') as pfile:
            self.synei_a = pickle.load(pfile)

        fpath = os.path.join(self.nsp['basepath'],
                             self.nsp['weight_path'])

        with open(fpath+'synei_a.p', 'rb') as pfile:
            self.synei_a_init = pickle.load(pfile)


    def test_shapes_EI_weights_match_shapes_loaded(self):

        self.assertEqual(np.shape(self.synei_a_init['a'][-1,:]),
                         np.shape(self.synei_a['a'][-1,:]))
            

    def test_EI_weights_equal_loaded_EI_weights(self):

        np.testing.assert_array_equal(self.synei_a_init['a'][-1,:],
                                      self.synei_a['a'][-1,:])


    def test_EI_syn_active_equal_loaded_EI_syn_active(self):

        np.testing.assert_array_equal(
            self.synei_a_init['syn_active'][-1,:],
            self.synei_a['syn_active'][-1,:])

       

    

if __name__ == '__main__':
    unittest.main()
