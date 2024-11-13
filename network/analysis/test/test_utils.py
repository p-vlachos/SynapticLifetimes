
import unittest, os, time, pickle
import numpy as np

import utils


class Test_convert_to_EE_adjacency_matrix(unittest.TestCase):


    def test_constructed_adj_matches_expected_output(self):

        with open('test/test_sets/synee_a_with-ij.p', 'rb') as pfile:
            synee_a = pickle.load(pfile)

        syn_array = synee_a['a'][-1]
        i, j = synee_a['i'], synee_a['j']
        
        adj_m = utils.convert_to_EE_adjacency_matrix(syn_array, 400)

        for k in np.random.choice(range(400*399),1000):
            self.assertEqual(adj_m[i[k],j[k]], syn_array[k]) 


    def test_diagonal_equals_zero(self):

        with open('test/test_sets/synee_a_with-ij.p', 'rb') as pfile:
            synee_a = pickle.load(pfile)

        syn_array = synee_a['a'][-1]
        i, j = synee_a['i'], synee_a['j']
        
        adj_m = utils.convert_to_EE_adjacency_matrix(syn_array, 400)

        for k in range(400):
            self.assertEqual(adj_m[k,k],0) 




class Test_convert_to_EI_adjacency_matrix(unittest.TestCase):


    def test_constructed_adj_matches_expected_output(self):

        with open('test/test_sets/synei_a_with-ij.p', 'rb') as pfile:
            synee_a = pickle.load(pfile)

        syn_array = synee_a['a'][-1]
        i, j = synee_a['i'], synee_a['j']
        
        adj_m = utils.convert_to_EI_adjacency_matrix(syn_array, 400, 80)

        for k in np.random.choice(range(400*80),1000):
            self.assertEqual(adj_m[i[k],j[k]], syn_array[k]) 

            


if __name__ == '__main__':
    unittest.main()        
