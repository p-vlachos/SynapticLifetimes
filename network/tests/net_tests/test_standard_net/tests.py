
import unittest, os, pickle

import numpy as np

from analysis.utils import WeightSnapshots


class Test_dynamic_synaptic_weights(unittest.TestCase):

    
    synee_a = WeightSnapshots('builds/0000/raw/synee_a.p')

    def test_final_weights_different_from_initial(self):

        self.assertFalse(np.allclose(
                         self.synee_a.dense(-1)[0],
                         self.synee_a.dense( 0)[0]))
    

if __name__ == '__main__':
    unittest.main()
