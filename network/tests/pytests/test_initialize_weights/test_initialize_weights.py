
import unittest, pickle
import numpy as np
import pytest

from tests.lib.testcaserunsim import TestCaseRunSim
from analysis.utils import WeightSnapshots

import explored_params


@pytest.mark.network
class Test_weight_matrix_equals_initial(TestCaseRunSim):
    """
    STDP/iSTDP and other weight changes are disabled,
    weight matrix should equal the initial weights
    """

    explore_dict = explored_params.explore_dict

    @classmethod
    def setUpClassAfterSim(cls)-> None:
        snapshots = WeightSnapshots('builds/0000/raw/synee_a.p')
        cls.synee_a = {'a': np.array([snapshots.dense(k)[0] for k in range(len(snapshots))])}

    def test_min_weight_equals_zero(self):
        self.assertEqual(np.min(self.synee_a['a']),0.)

    @unittest.skip("the delta should not be necessary")
    def test_max_weight_equals_aEE(self):
        self.assertAlmostEqual(np.max(self.synee_a['a']), 0.005, delta=0.0001)


if __name__ == '__main__':
    unittest.main()
