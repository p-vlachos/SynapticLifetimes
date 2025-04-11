
import unittest, os, time
from brian2.units import second
import numpy as np

# gets tested
from methods.process_turnover_statistics import get_insert_and_prune_counts


class Test_get_insert_and_prune_counts(unittest.TestCase):

    def test_events_outside_tcut_Tmax_are_not_considered(self):
        t_cut, Tmax = 5*second, 20*second
        turnover_data = np.array([[1, 2.85, 0, 0],
                                  [0, 7.85, 0, 0],
                                  [0, 9.85, 1, 0],
                                  [1, 15.85, 0, 3],
                                  [1, 20.85, 0, 3],])

        ins_c, prn_c = get_insert_and_prune_counts(turnover_data,
                                                   5, t_cut, Tmax)

        np.testing.assert_array_equal(ins_c, [0,0,0,1,0])
        np.testing.assert_array_equal(prn_c, [2,0,0,0,0])


    # def test_data(self):
    #     t_cut, Tmax = 2*second, 4*second
        
    #     turnover_data = np.loadtxt('test/test_sets/turnover_test_set1',
    #                                   delimiter=',')
    #     print(get_insert_and_prune_counts(turnover_data, 1000, t_cut, Tmax))
