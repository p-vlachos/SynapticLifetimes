
import unittest, os, time
from brian2.units import second
import numpy as np

# get tested
from methods.process_turnover_pd import extract_lifetimes


class Test_extract_lifetimes(unittest.TestCase):

    def test_undying_synapses_added_to_wthsrv_but_not_dthonly(self):
        t_cut, Tmax = 10*second, 100*second
        turnover_data = np.array([[1, 20.85, 0, 0]])

        lts_wthsrv, lts_dthonly, _ = extract_lifetimes(turnover_data, 10,
                                                    t_cut, Tmax)

        self.assertEqual(lts_wthsrv, [Tmax/second-20.85])
        self.assertEqual(lts_dthonly, [])


    def test_synases_below_grown_pre_tcut_are_not_inclded(self):
        t_cut, Tmax = 10*second, 100*second
        turnover_data = np.array([[1, 2.85, 0, 0],
                                  [0, 7.85, 0, 0],
                                  [0, 0.85, 1, 0],
                                  [1, -50.85, 0, 12]])

        lts_wthsrv, lts_dthonly, _ = extract_lifetimes(turnover_data, 10,
                                                    t_cut, Tmax)

        self.assertEqual(lts_wthsrv, [])
        self.assertEqual(lts_dthonly, [])


    def test_synpase_first_died_started_but_did_not_die_again(self):
        t_cut, Tmax = 10*second, 100*second
        turnover_data = np.array([[0, 25.85, 0, 0],
                                  [1, 32.85, 0, 0]])

        lts_wthsrv, lts_dthonly, _ = extract_lifetimes(turnover_data, 10,
                                                    t_cut, Tmax)

        self.assertEqual(lts_wthsrv, [Tmax/second-32.85])
        self.assertEqual(lts_dthonly, [])


    def test_even_number_growth_death_events(self):
        t_cut, Tmax = 10*second, 100*second
        turnover_data = np.array([[0, 25.85, 0, 0],
                                  [1, 32.85, 0, 0],
                                  [0, 38.85, 0, 0],
                                  [1, 82.85, 0, 0],
                                  [0, 88.85, 0, 0]])

        lts_wthsrv, lts_dthonly, _ = extract_lifetimes(turnover_data, 10,
                                                    t_cut, Tmax)

        self.assertEqual(lts_wthsrv, [6.,6.])
        self.assertEqual(lts_dthonly, [6.,6.])


    def test_odd_number_growth_death_events(self):
        t_cut, Tmax = 10*second, 100*second
        turnover_data = np.array([[0, 25.85, 0, 0],
                                  [1, 32.85, 0, 0],
                                  [0, 38.85, 0, 0],
                                  [1, 82.85, 0, 0],
                                  [0, 88.85, 0, 0],
                                  [1, 91.02, 0, 0]])

        lts_wthsrv, lts_dthonly, _ = extract_lifetimes(turnover_data, 10,
                                                    t_cut, Tmax)

        np.testing.assert_array_almost_equal(lts_wthsrv, [6.,6.,8.98])
        np.testing.assert_array_almost_equal (lts_dthonly, [6.,6.])

         
    def test_turnover_data_speed(self):
        t_split, t_cut, bin_w = 4*second, 2*second, 1*second
        turnover_data = np.loadtxt('test/test_sets/turnover_test_set1',
                                      delimiter=',')

        a = time.time()
        lts, dts, _ = extract_lifetimes(turnover_data, 1000,
                                        t_split, t_cut)
        b = time.time()

        print('Test Set 1 took :', b-a, ' s')
        


if __name__ == '__main__':
    unittest.main()
