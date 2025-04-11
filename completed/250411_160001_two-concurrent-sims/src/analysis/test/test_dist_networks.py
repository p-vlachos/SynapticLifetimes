
import unittest, os, time
from brian2.units import second
import numpy as np

# get tested
from methods.process_survival import  extract_survival


class Test_extract_survival(unittest.TestCase):


    def test_turnover_data_set1(self):
        t_split, t_cut, bin_w = 5*second, 2*second, 1*second
        turnover_data = [[1, 2.5, 0, 0]]

        full_t, ex_ids = extract_survival(np.array(turnover_data),
                                          bin_w, 10, t_split, t_cut)

        self.assertEqual(len(full_t),1)
        self.assertEqual(full_t[0],t_split/second)
        

    def test_turnover_data_set2(self):
        t_split, t_cut, bin_w = 5*second, 2*second, 1*second
        turnover_data = [[1, 2.5, 0, 0],
                         [0, 2.7, 0, 0],
                         [1, 3.8, 0, 0],
                         [0, 4.592, 0, 0]]

        full_t, ex_ids = extract_survival(np.array(turnover_data),
                                          bin_w, 10, t_split, t_cut)

        np.testing.assert_array_almost_equal(full_t, [0.2, 0.792])

        
    def test_turnover_data_set3(self):
        t_split, t_cut, bin_w = 5*second, 2*second, 1*second
        turnover_data = [[0, 2.3, 0, 0],
                         [1, 2.5, 0, 0],
                         [0, 2.7, 0, 0],
                         [1, 3.8, 0, 0],
                         [0, 4.592, 0, 0]]

        full_t, ex_ids = extract_survival(np.array(turnover_data),
                                          bin_w, 10, t_split, t_cut)

        np.testing.assert_array_almost_equal(full_t, [0.2, 0.792])


    def test_turnover_data_set4(self):
        t_split, t_cut, bin_w = 5*second, 2*second, 1*second
        turnover_data = [[1, 2.5, 0, 0],
                         [0, 2.7, 0, 0],
                         [1, 3.8, 0, 0],
                         [0, 8.6, 0, 0]]

        full_t, ex_ids = extract_survival(np.array(turnover_data),
                                          bin_w, 10, t_split, t_cut)

        # test correct lifetimes
        np.testing.assert_array_almost_equal(full_t, [0.2, 4.8])


    def test_turnover_data_set5(self):
        t_split, t_cut, bin_w = 5*second, 2*second, 1*second
        turnover_data = [[1, 2.5, 0, 0],
                         [0, 2.7, 0, 0],
                         [1, 3.8, 0, 0],
                         [1, 8.6, 0, 0]]

        full_t, ex_ids = extract_survival(np.array(turnover_data),
                                          bin_w, 10, t_split, t_cut)

        # test excluded ids
        self.assertEqual(len(ex_ids), 1)
        self.assertEqual(ex_ids[0], 0)

        
    def test_turnover_data_set6(self):
        t_split, t_cut, bin_w = 5*second, 2*second, 1*second
        turnover_data = [[1, 2.5, 0, 0],
                         [0, 2.7, 0, 0],
                         [1, 3.8, 0, 0]]

        full_t, ex_ids = extract_survival(np.array(turnover_data),
                                          bin_w, 10, t_split, t_cut)

        np.testing.assert_array_almost_equal(full_t, [0.2, t_split/second])

        
    def test_turnover_data_set7(self):
        t_split, t_cut, bin_w = 5*second, 2*second, 1*second
        turnover_data = [[1, 5.9, 0, 0]]

        full_t, ex_ids = extract_survival(np.array(turnover_data),
                                          bin_w, 10, t_split, t_cut)

        np.testing.assert_array_almost_equal(full_t, [5.])


    def test_turnover_data_set8(self):
        t_split, t_cut, bin_w = 5*second, 2*second, 1*second
        turnover_data = [[1, 7.9, 0, 0]]

        full_t, ex_ids = extract_survival(np.array(turnover_data),
                                          bin_w, 10, t_split, t_cut)

        np.testing.assert_array_almost_equal(full_t, [])

        
    # def test_turnover_data_speed(self):
    #     t_split, t_cut, bin_w = 4*second, 2*second, 1*second
    #     turnover_data = np.loadtxt('test/test_sets/turnover_test_set1',
    #                                   delimiter=',')

    #     a = time.time()
    #     full_t, ex_ids = extract_survival(turnover_data,
    #                                       bin_w, 1000,
    #                                       t_split, t_cut)
    #     b = time.time()

    #     print('Test Set 1 took :', b-a, ' s')
        


if __name__ == '__main__':
    unittest.main()
