import unittest

import net.param_tools as param_tools


class ParamToolsTestCase(unittest.TestCase):

    def test_exports_cartesian_product(self):
        self.assertIsNotNone(param_tools.cartesian_product)


class ParamToolsNListTestCase(unittest.TestCase):

    def test_same_n(self):
        input_dict = dict(
            A=["a1", "a2"],
            B=["b1", "b2"],
        )
        explore_dict = param_tools.n_list(input_dict)
        self.assertEqual(
            input_dict, explore_dict
        )

    def test_expands(self):
        input_dict = dict(
            A=["a1", "a2"],
            B=["b1"],
            C=["c1", "c2"],
        )
        explore_dict = param_tools.n_list(input_dict)
        self.assertEqual(
            explore_dict,
            dict(
                A=["a1", "a2"],
                B=["b1", "b1"],
                C=["c1", "c2"],
            )
        )

    def test_throws_not_equal(self):
        input_dict = dict(
            A=["a1", "a2"],
            B=["b1", "b2", "b3"],
        )
        with self.assertRaises(Exception):
            param_tools.n_list(input_dict)

if __name__ == '__main__':
    unittest.main()
