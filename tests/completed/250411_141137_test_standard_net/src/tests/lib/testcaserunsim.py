import unittest, os, shutil


class TestCaseRunSim(unittest.TestCase):

    explore_dict = None

    @classmethod
    def setUpClass(cls) -> None:
        if not bool(os.environ.get("EIB_STANDALONE_TEST")):
            from net.xstrct_module import main
            main(name=cls.__name__, explore_dict=cls.explore_dict, testrun=True)
        cls.setUpClassAfterSim()

    @classmethod
    def setUpClassAfterSim(cls):
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        for d in ["builds/", "data/", "figures/", "logs/"]:
            shutil.rmtree(d)