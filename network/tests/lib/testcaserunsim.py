import unittest, os, shutil


class TestCaseRunSim(unittest.TestCase):

    explore_dict = None

    @classmethod
    def setUpClass(cls) -> None:
        if not bool(os.environ.get("EIB_STANDALONE_TEST")):
            from net.xstrct_module import main
            failed_runs = main(name=cls.__name__, explore_dict=cls.explore_dict, testrun=True)
            if failed_runs:
                with open(failed_runs[0]) as f:
                    raise RuntimeError(f"simulation failed ({failed_runs[0]}):\n{f.read()}")
        cls.setUpClassAfterSim()

    @classmethod
    def setUpClassAfterSim(cls):
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        for d in ["builds/", "data/", "figures/", "logs/"]:
            shutil.rmtree(d)