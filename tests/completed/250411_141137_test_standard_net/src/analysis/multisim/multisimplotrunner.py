# make these settings and imports available to all importers of this module
import matplotlib
import matplotlib.pyplot as pl
from matplotlib import rc

import argparse
from os import path
import os
import pickle
from typing import List


class MultiSimPlotRunner:
    """
        Base class for scripts that want to plot graphs from multiple simulations (i.e. directories in ``builds/``), even
        if they come from different runs (created with ``./run.sh``).

        Usage (cf. ``srvprb_branching.py`` as an example):

        1. create a new script file
        2. create a subclass of this class

            a) overwrite the :meth:`__init__` function and set the parameters
            b) reimplement the :meth:`plot` function

        3. put the following snippet in the end of the script::

            if __name__ == '__main__':
                MyPlotClass().run()

        The script can then be run in two ways:

        1. without parameters, then it assumes to be executed in the directory of a run
        2. with a list of build directories (e.g. ``200515_212342_my_sim/builds/0000``)
           make use of BASH globs::

            ./my_renderer.py 200515_*_my_sim/builds/000{0,2,5}

        The final PNG file contains the list of directories (builds) that were used to
        create the plot in its metadata. On Unix with ImageMagick it can be extracted like this::

            identify -verbose <file name>.png | grep directories

    """

    def __init__(self, name, plot_count, use_directories=True, in_notebook=False):
        self.name = name
        self.plot_count = plot_count
        self.directories = []
        self.use_directories = use_directories

        if not in_notebook:
            matplotlib.use('Agg')

            rc('text', usetex=True)
            pl.rcParams['text.latex.preamble'] = [
                r'\usepackage{tgheros}',
                r'\usepackage{sansmath}',
                r'\sansmath'
                r'\usepackage{siunitx}',
                r'\sisetup{detect-all}',
            ]

    def _prepare(self):
        if self.use_directories:
            if self.directories is None or len(self.directories) == 0:
                raise Exception(f"No build directories given")
            for dir in self.directories:
                if not path.exists(dir):
                    raise Exception(f"directory {dir} does not exist")

    def _call_plot(self, directories):
        nsps = self._get_nsps(directories)
        fig, axs = pl.subplots(*self.plot_count, squeeze=False)
        # TODO solve usage of fig better to allow wrapping like SrvPrbBranchingWeights
        self.plot(directories, nsps, fig, axs)

    def _get_nsps(self, directories):
        nsps = []
        for bpath in directories:
            with open(bpath + '/raw/namespace.p', 'rb') as pfile:
                nsps.append(pickle.load(pfile))
        return nsps

    def plot(self, directories: List[str], nsps, fig, axs: List[List[pl.Axes]]) -> None:
        """ Reimplement this function to plot

        :param directories: is a list of the build directory paths
        :param nsps: is the trajectories of the builds, the list corresponds with the list 'directories'
        :params fig: fig as returned by pl.subplots
        :param axs: a matrix of the subplots, generated with pl.subplots

        The reimplementation does not need to save the figure.
        In order to load data one can use the :meth:`unpickle` function.
        """
        raise NotImplementedError()

    def unpickle(self, dir, raw_name):
        """ Load a pickle file called `raw_name` from the build at `dir` """
        with open(f'{dir}/raw/{raw_name}.p', 'rb') as handle:
            return pickle.load(handle)

    def _finalize(self):
        pl.tight_layout()

        directory = f"figures_multisim"
        if not os.path.exists(directory):
            os.makedirs(directory)

        metadata = {
            "directories": " ".join(self.directories)
        }

        metadata.update(self._metadata())

        pl.savefig(directory + "/{:s}.png".format(self.name),
                   dpi=200, bbox_inches='tight',
                   metadata=metadata)

    def _add_arguments(self, parser: argparse.ArgumentParser):
        pass

    def _process_args(self, args):
        pass

    def _metadata(self):
        return {}

    def run(self):
        parser = argparse.ArgumentParser()
        if self.use_directories:
            parser.add_argument("directories", type=str, nargs='*', help="paths to build directories")
        self._add_arguments(parser)
        args = parser.parse_args()
        self._process_args(args)

        if self.use_directories:
            self.directories = args.directories

            if self.directories is None or len(self.directories) == 0:
                if not os.path.exists("./builds/"):
                    raise Exception("Couldn't find ./builds/ - please either provide paths to build/000x directories"
                                    " or run this script in the output directory of a simulation run.")
                self.directories = [f'builds/{dir}' for dir in next(os.walk("./builds/"))[1]]

        self._prepare()
        self._call_plot(self.directories)
        self._finalize()
