"""
Runs all parameter sets of net/explored_params.py with pypet, several in parallel:
the simulation (xstrct_netw.run_net) and its post-processing (post_processing.py).
Called by run.sh from the run folder:

    python -m net.run [-n N] [--postprocess]

The results of parameter set k are in builds/<k>/, the pypet trajectory in data/<name>.hdf5.
"""
import argparse
import glob
import os
import random
import sys
import traceback

import matplotlib
matplotlib.use('Agg')
from pypet import Environment

from .add_parameters import add_params
from .xstrct_netw import run_net
from .post_processing import post_process


def simulate(tr):
    run_net(tr)
    post_process(tr)


def main(name, explore_dict, ncores=1, postprocess=False):
    """
        Runs the simulations of explore_dict (or only their post-processing), ncores at a time.
        Returns the files failed_run_<k>.txt with the errors of the failed simulations.
    """
    # pypet only logs the errors of the single runs (in their own processes), so each failed
    # run writes its error to a file
    function = post_process if postprocess else simulate

    def run(tr):
        try:
            function(tr)
        except BaseException:
            with open('failed_run_%.4d.txt' % tr.v_idx, 'w') as f:
                f.write(traceback.format_exc())
            raise

    for fname in glob.glob('failed_run_*.txt'):
        os.remove(fname)

    # a new trajectory for each post-processing, in the same file
    label = 'tr1'
    if postprocess:
        label += '_postprocess-%.6d' % random.randint(0, 999999)

    env = Environment(trajectory=label,
                      add_time=False,
                      filename=os.path.join(os.getcwd(), 'data', name + '.hdf5'),
                      multiproc=True,
                      ncores=ncores,
                      use_pool=False,      # a new process for each run
                      wrap_mode='QUEUE',   # one process writes the HDF5 file
                      overwrite_file=False)
    tr = env.trajectory
    add_params(tr)
    tr.f_explore(explore_dict)

    env.run(run)

    return sorted(glob.glob('failed_run_*.txt'))


if __name__ == '__main__':
    from .explored_params import explore_dict, name

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=None,
                        help='number of simulations run in parallel (default: all parameter '
                             'sets, at most the number of available cores)')
    parser.add_argument('--postprocess', action='store_true',
                        help='only run the post-processing of finished simulations')
    args = parser.parse_args()

    n_sets = len(next(iter(explore_dict.values())))
    ncores = args.n or min(n_sets, len(os.sched_getaffinity(0)))
    print(f"{n_sets} parameter set(s), {ncores} at a time")

    failed = main(name, explore_dict, ncores, args.postprocess)
    if failed:
        print(f"ERROR: {len(failed)} of {n_sets} simulation(s) failed, see {', '.join(failed)}")
        sys.exit(1)
