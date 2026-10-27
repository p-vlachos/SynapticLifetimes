import argparse
import os

import git

from .explored_params import explore_dict, name
from .xstrct_module import main

parser = argparse.ArgumentParser()
parser.add_argument("--ncores", "-c", type=int, default=None,
                    help="number of simulations run in parallel (default: all parameter sets, "
                         "at most the number of available cores)")
parser.add_argument("--testrun", "-t", action='store_true')
parser.add_argument("--postprocess", "-x", action='store_true')
args = parser.parse_args()

# number of simulations run in parallel
n_parameter_sets = len(next(iter(explore_dict.values())))
ncores = args.ncores or min(n_parameter_sets, len(os.sched_getaffinity(0)))
print("{:d} parameter set(s), {:d} run in parallel".format(n_parameter_sets, ncores))

commit, uncommitted_changes = None, False
if not args.testrun:
    # record the state of the git repository; simulations run from
    # running/<timestamp>_<name>/, which lives inside the repository
    repo = git.Repo('.', search_parent_directories=True)
    commit = repo.commit(None)

    # uncommitted changes of tracked files (ignoring submodules) are saved
    # in the run folder, so that the run can be reproduced
    diff = repo.git.diff('HEAD', '--ignore-submodules')
    if diff:
        uncommitted_changes = True
        with open('uncommitted_changes.diff', 'w') as f:
            f.write(diff + '\n')
        print("WARNING: uncommitted changes, saved in uncommitted_changes.diff "
              "(relative to commit {})".format(commit.hexsha[:7]))

main(
    name=name,
    postprocess=args.postprocess,
    ncores=ncores,
    testrun=args.testrun,
    explore_dict=explore_dict,
    commit=commit,
    uncommitted_changes=uncommitted_changes
)
