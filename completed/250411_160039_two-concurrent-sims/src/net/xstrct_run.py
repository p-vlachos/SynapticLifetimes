import argparse

import git
import pypet.pypetexceptions as pex

# from .workarounds import fix_pypet_macos
from .explored_params import explore_dict, name
from .xstrct_module import main

parser = argparse.ArgumentParser()
parser.add_argument("--ncores", "-c", help="No. cores", nargs=1)
parser.add_argument("--testrun", "-t", action='store_true')
parser.add_argument("--postprocess", "-x", action='store_true')
parser.add_argument("--gpu", default=False, action='store_true')
args = parser.parse_args()
# control the number of cores to be used for computation
ncores = int(args.ncores[0])

print("Using {:d} cores".format(ncores))

# fix_pypet_macos()

# check the state of the git repository
# repo = git.Repo('./src/')
repo = git.Repo('../../')

commit = None
if not args.testrun:
    # check for changes, while ignoring submodules
    if repo.git.status('-s', '--ignore-submodules', '--untracked-files=no'):
        raise pex.GitDiffError('Found not committed changes!')

    commit = repo.commit(None)

main(
    name=name,
    postprocess=args.postprocess,
    ncores=ncores,
    testrun=args.testrun,
    explore_dict=explore_dict,
    commit=commit,
    gpu=args.gpu
)