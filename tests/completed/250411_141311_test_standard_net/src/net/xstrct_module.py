import matplotlib
matplotlib.use('Agg')

import os, random
from pypet import Environment
from .add_parameters import add_params
from .xstrct_netw import run_net
from .post_processing import post_process


def main(name, explore_dict, postprocess=False, ncores=1, testrun=False, commit=None, gpu=False):

    if not testrun:
        if commit is None:
            raise Exception("Non testrun needs a commit")

    filename = os.path.join(os.getcwd(), 'data/', name + '.hdf5')

    # if not the first run, tr2 will be merged later
    label = 'tr1'

    # if only post processing, can't use the same label
    # (generates HDF5 error)
    if postprocess:
        label += '_postprocess-%.6d' % random.randint(0, 999999)

    env = Environment(trajectory=label,
                      add_time=False,
                      filename=filename,
                      continuable=False,  # ??
                      lazy_debug=False,  # ??
                      multiproc=True,
                      ncores=ncores,
                      use_pool=False,  # likely not working w/ brian2
                      wrap_mode='QUEUE',  # ??
                      overwrite_file=False)

    tr = env.trajectory

    add_params(tr)

    if not testrun:
        tr.f_add_parameter('mconfig.git.sha1', str(commit))
        tr.f_add_parameter('mconfig.git.message', commit.message)

    tr.f_explore(explore_dict)

    def run_sim(tr):
        try:
            run_net(tr, gpu=gpu)
        except TimeoutError:
            print("Unable to plot, must run analysis manually")

        post_process(tr)

    if postprocess:
        env.run(post_process)
    else:
        env.run(run_sim)






