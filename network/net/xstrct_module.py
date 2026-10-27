import matplotlib
matplotlib.use('Agg')

import glob, os, random, traceback
from pypet import Environment
from .add_parameters import add_params
from .xstrct_netw import run_net
from .post_processing import post_process


def main(name, explore_dict, postprocess=False, ncores=1, testrun=False, commit=None,
         uncommitted_changes=False):

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
        # the changes are saved in uncommitted_changes.diff of the run folder
        tr.f_add_parameter('mconfig.git.uncommitted_changes', uncommitted_changes)

    tr.f_explore(explore_dict)

    def run_sim(tr):
        try:
            run_net(tr)
        except TimeoutError:
            print("Unable to plot, must run analysis manually")

        post_process(tr)

    def mark_failures(run_function):
        """ pypet only logs the errors of single runs (in their own processes), so failed
            runs leave a file failed_run_<idx>.txt with the traceback """
        def run(tr):
            try:
                run_function(tr)
            except BaseException:
                with open('failed_run_%.4d.txt' % tr.v_idx, 'w') as f:
                    f.write(traceback.format_exc())
                raise
        return run

    for fname in glob.glob('failed_run_*.txt'):
        os.remove(fname)

    env.run(mark_failures(post_process if postprocess else run_sim))

    # the failed runs
    return sorted(glob.glob('failed_run_*.txt'))






