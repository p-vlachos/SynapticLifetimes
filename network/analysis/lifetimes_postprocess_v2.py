import os
import pickle
from argparse import Namespace

from net.post_processing import post_process_turnover

if __name__ == '__main__':
    build_dirs = sorted(['builds/'+pth for pth in next(os.walk("builds/"))[1]])

    for bpath in build_dirs:
        print(f"{bpath[-4:]}...", end="")
        try:
            with open(bpath + '/raw/namespace.p', 'rb') as pfile:
                nsp = pickle.load(pfile)
        except FileNotFoundError:
            print(bpath[-4:], "reports: No namespace data. Skipping.")
            continue

        tr = Namespace(v_idx=nsp['idx'], **nsp)  # fake ourselves a Trajectory
        post_process_turnover(tr)
        print("done")
