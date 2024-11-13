import pypet
import numpy as np


def a_ee_init(tr: pypet.Trajectory, initial_active):
    if tr.a_ee_mode == "constant":
        return initial_active * tr.a_ee
    elif tr.a_ee_mode == "lognormal":
        return initial_active * 10 ** np.random.normal(tr.a_ee_init_lognormal_mu,
                                                       tr.a_ee_init_lognormal_sig,
                                                       size=len(initial_active))
    else:
        raise Exception(f"'a_ee_mode' cannot be {tr.a_ee_mode}")
