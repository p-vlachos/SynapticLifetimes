import numpy as np
from brian2 import *
from scipy.stats import norm


def simulate_kesten(params = None, N=100, random_seed=43, method='heun'):
    prefs.codegen.target = 'cython'
    if params is None:
        params = dict()
    start_scope()
    seed(random_seed)
    params_ = dict(
        syn_kesten_mu_epsilon_1 = -0.015 / (1 * second),
        syn_kesten_mu_eta = 0.0020 / (1 * second),
        syn_kesten_var_epsilon_1 = 0.000005 / (1 * second),
        syn_kesten_var_eta =  0.000000005 / (1 * second),
        a_init = 0.0026,
        T=10*second,
        dt=1000*ms,
        amin=None,
    )
    params_.update(params)
    G = NeuronGroup(N=N, dt=params_['dt'], namespace=params_, method=method,
        model='''
            da/dt = ( 
                  (syn_kesten_mu_epsilon_1 * a + syn_kesten_mu_eta)
                  + (syn_kesten_var_epsilon_1 * a**2 + syn_kesten_var_eta)**0.5 * xi_kesten
                ) : 1
        ''')
    G.a = params_['a_init']
    if params_['amin'] is not None:
        G.run_regularly("a = clip(a, amin, 10**1.0)", params_['dt'], when='end')
    run(params_['T'])
    return G


def kesten_diff_second(G):
    a1 = np.array(G.a)
    store()
    run(1*second)
    a2 = np.array(G.a)
    restore()

    dw = a2 - a1
    return dw


def plot_kesten_diff_second(ax, dw):
    ax.hist(dw, bins=100, density=True)
    ax.set_xlabel("weight change")
    ax.set_title("weight change due to kesten over 1 second")


def plot_kesten(ax, Glongmany, N_e = 1600, print_fit=True, plot_fit=True, color="blue", label="kesten weights"):
    from scipy.stats import norm
    thres = 10**-2.0
    weights = np.log10(Glongmany.a[Glongmany.a > thres])
    ax.hist(weights, bins=50, density=True, label=label, histtype="step", color=color)
    ax.set_xlabel("log(weights)")
    if plot_fit:
        floc, fscale = norm.fit(weights)
        f_rv = norm(loc=floc, scale=fscale)
        xs = np.linspace(np.min(weights), np.max(weights), num=100)
        ax.plot(xs, f_rv.pdf(xs), color="green", label="lognormal kesten fit")
        if print_fit:
            print("mu", floc, "sigma", fscale)
            print("avg", np.sum(Glongmany.a) / (N_e))
            print("min", np.min(weights), "max", np.max(weights))

