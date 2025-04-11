from brian2 import *
import matplotlib.pyplot as plt
import scipy.stats as stats


def plot(Vt=-50*mV, mu=8.0*mV, sigma=0.0*mV, T=100*ms):
    p = dict(Vt=Vt, mu=mu, sigma=sigma)
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(121)
    ax.set_title("membrane noise")
    treshold(ax, T=T, **p)
    ax = fig.add_subplot(122)
    ax.set_title("membrane noise distribution vs threshold")
    gauss(ax, **p)


def treshold(ax: plt.Axes, Vt=-50*mV, mu=8.0*mV, sigma=0.0*mV, T=100*ms):
    start_scope()
    El = -60*mV
    tau = 20*ms
    G = NeuronGroup(N=1, model="""
        dV/dt = (El-V)/tau +  mu/tau + (sigma * xi) / (tau **.5) : volt
    """, threshold="V >= Vt", reset="V = El", method="euler")
    G.V = El

    M = StateMonitor(G, True, record=True)

    run(T)

    ax.set_ylim(-60, -49.5)
    ax.hlines(Vt/mV, xmin=0, xmax=T/ms, color="red", label="firing threshold")
    ax.hlines((El+mu)/mV, xmin=0, xmax=T/ms, color="green", ls="-", label="$\mu$")
    ax.plot(M.t/ms, M.V.T/mV, label="membrane potential")
    ax.set_ylabel("V [mV]")
    ax.set_xlabel("t [ms]")
    ax.legend(loc="lower right")


def gauss(ax: plt.Axes, Vt=-50*mV, mu=8.0*mV, sigma=0.0*mV):
    El = -60*mV
    norm = stats.norm(loc=El + mu, scale=sigma)
    xs = np.linspace(*(np.array([-1, 1])*5*sigma + El+mu))
    ys = norm.pdf(xs)

    percentage_thres = 1-norm.cdf(Vt)

    ax.plot(xs/mV, ys)
    ax.vlines(Vt/mV, ymin=-1, ymax=np.max(ys), color="red")
    ax.text(Vt/mV, np.max(ys)*2/3,
            f"P(V> Vt) = {percentage_thres*100:0.2f} %",
            horizontalalignment='right',
            bbox=dict(facecolor="white"))
    ax.set_xlabel("membrane potential [mV]")
    ax.set_xlim(El/mV, (Vt+1*mV)/mV)


def sigma_for_mus(ax: plt.Axes, Vt, mus, q):
    El = -60*mV
    sigmas = ((Vt - El) - mus) / stats.norm.isf(q)
    ax.plot(mus/mV, sigmas/mV, linestyle="None", marker="x", ms=15)
    ax.set_ylabel("$\\sigma$ [mV]")
    ax.set_xlabel("$\\mu$ [mV]")

    return mus, sigmas


