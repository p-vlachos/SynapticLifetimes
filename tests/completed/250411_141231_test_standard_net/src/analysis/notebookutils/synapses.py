from brian2 import *
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from notebookutils.analysis import pickle_load


def plot_synapses(voltage_ax, conductance_ax, weight, tau_e=1.7 * ms, tau_e_rise=0.25 * ms, tau=20 * ms, label="",
                  Ee = 0.0*mV,
                  colors=["blue", "cornflowerblue"], plot=True):
    norm_f_EE = 1.0
    El = -60. * mV
    Vr_e = -60. * mV
    Vt_e = -50. * mV
    invpeakEE = (tau_e / tau_e_rise) ** \
                (tau_e_rise / (tau_e - tau_e_rise))

    start_scope()

    E = NeuronGroup(N=2, model='''
        dV/dt = (El-V + ge*(Ee-V))/tau + Ex : volt
        dge/dt = (invpeakEE*xge-ge)/tau_e_rise : 1
        dxge/dt = -xge/tau_e                   : 1
        Ex : volt/second
        El : volt
    ''', threshold="V > Vt_e", reset="V = Vr_e", method='euler', dt=0.1 * ms)
    S = Synapses(E, E, model='''
        a : 1
    ''', on_pre='''
        xge_post += a
    ''')
    S.connect(i=0, j=1)
    S.a = weight

    state = StateMonitor(E, ["V", "ge"], True, dt=.1 * ms)

    E[0].V = E[0].El = El
    E[1].V = E[1].El = -60 * mV

    run(10 * ms)
    E[0].V = -49 * mV
    run(90 * ms)

    if plot:
        voltage_ax.plot(state.t / ms, state.V[1].T / mV, label=f"V {label}", color=colors[0])
        conductance_ax.plot(state.t / ms, state.ge[1].T, label=f"ge {label}", color=colors[1])

    return state.t, state.V[1].T


def plot_synapses_singleexp(voltage_ax, conductance_ax, weight, tau_e=1.7 * ms, tau=20 * ms, label="",
                  Ee = 0.0*mV,
                  colors=["blue", "cornflowerblue"], plot=True):
    norm_f_EE = 1.0
    El = -60. * mV
    Vr_e = -60. * mV
    Vt_e = -50. * mV

    start_scope()

    E = NeuronGroup(N=2, model='''
        dV/dt = (El-V + ge*(Ee-V))/tau + Ex : volt
        dge/dt = -ge/tau_e                   : 1
        Ex : volt/second
        El : volt
    ''', threshold="V > Vt_e", reset="V = Vr_e", method='euler', dt=0.1 * ms)
    S = Synapses(E, E, model='''
        a : 1
    ''', on_pre='''
        ge_post += a
    ''')
    S.connect(i=0, j=1)
    S.a = weight

    state = StateMonitor(E, ["V", "ge"], True, dt=.1 * ms)

    E[0].V = E[0].El = El
    E[1].V = E[1].El = -60 * mV

    run(10 * ms)
    E[0].V = -49 * mV
    run(90 * ms)

    if plot:
        voltage_ax.plot(state.t / ms, state.V[1].T / mV, label=f"V {label}", color=colors[0])
        conductance_ax.plot(state.t / ms, state.ge[1].T, label=f"ge {label}", color=colors[1])

    return state.t, state.V[1].T


def prepare_plot(fig = None, ax1 = None, inhibitory=False):
    if fig is None or ax1 is None:
        fig, ax1 = plt.subplots()
    ax1.xaxis.set_major_locator(MultipleLocator(10))
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.set_xlabel("t [ms]")
    ax1.set_ylabel("V [mV]")
    if inhibitory:
        ax1.set_ylim(-70.0, -59.0)
    else:
        ax1.set_ylim(-60.0, -50.0)
    ax1.set_xlim(0, 100)
    # ax1.grid(True, linewidth=0.5, color='coral', linestyle='-', which='minor')

    ax2 = ax1.twinx()
    # ax2.plot(state.t/ms, (Ee/mV - state.V[1].T/mV)*state.ge[1].T, color="blue", label="(Ee-V)*ge")
    ax2.tick_params(axis='y', labelcolor="cornflowerblue")
    ax2.grid(True, linewidth=0.5, color='cornflowerblue', linestyle='-', which='major')
    ax2.set_ylabel("conductance")

    return fig, ax1, ax2


def plot_last_weight_hist(ax, build, color="blue", label=None, return_weights=False, bins=None):
    if bins is None:
        bins = np.linspace(-1.51, -0.5, 50)
    amin = build[2]['amin']
    syn = pickle_load(build, "synee_a", print_attrs=False)
    indices = syn['a'][-1] > 0
    a = syn['a'][-1][indices]
    label = f"{build[1]}" if label is None else label
    ax.hist(np.log10(a), bins=bins, histtype="step", label=label, density=True, color=color)
    ax.axvline(np.log10(amin), 0, 1, color=color)
    ax.set_xlabel("log weight")
    ax.set_ylabel("density")
    if return_weights:
        return np.log10(a), bins


# TODO properly import from where it is coming from
def generate_full_connectivity(Nsrc, Ntar=0, same=True):
    if same:
        i = []
        j = []
        for k in range(Nsrc):
            i.extend([k]*(Nsrc-1))
            targets = list(range(Nsrc))
            del targets[k]
            j.extend(targets)

        assert len(i)==len(j)
        return np.array(i), np.array(j)

    else:
        i = []
        j = []
        for k in range(Nsrc):
            i.extend([k]*Ntar)
            targets = list(range(Ntar))
            j.extend(targets)

        assert len(i)==len(j)
        return np.array(i), np.array(j)


def find_synapses_active_beginning(build, index=0):
    Ne = 1600  # TODO read in
    i, j = generate_full_connectivity(Ne, same=True)
    # find out which synapse ids where active initially
    syneea = pickle_load(build, "synee_a", print_attrs=False)
    synactive0 = np.full((Ne, Ne), np.nan)
    synactive0[i, j] = syneea['syn_active'][index][:]
    iactive, jactive = np.where(synactive0 == 1.0)
    ids_active = set(np.unique(iactive * 1600 + jactive).astype(int).flatten())
    return ids_active


def get_surviving_weights(build):
    """

    :param build: build information from Analysis
    :return: np.array of weights that survived from beginning of simulation to end
    """
    Ne = 1600  # TODO read in
    i, j = generate_full_connectivity(Ne, same=True)
    # find out which synapse ids where active initially
    syneea = pickle_load(build, "synee_a", print_attrs=False)
    synactive0 = np.full((Ne, Ne), np.nan)
    synactive0[i, j] = syneea['syn_active'][0][:]
    iactive, jactive = np.where(synactive0 == 1.0)
    ids_active = set(np.unique(iactive * 1600 + jactive).astype(int).flatten())
    # print("alive at beginning", len(ids_active), "per neuron", len(ids_active) / 1600)
    # find out which synapses were destroyed over the course of the simulation
    turnover = pickle_load(build, "turnover", print_attrs=False)  # 1 create 0 destroy | t | i | j
    idestroyed, jdestroyed = turnover[turnover[:, 0] == 0.0][:, [2, 3]].T
    ids_destroyed = set(np.unique(idestroyed * 1600 + jdestroyed).astype(int).flatten())
    # print("destroyed over full sim time", len(ids_destroyed), "per neuron", len(ids_destroyed) / 1600)
    # we now have to sets and find what is in ids_active that hasn't been destroyed
    alive_at_end = ids_active - ids_destroyed
    count_alive_at_end = len(alive_at_end)
    percent_alive_at_end = count_alive_at_end/len(ids_active)
    # print("alive at end", count_alive_at_end, "percent", percent_alive_at_end)

    # TODO double check that for these ids there is no creation event

    # find weights at end
    aend = np.full((Ne, Ne), np.nan)
    aend[i, j] = syneea['a'][-1][:]
    np_alive_at_end = np.array(list(alive_at_end))
    jalive = (np_alive_at_end % 1600).astype(int)
    ialive = ((np_alive_at_end  - jalive) / 1600).astype(int)
    surviving_a = aend[ialive, jalive]

    # verify indices
    aactive = np.full((Ne, Ne), np.nan)
    aactive[i, j] = syneea['syn_active'][-1][:]
    surviving_active = aactive[ialive, jalive]
    count_surviving_active = np.sum(surviving_active).astype(int)
    print(build[1], "#surviving active", count_surviving_active, "#surviving indices", len(ialive), "same", count_surviving_active == len(ialive))

    return surviving_a


def h0_same_dist(surviving, baseline):
    import textwrap
    from scipy.stats import chi2_contingency

    p_expected = 0.05
    input = np.array([surviving, baseline])
    # print("input", input.shape, input)
    _, p, dof, ex = chi2_contingency(input)
    description = textwrap.dedent(f"""
        H0: weights are drawn from same distribution
        reject if p < {p_expected}
        p ~ {p:.4f} => {'reject' if p < p_expected else 'accept'}""")
    print(description)


def plot_surviving_weights_vs_weights(ax1, build, print_info=False, color="darkgreen"):
    bins = np.linspace(-1.44, -0.7, 50)
    surviving_a = get_surviving_weights(build)
    baseline_a, _ = plot_last_weight_hist(ax1, build, color=color, label="all synapses",
                                          return_weights=True, bins=bins)
    centers = (bins[1:] + bins[:-1])/2

    baseline_counts, _ = np.histogram(baseline_a, bins=bins, density=False)
    surviving_counts, _ = np.histogram(np.log10(surviving_a), bins=bins, density=False)
    if print_info:
        print("#baseline", np.sum(baseline_counts), "#surviving", np.sum(surviving_counts))
    if print_info:
        h0_same_dist(baseline_counts, surviving_counts)

    surviving_density, _ = np.histogram(np.log10(surviving_a), bins=bins, density=True)
    ax1.plot(centers, surviving_density, label="surviving", color="red", drawstyle="steps-mid", lw=1.0)
    ax1.set_title(f"Sim. {build[1]} (rev. reg.)")
    ax1.legend(loc="lower center")
    if print_info:
        print()