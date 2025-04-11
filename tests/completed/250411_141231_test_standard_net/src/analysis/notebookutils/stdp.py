import pandas as pd
from brian2 import *
from .analysis import pickle_load, get_label


def plot_weights_onto_exc(ax: plt.Axes, file, builds, label_attr=None):
    N_e = builds[0][2]['N_e'] if 'N_e' in builds[0][2] else 1600
    for build in builds:
        syn = pickle_load(build, file, print_attrs=False)
        onto_over_time = np.sum(syn['a'], axis=1)/N_e
        ax.plot(syn['t']/second, onto_over_time, label=get_label(build, label_attr))
    ax.set_xlabel("time [second]")
    ax.set_ylabel("avg total weight")
    ax.legend()


def plot_ei_onto(ax: plt.Axes, builds, label_attr=None):
    plot_weights_onto_exc(ax, "synei_a", builds, label_attr)


def plot_ee_onto(ax: plt.Axes, builds, label_attr=None):
    plot_weights_onto_exc(ax, "synee_a", builds, label_attr)


def calc_stdp_weight_diff(build, return_weights=False):
    # load dw by STDP event (multiple events per synapse)
    # format: t, i, j, a, Apre, Apost, preorpost
    spk_register = pickle_load(build, "spk_register", print_attrs=False)
    pre_post_index = spk_register[:, 6].astype(int)  # pre=0, post=1
    da = spk_register[:, [4, 5]]  # pre at 0, post at 1
    mask_basis = np.tile([1, 0], da.shape[0]).reshape(da.shape)  # [1, 0] because for pre to look at Apost
    mask_index = np.repeat(pre_post_index, da.shape[1]).reshape(da.shape)
    mask = mask_basis == mask_index
    dw = da[mask]  # by STDP event, not by synapse

    # calculate STDP change per synapse
    indices = (spk_register[:, 1] * 1600 + spk_register[:, 2]).astype(int)
    change_by_synapse = np.vstack([indices, dw]).T
    df = pd.DataFrame(change_by_synapse, columns=["synapse_id", "dw"])
    df_dw_second = df.groupby("synapse_id").sum()
    if return_weights:
        ts = spk_register[:, 0]
        weights = spk_register[:, 3]
        weight_by_synapse = np.vstack([indices, ts, weights]).T
        df = pd.DataFrame(weight_by_synapse, columns=["synapse_id", "t", "w"])
        df_weights = df.loc[df.groupby("synapse_id").t.idxmin()]
        return pd.merge(df_weights[['synapse_id', 'w', 't']], df_dw_second[['dw']], on='synapse_id')
    else:
        return df_dw_second.to_numpy()


def plot_stdp_by_regime(wdw_revreg, wdw_subcrit, bins=None, colors=["green", "blue"], labels=["rev reg", "sub crit"]):
    bins = np.linspace(-0.04, 0.06, 75) if bins is None else bins
    fig, axs = plt.subplots(1, 2, figsize=(2 * 6, 4))
    (ax0, ax1) = axs
    ax0.set_title("all synapses")
    ax0.hist(wdw_revreg['dw'], bins=bins, histtype="step", color=colors[0], label=labels[0])
    ax0.hist(wdw_subcrit['dw'], bins=bins, histtype="step", color=colors[1], label=labels[1])

    ax1.set_title("all synapses (y-axis cut)")
    ax1.hist(wdw_revreg['dw'], bins=bins, histtype="step", color=colors[0], label=labels[0])
    ax1.hist(wdw_subcrit['dw'], bins=bins, histtype="step", color=colors[1], label=labels[1])
    ax1.set_ylim(0, 1500)
    for ax in axs:
        ax.set_xlabel("dw/second")
        ax.legend()


def plot_std_by_regime_small_large(wdw_revreg, wdw_subcrit, bins=None, colors=["green", "blue"], labels=["rev reg", "sub crit"]):
    bins = np.linspace(-0.04, 0.06, 75) if bins is None else bins
    topp = 0.1
    dw_revreg_top = wdw_revreg.nlargest(int(topp * len(wdw_revreg)), 'w')['dw'].to_numpy()
    dw_subcrit_top = wdw_subcrit.nlargest(int(topp * len(wdw_subcrit)), 'w')['dw'].to_numpy()

    dw_revreg_bottom = wdw_revreg.nsmallest(int(topp * len(wdw_revreg)), 'w')['dw'].to_numpy()
    dw_subcrit_bottom = wdw_subcrit.nsmallest(int(topp * len(wdw_subcrit)), 'w')['dw'].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(2 * 6, 2 * 4), sharey='col')
    (ax1, ax2), (ax3, ax4) = axes
    for ax1, ax3 in axes.T:
        ax1.set_title("10 % smallest synapses")
        ax1.hist(dw_revreg_bottom, bins=bins, histtype="step", color=colors[0], label=labels[0])
        ax1.hist(dw_subcrit_bottom, bins=bins, histtype="step", color=colors[1], label=labels[1])
        ax3.set_title("10 % largest synapses")
        ax3.hist(dw_revreg_top, bins=bins, histtype="step", color=colors[0], label=labels[0])
        ax3.hist(dw_subcrit_top, bins=bins, histtype="step", color=colors[1], label=labels[1])
    for ax in axes.flatten():
        ax.set_xlabel("dw/second")
        ax.legend()
    for right_col_ax in axes.T[1]:
        right_col_ax.set_title(right_col_ax.get_title() + " (y-axis cut)")
        right_col_ax.set_ylim(0, 1000)
    fig.tight_layout()


def plot_stdp_avg(rev_mean, rev_std, sub_mean, sub_std, colors=["green", "blue"], labels=["rev reg", "sub crit"]):
    bins = np.linspace(-0.04, 0.06, 75)
    centers = (bins[1:] + bins[:-1]) / 2
    fig, axs = plt.subplots(1, 2, figsize=(2 * 6, 4))
    (ax0, ax1) = axs
    ax0.set_title("all synapses")
    ax0.plot(centers, rev_mean, color=colors[0], label=labels[0])
    ax0.fill_between(centers, rev_mean - rev_std, rev_mean + rev_std, color=f"light{colors[0]}", alpha=0.1)
    ax0.plot(centers, sub_mean, color=colors[1], label=labels[1])
    ax0.fill_between(centers, sub_mean - sub_std, sub_mean + sub_std, color=f"light{colors[1]}", alpha=0.1)

    ax1.set_title("all synapses (y-axis cut)")
    ax1.plot(centers, rev_mean, color=colors[0], label=labels[0])
    ax1.fill_between(centers, rev_mean - rev_std, rev_mean + rev_std, color=f"light{colors[0]}", alpha=0.1)
    ax1.plot(centers, sub_mean, color=colors[1], label=labels[1])
    ax1.fill_between(centers, sub_mean - sub_std, sub_mean + sub_std, color=f"light{colors[1]}", alpha=0.1)
    ax1.set_ylim(0, 25)
    for ax in axs:
        ax.set_xlabel("dw/second")
        ax.legend()
