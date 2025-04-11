import matplotlib.pyplot as plt
from brian2 import *

from .analysis import pickle_load, get_label


def plot_threshold_hist(ax: plt.Axes, builds):
    ts = []
    for build in builds:
        stat = pickle_load(build, "gexc_stat")
        ax.hist(stat['Vt'][-1, :] / mV, histtype='step', label=get_label(build))
        ax.legend()
        ts.append(stat['t'][-1])
    return ts


def plot_threshold_progression(ax: plt.Axes, builds, time_frame):
    actual_tf = [np.inf, -np.inf]
    for build in builds:
        stat = pickle_load(build, "gexc_stat")
        indices = np.logical_and(stat['t'] > time_frame[0], stat['t'] < time_frame[1])
        ts = stat['t'][indices]
        mean = np.mean(stat['Vt'][indices, :], axis=1)
        std = np.std(stat['Vt'][indices, :], axis=1)
        ax.plot(ts, mean, label=get_label(build))
        ax.fill_between(ts, mean-std, mean+std, alpha=0.1)
        ax.legend()
        actual_tf[0] = np.min([np.min(ts), actual_tf[0]])
        actual_tf[1] = np.max([np.max(ts), actual_tf[1]])
    return actual_tf


def plot_firingrate(ax: plt.Axes, builds, time_frame, toffset=0, label_attr=None, skip=1):
    actual_tf = [np.inf, -np.inf]
    for build in builds:
        rate, smth = pickle_load(build, "gexc_rate", 2)
        indices = np.logical_and(rate['t'] > time_frame[0], rate['t'] < time_frame[1])
        ts = rate['t'][indices] - toffset
        ax.plot(ts[::skip], smth[indices][::skip], label=get_label(build, label_attr))
        actual_tf[0] = np.min([np.min(ts), actual_tf[0]])
        actual_tf[1] = np.max([np.max(ts), actual_tf[1]])
    ax.set_xlabel("time [seconds]")
    ax.set_ylabel("rate [Hz]")
    return actual_tf


def plot_firingrate_hist(ax: plt.Axes, builds, time_frame, toffset=0, label_attr=None):
    actual_tf = [np.inf, -np.inf]
    for build in builds:
        rate, smth = pickle_load(build, "gexc_rate", 2)
        indices = np.logical_and(rate['t'] > time_frame[0], rate['t'] < time_frame[1])
        ts = rate['t'][indices] - toffset
        ax.hist(smth[indices]/Hz, label=get_label(build, label_attr), histtype="step", bins=50, density=True)
        actual_tf[0] = np.min([np.min(ts), actual_tf[0]])
        actual_tf[1] = np.max([np.max(ts), actual_tf[1]])
    ax.set_xlabel("firing rate [Hz]")
    ax.set_ylabel("density")
    return actual_tf


def plot_activity_hist(ax: plt.Axes, build, time_frame, bin_w = 5 * ms, color="tab:blue", tunit=ms, type="gexc_spks"):
    spks = pickle_load(build, type)
    indices = np.logical_and(spks['t'] > time_frame[0], spks['t'] < time_frame[1])
    ts = spks['t'][indices]

    bins = np.arange(np.min(ts) / tunit, (np.max(ts) + bin_w) / tunit, bin_w / tunit)
    bins = bins * tunit
    counts, _ = np.histogram(ts / tunit, bins=bins / tunit, density=False)
    ax.bar(bins[1:] / tunit, counts, width=0.001, facecolor=color, edgecolor=color)
    ax.set_xlabel(f"time [{tunit.name}]")
    ax.set_ylabel("exc. spike count")

    return indices
