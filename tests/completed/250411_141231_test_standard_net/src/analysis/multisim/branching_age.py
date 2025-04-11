import numpy as np
from brian2.units import *
import matplotlib.pyplot as plt

from analysis.multisim.multisimplotrunner import *
from analysis.multisim.helpers import operate_by_regime


class BranchingAge(MultiSimPlotRunner):

    bin_w = 4*ms

    def __init__(self, name="branching_age", plot_count=(1, 1)):
        super(BranchingAge, self).__init__(name, plot_count)
        self.additional_data = {}

    def plot(self, directories, nsps, fig, axs):
        ax = axs[0][0]
        fontsize = 18
        Tmax = 26*ksecond
        bins = np.linspace(0, Tmax/second, 13)

        def operation(dir):
            df = self.unpickle(dir, "survival_full_t")
            full_t = np.array(df['full_t'])
            return np.histogram(full_t, bins=bins)[0]  # (counts, bins)
        out_subcrit, out_revreg = operate_by_regime(directories, self.bin_w, operation)

        colors = ["blue", "darkgreen"]
        counts = np.array([out_subcrit, out_revreg])
        avg, std = np.mean(counts, axis=1), np.std(counts, axis=1)
        # ax.set_title(f"Number of synapses surviving more than {str(t_min)}", fontsize=fontsize)
        width = 0.9
        bar_positions = np.array(list(range(len(avg[0]))))
        labels = [f"{bins[i+1]/1000:.0f}" for i in bar_positions]
        ax.bar(bar_positions, avg[0], align='edge', width=-width, yerr=std[0], color=colors[0], label="sub-critical", tick_label=labels)
        ax.bar(bar_positions, -avg[1], align='edge', width=-width, yerr=std[1], color=colors[1], label="reverberating")
        # ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.legend(fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.setp(ax.get_xticklabels(), horizontalalignment='center')
        ax.set_xlabel("time [Ks]", fontsize=fontsize)
        ax.set_ylabel("synapse count", fontsize=fontsize)
        ax.set_title("Synapse count over lifetime by regime", fontsize=fontsize)


if __name__ == '__main__':
    BranchingAge().run()
