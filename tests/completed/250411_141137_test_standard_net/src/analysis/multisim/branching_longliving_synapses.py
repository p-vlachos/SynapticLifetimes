import numpy as np
from brian2.units import *

from analysis.multisim.multisimplotrunner import *
from analysis.multisim.helpers import operate_by_regime


class BranchingLongLivingSynapses(MultiSimPlotRunner):

    bin_w = 4*ms

    def __init__(self, name="branching_longliving_synapses", plot_count=(1, 1)):
        super(BranchingLongLivingSynapses, self).__init__(name, plot_count)
        self.additional_data = {}

    def plot(self, directories, nsps, fig, axs):
        t_min = 20000*second
        ax = axs[0][0]
        fontsize = 18

        def operation(dir):
            df = self.unpickle(dir, "survival_full_t")
            full_t = np.array(df['full_t'])
            return np.sum(full_t > t_min / second)
        out_subcrit, out_revreg = operate_by_regime(directories, self.bin_w, operation)

        colors = ["blue", "darkgreen"]
        counts = np.array([out_subcrit, out_revreg])
        avg, std = np.mean(counts, axis=1), np.std(counts, axis=1)

        ax.set_title(f"Number of synapses surviving more than {str(t_min)}", fontsize=fontsize)
        ax.bar([0, 1], avg, yerr=std, color=colors, tick_label=["sub-critical", "reverberating"])
        ax.tick_params(axis='both', which='major', labelsize=fontsize)


if __name__ == '__main__':
    BranchingLongLivingSynapses().run()
