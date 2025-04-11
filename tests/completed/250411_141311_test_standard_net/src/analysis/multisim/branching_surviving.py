import numpy as np
from brian2.units import *
import matplotlib.pyplot as plt

from analysis.multisim.multisimplotrunner import *
from analysis.multisim.helpers import operate_by_regime
from net.utils import generate_full_connectivity


class SurvivingSynapses(MultiSimPlotRunner):

    bin_w = 4*ms

    def __init__(self, name="branching_surviving", plot_count=(1, 1), in_notebook=False):
        super(SurvivingSynapses, self).__init__(name, plot_count, in_notebook=in_notebook)
        self.additional_data = {}

    def plot(self, directories, nsps, fig, axs, fontsize=18):
        ax = axs[0][0]
        Tmax = 26*ksecond
        bins = np.linspace(0, Tmax/second, 13)

        def operation(dir, debug=True):
            Ne = 1600
            i, j = generate_full_connectivity(Ne, same=True)
            # find out which synapse ids where active initially
            syneea = self.unpickle(dir, "synee_a")
            synactive0 = np.full((Ne, Ne), np.nan)
            synactive0[i, j] = syneea['syn_active'][0][:]
            iactive, jactive = np.where(synactive0 == 1.0)
            ids_active = set(np.unique(iactive * 1600 + jactive).astype(int).flatten())
            if debug:
                print("alive at beginning", len(ids_active), "per neuron", len(ids_active) / 1600)
            # find out which synapses were destroyed over the course of the simulation
            turnover = self.unpickle(dir, "turnover")  # 1 create 0 destroy | t | i | j
            idestroyed, jdestroyed = turnover[turnover[:, 0] == 0.0][:, [2, 3]].T
            if debug:
                print("destroy events full sim time", len(idestroyed), "per neuron", len(idestroyed)/1600)
            ids_destroyed = set(np.unique(idestroyed * 1600 + jdestroyed).astype(int).flatten())
            if debug:
                print("destroyed over full sim time", len(ids_destroyed), "per neuron", len(ids_destroyed) / 1600)
            # we now have to sets and find what is in ids_active that hasn't been destroyed
            alive_at_end = ids_active - ids_destroyed
            count_alive_at_end = len(alive_at_end)
            percent_alive_at_end = count_alive_at_end/len(ids_active)
            if debug:
                print("alive at end", count_alive_at_end, "percent", percent_alive_at_end)
            return percent_alive_at_end
        out_subcrit, out_revreg = operate_by_regime(directories, self.bin_w, operation)
        print("subcrit", out_subcrit)
        print("revreg", out_revreg)

        colors = ["blue", "darkgreen"]
        counts = np.array([out_subcrit, out_revreg])
        avg, std = np.mean(counts, axis=1), np.std(counts, axis=1)

        ax.set_title(f"Fraction of synapses surviving full simulation", fontsize=fontsize)
        ax.bar([0, 1], avg, yerr=std, color=colors, tick_label=["sub-critical", "reverberating"])
        ax.tick_params(axis='both', which='major', labelsize=fontsize)


if __name__ == '__main__':
    SurvivingSynapses().run()
