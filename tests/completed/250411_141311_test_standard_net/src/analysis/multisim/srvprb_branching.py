import textwrap

import numpy as np
from brian2.units import *
from scipy.stats import chi2_contingency

from analysis.multisim.multisimplotrunner import *
from net.analysis_cache import CachedAnalysis


class SrvPrbBranching(MultiSimPlotRunner):

    bin_w = 4*ms

    def __init__(self, name="srvprb_branching", plot_count=(1, 1), in_notebook=False, no_subcritical=False,
                 colors=None):
        super(SrvPrbBranching, self).__init__(name, plot_count, in_notebook=in_notebook)
        self.additional_data = {}
        self.metadata = {}
        self.no_subcritical = no_subcritical
        self.colors = colors

    def _add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument("--group", type=str, nargs='*', action="append",
                            help="an additional group, name first, then directory list")

    def _process_args(self, args):
        if args.group is None or len(args.group) == 0:
            return
        for group in args.group:
            name, dirs = group[0], group[1:]
            self.additional_data[name] = []
            for dir in dirs:
                self.additional_data[name].append(self.unpickle(dir, "survival_full_t"))

    def _metadata(self):
        return self.metadata

    def plot(self, directories, nsps, fig, axs, fontsize=18):
        ax = axs[0][0]
        revreg, subcrit = "reverberating", "sub-critical"
        if self.colors is None:
            colors = {revreg: "green", subcrit: "blue"}
        else:
            colors = self.colors
        if self.no_subcritical:
            data, mres = {revreg: []}, {revreg: []}
        else:
            data, mres = {revreg: [], subcrit: []}, {revreg: [], subcrit: []}
        for dir, nsp in zip(directories, nsps):
            with CachedAnalysis(dir) as cache:
                fit_mre = cache.get_mre(self.bin_w)

                if fit_mre > 0.995:
                    print(f"skipping: mre is critical or super-critical: {dir}")
                    continue
                group = revreg if fit_mre > 0.9 else subcrit
                if self.no_subcritical and group == subcrit:
                    continue
                data[group].append(self.unpickle(dir, "survival_full_t"))
                mres[group].append(fit_mre)

        assert(len(data[revreg]) > 0 and (self.no_subcritical or len(data[subcrit]) > 0))

        # strct plasticity is only applied every 1000ms
        # so bin size needs to be at least 1s, which logspace doesn't do for range < 10
        # bins = np.logspace(np.log10(10**1.3),
        #                    np.log10(np.max([data[revreg][i]['t_split']/second for i in range(0, len(data[revreg]))])+0.1),
        #                    num=30)  # 82 exhibits jump in blue curve
        # bins = np.hstack([np.arange(1, 17, 1), bins])
        # bins = np.linspace(1, np.max([data[revreg][i]['t_split']/second for i in range(0, len(data[revreg]))])+0.1, 100)
        min_bin_size = np.log10(2) - np.log10(1)
        min_log_weight = np.log10(1)
        max_log_weight = np.log10(np.max([data[revreg][i]['t_split'] / second for i in range(0, len(data[revreg]))]))
        bins = 10**np.arange(min_log_weight, max_log_weight + min_bin_size, step=min_bin_size)
        centers = (bins[:-1] + bins[1:])/2.

        print("min_bin_size", min_bin_size)
        print("bins", np.log10(bins))
        print("centers", np.log10(centers))

        ax.set_xscale('log')
        ax.set_yscale('log')

        data.update(self.additional_data)

        chi_input_data = []
        additional_colors = ["seagreen", "steelblue", "coral", "pink", "yellow", "cyan"]
        additional_color_i = 0
        def sub_rev_front(t):
            if t[0] == "sub-critical":
                return " 0"
            elif t[0] == "reverberating":
                return " 1"
            else:
                return t[0]
        for label, dfs in sorted(data.items(), key=sub_rev_front):
            if label in colors:
                color = colors[label]
            else:
                color = additional_colors[additional_color_i]
                additional_color_i += 1
            linestyle, linewidth, markerstyle = "-", 1.0, "o"
            counts_density = [np.histogram(df['full_t'], bins=bins, density=True)[0] for df in dfs]
            avg_density, std = np.mean(counts_density, axis=0), np.std(counts_density, axis=0)
            print(f"{label} avg counts estimate", avg_density*np.diff(bins)*len(dfs[0]['full_t']))
            print("std", std)
            ax.plot(centers, avg_density, label=label, color=color, ls=linestyle, lw=linewidth, marker=markerstyle, ms=1.75)
            ax.fill_between(centers, avg_density-std, avg_density+std, facecolor=f"{color}", linewidth=0, alpha=0.2)

            # for significance test
            counts = [np.histogram(df['full_t'], bins=bins, density=False)[0] for df in dfs]
            avg = np.mean(counts, axis=0)
            chi_input_data.append(avg)

        group_info = [(label, len(dfs), np.mean(mres[label]), np.std(mres[label])) for label, dfs in data.items() if label in mres]
        group_info_str = [f"{label}: N={no:2d}, $\hat{{m}}={mean:.2f}\pm{std:.2f}$" for label, no, mean, std in group_info if label in mres]
        ax.legend(loc="lower left", fontsize=12)
        # ax.text(1.0, -0.25, '\n'.join(group_info_str), transform=ax.transAxes, ha='right')
        ax.set_xlabel("survival time [s]", fontsize=fontsize)
        ax.set_ylabel("probability density", fontsize=fontsize)
        ax.set_title("Survival times of EE synapses", fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        # ax.tick_params(axis='x', which='minor', labelsize=fontsize/4)
        # ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))

        print("")
        p_expected = 0.05
        # print(chi_input_data)
        _, p, dof, ex = chi2_contingency(np.array(chi_input_data))
        description = textwrap.dedent(f"""
            H0: subcrit & rev survival times are drawn from same distribution
            test with X-squared contingency/two samples test
            reject if p < {p_expected}
            p = {p} => {'reject' if p < p_expected else 'accept'}""")
        self.metadata = dict(Description=description)


if __name__ == '__main__':
    SrvPrbBranching().run()
