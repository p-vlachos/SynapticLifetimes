import enum
import os

from PIL.Image import Exif
from PIL.ExifTags import TAGS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import List

from analysis.isi import calc_isi
from notebookutils.analysis import *

# Matplotlib Tools
class SizeMode(enum.IntEnum):
    FigSizeIsSingleAxSize = 0
    FigSizeIsAllAxSize = 1


def set_fontsize(fontsize):
    matplotlib.rc('font', **{'size': fontsize})
    matplotlib.rc('figure', **{'titlesize': fontsize})
    matplotlib.rc('axes', **{'labelsize': fontsize, 'titlesize': fontsize})
    matplotlib.rc('xtick', **{'labelsize': fontsize})
    matplotlib.rc('ytick', **{'labelsize': fontsize})
    matplotlib.rc('legend', **{'fontsize': fontsize})


class Style:

    def __init__(self, fapply, smallfontsize):
        self._apply = fapply
        self._style = None
        self._smallfontsize = smallfontsize

    def apply(self):
        self._style = self._apply()
        self._style['scale'] = 1
        self._style['size_kwargs'] = {}
        return self._style

    def __getitem__(self, item):
        if self._style is None:
            raise Exception('Call .apply() before accessing properties')
        return self._style[item]

    def select(self, params):
        return params[self._style['name']]

    def now_use_small_fontsize(self):
        set_fontsize(self._smallfontsize)

    def now_scale_axs(self, scale):
        self._style['scale'] = scale

    def now_size_kwargs(self, kwargs):
        self._style['size_kwargs'] = kwargs


def create_style(name, fontsize, figsize, sizeMode, dpi=300, mrc={}, smallfontsize=None):
    smallfontsize = 0.75 * fontsize if smallfontsize is None else smallfontsize
    def style():
        # TODO somehow Agg backend messes up fonts and uses one of the two fonts, while fig.savefig works flawlessly
        # TODO use matplotlib API better
        matplotlib.rc('font', **{'family': 'serif', 'weight': 'normal', 'serif': ['Computer Modern']})
        set_fontsize(fontsize)
        matplotlib.rc('text', **{'usetex': True})
        for group in ['patch', 'lines', 'axes']:
            matplotlib.rc(group, **{'linewidth': 0.8})
        for k, v in mrc.items():
            print("applying", k, v)
            matplotlib.rc(k, **v)
        return dict(name=name, fontsize=fontsize, figsize=figsize, dpi=dpi, sizeMode=sizeMode)
    return Style(fapply=style, smallfontsize=smallfontsize)


style_MA = create_style(name='MA', fontsize=10, figsize=(4.8, 3.0), sizeMode=SizeMode.FigSizeIsAllAxSize)
style_PR = create_style(name='Presentation', fontsize=14, figsize=(3.6, 2.25), sizeMode=SizeMode.FigSizeIsSingleAxSize,
                        mrc={
                            # 'font': {'family': 'serif', 'serif': ['Latin Modern Roman']},
                             'text': {'usetex': False},
                             'font': {'family': 'sans-serif', 'weight': 'normal', 'sans-serif': ['DejaVu Sans']}
                        }, smallfontsize=9)
style_PO = create_style(name='Poster', fontsize=24, figsize=(14.6, 9.125), sizeMode=SizeMode.FigSizeIsAllAxSize,
                        mrc={'text': {'usetex': False},
                             'font': {'family': 'sans-serif', 'weight': 'normal', 'sans-serif': ['DejaVu Sans']}
                             })
style_POs = create_style(name='Poster', fontsize=24, figsize=(6.5, 4.06), sizeMode=SizeMode.FigSizeIsAllAxSize,
                         mrc={'text': {'usetex': False},
                              'font': {'family': 'sans-serif', 'weight': 'normal', 'sans-serif': ['DejaVu Sans']}
                              })


def plot_styles(axInfo, styles: List[Style]=[style_MA], name=None, subplots_kwargs={}):
    for style in styles:
        style.apply()
        figsize = np.array(style['figsize'])
        axInfo = np.array(axInfo)
        if style['sizeMode'] == SizeMode.FigSizeIsSingleAxSize:
            figsize = figsize*np.flip(axInfo)
        elif style['sizeMode'] == SizeMode.FigSizeIsAllAxSize:
            figsize = figsize
        if 'dpi' not in subplots_kwargs:
            subplots_kwargs['dpi'] = style['dpi']
        fig, axs = plt.subplots(*axInfo, figsize=figsize, **subplots_kwargs)
        yield (fig, axs, style)
        # TODO pass wspace, hspace through
        if name is not None:
            stylizedName = f'{style["name"]}/{name}'
        ax = axs if isinstance(axs, plt.Axes) else axs.flatten()[0]
        scale = style['scale']
        set_size_kwargs = style['size_kwargs']
        if name is None:
            print("Did not save to file, because no name was provided.")
        else:
            if 'size' not in set_size_kwargs:
                set_size_kwargs['size'] = figsize
            actualFigSize = ax_set_size_dpi(ax, name=stylizedName, scale=scale, **set_size_kwargs)
            print(f'{stylizedName}: {actualFigSize} in')


def ax_set_size_dpi(ax: plt.Axes, dpi=300, wspace=0.3, hspace=0.5, size=None, name=None, scale=1):
    """
    As last act of plotting a figure, call this with an arbitrary axes of the figure.
    Returns the size (in inches) that the final PNG will have.

    :param ax: An arbitrary axes of the figure
    :param dpi:
    :param wspace: horizontal spacing between axes, in percentage of average axes width
    :param hspace: vertical spacing between axes, in percentage of average axes height
    :param size: desired size that the axes (without labels, titles etc.) should take, defaults to figsize
    :param name: if desired, this is saved as PGF for LaTeX documents with this file name
    :return: size of final PNG in inches
    """
    # adapted from https://stackoverflow.com/a/44971177/4845703

    fig: plt.Figure = ax.figure
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    fig.set_dpi(dpi)

    l = fig.subplotpars.left
    r = fig.subplotpars.right
    t = fig.subplotpars.top
    b = fig.subplotpars.bottom

    if size is None:
        w, h = fig.bbox_inches.size
    else:
        w, h = size
    figw = float(w)*scale/(r-l)  # r, l are not in inches but are fractions
    figh = float(h)*scale/(t-b)
    fig.set_size_inches(figw, figh)

    canvas: plt.FigureCanvasBase = fig.canvas
    tightBox = fig.get_tightbbox(canvas.get_renderer()).padded(matplotlib.rcParams['savefig.pad_inches'])
    actualSizeIn = tightBox.size

    if name is not None:
        import os
        figures_dir = os.path.join(os.path.dirname(__file__), "../../../figures")
        path = os.path.join(figures_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # fig.savefig(f'{path}.pgf', bbox_inches='tight')

        description = f'{actualSizeIn[0]:.2f} x {actualSizeIn[1]:.2f} [inch]'
        # most tools ignore Exif on PNG, but there's a chance
        # that some GUI shows the 'ImageDescription' field
        import piexif
        pil_kwargs = dict(exif=piexif.dump({
            '0th': {
                piexif.ImageIFD.XResolution: (dpi, 1),  # 1 is inch
                piexif.ImageIFD.YResolution: (dpi, 1),
                piexif.ImageIFD.ImageDescription: description,
            },
        }))
        # same here, most tools don't show these, but some might
        metadata = { 'Description': description, 'Comment': description, }
        fig.savefig(f'{path}.png', bbox_inches='tight', metadata=metadata, pil_kwargs=pil_kwargs)
    return np.round(actualSizeIn, 4)


# Weight <-> PSP

from brian2 import ms, mV


def plot_synapses_singleexp(weight,
                            tau_synapse=1.7 * ms, tau_m=20 * ms,
                            Ereversal=0.0 * mV, Vr=-60. * mV, El=-60. * mV, Vt=-50. * mV):
    from brian2 import start_scope, NeuronGroup, Synapses, StateMonitor, run, ms, mV
    start_scope()

    E = NeuronGroup(N=2, model='''
        dV/dt = (El-V + ge*(Ereversal-V))/tau_m : volt
        dge/dt = -ge/tau_synapse                   : 1
    ''', threshold="V > Vt", reset="V = Vr", method='euler', dt=0.1 * ms)
    S = Synapses(E, E, model='''
        a : 1
    ''', on_pre='''
        ge_post += a
    ''')
    S.connect(i=0, j=1)
    S.a = weight

    E.V = El

    state = StateMonitor(E, ["V", "ge"], True, dt=.1 * ms)

    run(10 * ms)
    E[0].V = Vt + 1 * mV
    run(90 * ms)

    return state.t, state.V[1].T


def plot_synapses_doubleexp_old(weight,
                            tau_m=20 * ms,
                            Ereversal=0.0 * mV, Vr=-60. * mV,
                            norm_f_EE=2.75, tau_e_rise=3.75*ms, tau_e=6*ms,
                            El=-60. * mV, Vt=-50. * mV):
    from brian2 import start_scope, NeuronGroup, Synapses, StateMonitor, run, ms, mV
    start_scope()
    invpeakEE = (tau_e / tau_e_rise) ** (tau_e_rise / (tau_e - tau_e_rise))

    E = NeuronGroup(N=2, model='''
        dV/dt = (El-V + ge*(Ereversal-V))/tau_m : volt
        dge/dt = (invpeakEE*xge-ge)/tau_e_rise : 1
        dxge/dt = -xge/tau_e                   : 1
    ''', threshold="V > Vt", reset="V = Vr", method='euler', dt=0.1 * ms)
    S = Synapses(E, E, model='''
        a : 1
    ''', on_pre='''
        xge_post += a/norm_f_EE
    ''')
    S.connect(i=0, j=1)
    S.a = weight

    E.V = El

    state = StateMonitor(E, ["V", "ge"], True, dt=.1 * ms)

    run(10 * ms)
    E[0].V = Vt + 1 * mV
    run(90 * ms)

    return state.t, state.V[1].T


def determine_max_amplitude(weight, f, **kwargs):
    ts, Vs = f(weight=weight, **kwargs)
    if kwargs['Ereversal'] > kwargs['Vt']:  # excitatory
        return (np.max(Vs) - np.min(Vs)) / mV
    else:  # inhibitory
        return (np.min(Vs) - np.max(Vs)) / mV


def weight_to_psp(weights, f=plot_synapses_singleexp, **kwargs):
    return [determine_max_amplitude(weight, f=f, **kwargs) for
            weight in weights]


def plot_weights(tau_synapse, Ereversal, Vr, El, Vt, w_min=0.001, w_max=0.3, count=15):
    weights = np.linspace(w_min, w_max, count)
    amplitudes = weight_to_psp(weights, tau_synapse=tau_synapse, Ereversal=Ereversal, Vr=Vr, El=El, Vt=Vt)
    plt.plot(np.log10(weights), amplitudes, linestyle="None", marker="x", color="blue")
    plt.xlabel("log weight")
    plt.ylabel("amplitude [mV]")
    plt.title("Amplitude by weight")
    return weights, amplitudes


def analyze_synapse_over_weights(tau_synapse, Ereversal, Vr, El, Vt, ampl_query, w_min=0.001, w_max=0.5, count=10):
    from scipy.optimize import curve_fit

    def f_linear(x, a, b):
        return a * x + b

    def f_lin_inv(y, a, b):
        # y = ax + b => y-b = ax => (y-b)/a = x
        return (y - b) / a

    def fit_weights_amplitudes(weights, w_range, amplitudes):
        fit_par, fit_cov = curve_fit(f_linear, weights, amplitudes)
        fit_err = np.sqrt(np.diag(fit_cov))
        xs = np.arange(*w_range, 0.01)
        plt.plot(xs, f_linear(xs, *fit_par), linestyle="--", label="linear fit", color="green")
        #     plt.fill_between(xs, f_linear(xs, *(fit_par-fit_err)),
        #                          f_linear(xs, *(fit_par+fit_err)), alpha=0.1, color="lightgreen")
        plt.plot(weights, amplitudes, ls="None", marker="x", label="simulated", color="blue")
        plt.xlabel("weights")
        plt.ylabel("amplitudes [mV]")
        plt.legend()
        plt.title("Linear fit to amplitudes")

        print(f"linear fit: slope={fit_par[0]:0.4f} intersect={fit_par[1]:0.4f})")

        return fit_par

    def amplitude_to_weight(ampl_query, fit_par):
        for q_ampl in ampl_query:
            print(f"weight({q_ampl:0.2f} mV) = {f_lin_inv(q_ampl, *fit_par):.4f}")

    weights, amplitudes = plot_weights(w_min=w_min, w_max=w_max, count=count,
                                       tau_synapse=tau_synapse, Ereversal=Ereversal, Vr=Vr, El=El, Vt=Vt)
    plt.show()
    fit_par = fit_weights_amplitudes(weights, [w_min, w_max], amplitudes)
    plt.show()
    amplitude_to_weight(ampl_query, fit_par)


## Plot Weights

def load_weights_from_build(build, type, tindex):
    syn = pickle_load(build, f"syn{type}_a", print_attrs=False)
    indices = syn['a'][tindex] > 0
    a = syn['a'][tindex][indices]

    return a


def find_time_for_tindex_from_build(builds, type, tindex):
    syn = pickle_load(builds[0], f"syn{type}_a", print_attrs=False)
    return syn['t'][tindex]


def plot_weights_with_fits(ax, builds, type="ee", show_threshold=True, xlim=None, xlinspace=[-1.51, -0.5],
                           fit_p0=[-1.0, 0.01], tindex=-1,
                           division_by_regime={"rev": [0, 1, 2], "sub": [3, 4, 5]},
                           colors = {"rev": "green", "sub": "blue"},
                           labels = {"rev": "rev. reg.", "sub": "sub. reg."}, llognorm="logn. fit", lthres="prun. thr.",
                           load_weights=load_weights_from_build,
                           find_time_for_tindex=find_time_for_tindex_from_build,
                           legend_kwargs=dict()):
    if labels is None:
        labels = {k: k for k in division_by_regime.keys()}

    tpoint = find_time_for_tindex(builds, type, tindex)

    def only_between(a, bounds):
        return a[ (bounds[0] <= a) & (a <= bounds[1])]

    wbins = np.linspace(xlinspace[0], xlinspace[1], 50)
    wbinc = (wbins[1:]+wbins[:-1])/2
    weights = [load_weights(build, type, tindex) for build in builds]
    histsw = np.array([np.histogram(only_between(np.log10(build_weights), xlinspace), bins=wbins, density=True)[0] for build_weights in weights])
    wmeans = {reg: np.mean(histsw[division_by_regime[reg]], axis=0) for reg in division_by_regime.keys()}
    wstds = {reg: np.std(histsw[division_by_regime[reg]], axis=0) for reg in division_by_regime.keys()}

    from scipy.optimize import curve_fit
    from scipy.stats import norm

    fit = {}
    for reg, ys in wmeans.items():
        fit[reg] = curve_fit(f=norm.pdf, xdata=wbinc, ydata=ys, p0=fit_p0)[0]

    for reg, wmean in wmeans.items():
        color = colors[reg]
        ax.plot(wbinc, wmean, color=color, label=labels[reg])
        ax.fill_between(wbinc, wmean-wstds[reg], wmean+wstds[reg], color=color, alpha=0.2)

        ax.plot(wbinc, norm(*fit[reg]).pdf(wbinc), color=color, ls='--')

    if show_threshold:
        show_threshold = show_threshold if isinstance(show_threshold, str) else 'amin'
        ax.axvline(np.log10(builds[0][2][show_threshold]), 0, 1, color="red")

    ax.set_xlabel("log weight")
    ax.set_ylabel("density")
    if tindex == -1:
        ax.set_title(f"{type.upper()} weights at simulation end")
    else:
        ax.set_title(f"{type.upper()} weights at {tpoint/second} s")

    import matplotlib.lines as mlines
    handles, labels = ax.get_legend_handles_labels()
    if show_threshold:
        handles.append(mlines.Line2D([], [], color='red'))
        labels.append(lthres)
    handles.append(mlines.Line2D([], [], color='black', ls='--'))
    labels.append(llognorm)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if 'handlelength' not in legend_kwargs:
        legend_kwargs['handlelength'] = 0.75

    ax.legend(handles, labels, loc='upper right', **legend_kwargs)


def plot_weight_history(ax, builds, type="ee", show_threshold=True, xlim=None,
                           xlinspace=[-1.51, -0.5], tindexs=[-1], nlinspace=50,
                           devision_by_regime={"rev": [0, 1, 2], "sub": [3, 4, 5]},
                           colors={"rev": "green", "sub": "blue"},
                           labels = {"rev": "reverberating", "sub": "sub-critical"},
                           plot_std=False,
                           plot_mV_params=None,
                           plot_mv_function=plot_synapses_singleexp):
    plot_mV = plot_mV_params is not None
    def load_weights(syn, tindex):
        indices = syn['a'][tindex] > 0
        a = syn['a'][tindex][indices]
        return a

    syns = [pickle_load(build, f"syn{type}_a", print_attrs=False) for build in builds]
    alphas = np.linspace(0.3, 1.0, len(tindexs))
    wbins = np.linspace(xlinspace[0], xlinspace[1], nlinspace)
    wbinw = np.diff(wbins)
    wbinc = (wbins[1:] + wbins[:-1]) / 2
    winbc_for_plotting = weight_to_psp(10**wbinc, f=plot_mv_function, **plot_mV_params) if plot_mV else wbinc

    for tindex, alpha in zip(tindexs, alphas):
        weights = [load_weights(syn, tindex) for syn in syns]
        histsw = np.array([np.histogram(np.log10(build_weights), bins=wbins, density=True)[0] for build_weights in weights])
        # wmeans = {"rev": np.mean(histsw[devision_by_regime['rev'], :], axis=0),
        #           "sub": np.mean(histsw[devision_by_regime['sub'], :], axis=0)}
        wmeans = {k:np.mean(histsw[devision_by_regime[k], :], axis=0) for k in devision_by_regime.keys()}
        if plot_std:
            wstds = {k:np.std(histsw[devision_by_regime[k], :], axis=0) for k in devision_by_regime.keys()}
            # wstds = {"rev": np.std(histsw[devision_by_regime['rev'], :], axis=0),
            #          "sub": np.std(histsw[devision_by_regime['sub'], :], axis=0)}
        for reg, wmean in wmeans.items():
            color = colors[reg]
            t = syns[0]['t'][tindex]

            if plot_mV:
                print("average synapse strength", np.sum(wmean*(10**wbinw)*(10**wbinc)), "mV for tindex", tindex, reg)

            ax.plot(winbc_for_plotting, wmean, color=color, alpha=alpha, label=t)
            if plot_std:
                ax.fill_between(winbc_for_plotting, wmean - wstds[reg], wmean + wstds[reg], color=color, alpha=0.2)

    if show_threshold:
        show_threshold = show_threshold if isinstance(show_threshold, str) else 'amin'
        threshold_x = np.log10(builds[0][2][show_threshold])
        if plot_mV:
            threshold_x = weight_to_psp([10 ** threshold_x], f=plot_mv_function, **plot_mV_params)
        ax.axvline(threshold_x, 0, 1, color="red")

    if plot_mV:
        ax.set_xlabel("PSP amplitude [mV]")
    else:
        ax.set_xlabel("log weight")
    ax.set_ylabel("density")
    ax.set_title(f"{type.upper()} weight distribution over time")

    import matplotlib.lines as mlines
    mhandles, mlabels = [], []
    if show_threshold:
        mhandles.append(mlines.Line2D([], [], color='red'))
        mlabels.append("prun. thr.")

    if xlim is not None:
        ax.set_xlim(*xlim)

    import matplotlib.lines as mlines
    for l in labels.keys():
        mhandles.append(mlines.Line2D([], [], color=colors[l], ls="-"))
        mlabels.append(labels[l])
    ax.legend(mhandles, mlabels, handlelength=0.75)
    syninfo = pickle_load(builds[0], "synee_a", print_attrs=False)
    print("ts", " ".join([str(t) for t in zip(tindexs, syninfo['t'][tindexs])]))


def raster_plot(ax, bpath, nsp, tmin, tmax, show_xlabeltick=True, show_labels=True):
    with open(bpath +'/raw/gexc_spks.p', 'rb') as pfile:
        GExc_spks = pickle.load(pfile)
    with open(bpath +'/raw/ginh_spks.p', 'rb') as pfile:
        GInh_spks = pickle.load(pfile)

    try:
        indx = np.logical_and(GExc_spks['t' ] /ms >tmin /ms, GExc_spks['t' ] /ms <tmax /ms)
        ax.plot(GExc_spks['t'][indx]/second, GExc_spks['i'][indx],
                marker=',', color='blue',
                linestyle='None')
    except AttributeError:
        print(bpath[-4:], "reports: AttributeError. Guess: no exc. spikes from",
              "{:d}s to {:d}s".format(int(tmin /second) ,int(tmax /second)))

    try:
        indx = np.logical_and(GInh_spks['t' ] /ms >tmin /ms, GInh_spks['t' ] /ms <tmax/ms)
        ax.plot(GInh_spks['t'][indx ] /second,
                GInh_spks['i'][indx ] +nsp['N_e'], marker=',',
                color='red', linestyle='None')
    except AttributeError:
        print(bpath[-4:], "reports: AttributeError. Guess: no inh. spikes from",
              "{:d}s to {:d}s".format(int(tmin /second) ,int(tmax /second)))


    ax.set_xlim(tmin/second, tmax /second)
    if show_labels:
        ax.set_ylabel('neuron')
    ax.set_ylim(0, nsp['N_e'] + nsp['N_i'])

    # ax.set_title('T='+str(T/second)+' s')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_ticks([])

    if show_xlabeltick:
        ax.xaxis.set_ticks([tmin/second, tmax/second])
        ax.xaxis.set_ticklabels([f"{tmin / second:.0f}", f"{tmax / second:.0f}"])
        ax.set_xlabel('time [second]', labelpad=-22.5)
        ax.xaxis.set_ticks_position('bottom')

    if show_labels:
        ax.text(1.03, 0.875, "I", color="red", transform=ax.transAxes)
        ax.text(1.03, 0.4, "E", color="blue", transform=ax.transAxes)

def population_activity(ax, bpath, nsp, tmin, tmax, show_xlabeltick=True, bs=1*second, offset=0.0*second):
    with open(bpath +'/raw/gexc_spks.p', 'rb') as pfile:
        GExc_spks = pickle.load(pfile)
    with open(bpath +'/raw/ginh_spks.p', 'rb') as pfile:
        GInh_spks = pickle.load(pfile)

    tbins = np.arange(tmin+offset, tmax+bs+offset, bs)
    tbinsc = (tbins[1:] + tbins[:-1])/2

    try:
        indx = np.logical_and(GExc_spks['t' ] /ms >tmin /ms, GExc_spks['t' ] /ms <tmax /ms)
        hist = np.histogram(GExc_spks['t'][indx], bins=tbins)[0]
        ax.step(tbinsc, hist, color='blue', where='mid')
        ax.plot(tbinsc, hist, color='blue', ls='none', marker='x')
    except AttributeError:
        print(bpath[-4:], "reports: AttributeError. Guess: no exc. spikes from",
              "{:d}s to {:d}s".format(int(tmin /second) ,int(tmax /second)))

    ax.set_xlim(tmin/second, tmax /second)
    ax.set_ylabel("active neuron count")

def population_activity_unique(ax, bpath, nsp, tmin, tmax, show_xlabeltick=True, bs=1*second, offset=0.0*second, show_markers=True):
    import pandas as pd
    with open(bpath +'/raw/gexc_spks.p', 'rb') as pfile:
        GExc_spks = pickle.load(pfile)

    tbins = np.arange(tmin+offset, tmax+bs+offset, bs)
    tbinsc = (tbins[1:] + tbins[:-1])/2

    try:
        indx = np.logical_and(GExc_spks['t'] /ms >tmin /ms, GExc_spks['t'] /ms <tmax /ms)
        df = pd.DataFrame(data=dict(t=GExc_spks['t'][indx], i=GExc_spks['i'][indx]))
        df_unique = df.groupby(pd.cut(df['t'], tbins))['i'].nunique()
        ax.step(tbinsc, df_unique, color='blue', where='mid')
        if show_markers:
            ax.plot(tbinsc, df_unique, color='blue', ls='none', marker='x')
    except AttributeError:
        print(bpath[-4:], "reports: AttributeError. Guess: no exc. spikes from",
              "{:d}s to {:d}s".format(int(tmin /second) ,int(tmax /second)))

    ax.set_xlim(tmin/second, tmax /second)
    ax.set_ylabel("unique active neurons")

def population_activity_linreg(ax, bpath, nsp, tmin, tmax, show_xlabeltick=True, bs=1*second, offset=0.0*second, plot=True):
    import scipy.stats as sstats
    with open(bpath +'/raw/gexc_spks.p', 'rb') as pfile:
        GExc_spks = pickle.load(pfile)
    with open(bpath +'/raw/ginh_spks.p', 'rb') as pfile:
        GInh_spks = pickle.load(pfile)

    tbegin = tmin+offset
    tbins = np.arange(tbegin, tmax+bs+offset, bs)
    tbinsc = (tbins[1:] + tbins[:-1])/2

    try:
        indx = np.logical_and(GExc_spks['t'] /ms >tmin /ms, GExc_spks['t'] /ms <tmax /ms)
        hist = np.histogram((GExc_spks['t'][indx]/second - tbegin/second),
                            bins=tbins-tbegin/second)[0]
        res = sstats.linregress(tbinsc-tbegin/second, hist)
        print(res)
        if plot:
            ax.step(tbinsc-tbegin/second, hist, color='blue', where='mid')
            ax.plot(tbinsc-tbegin/second, hist, color='blue', ls='none', marker='x')
            ax.plot(tbinsc-tbegin/second, res.intercept + res.slope*(tbinsc-tbegin/second), color='red')
    except AttributeError:
        print(bpath[-4:], "reports: AttributeError. Guess: no exc. spikes from",
              "{:d}s to {:d}s".format(int(tmin /second) ,int(tmax /second)))

    if plot:
        ax.set_xlim((tmin-tbegin)/second, (tmax-tbegin) /second)
        ax.set_ylabel("active neuron count")
        ax.set_xlabel("time (shifted to zero) [s]")


def time_to_end_of(analysis, build, index):
    nsp0 = analysis.get_nsp(build)
    return np.sum([v for k, v in nsp0.items() if k[0] == 'T' and k[1].isdigit() and int(k[1]) <= index])*second


## ISI & Firing Rates
def get_spk_data(build, nsp, pop):
    assert pop == 'exc' or pop == 'inh'
    Nneuron = nsp['N_e'] if pop == 'exc' else nsp['N_i']
    Sfile = 'gexc_spks' if pop == 'exc' else 'ginh_spks'
    spks = pickle_load(build, Sfile, print_attrs=False)
    return Nneuron, spks


def calc_isi_all_neurons(build, nsp, pop, tmin, tmax):
    Nneuron, spks = get_spk_data(build, nsp, pop)
    isis = []
    for i in range(Nneuron):
        isi = calc_isi(spks, tmin, tmax, i)
        isis.append(isi)
    isis = np.hstack(isis)
    return isis


def calc_fr_all_neurons(build, nsp, pop, tbegin, tend):
    Nneuron, spks = get_spk_data(build, nsp, pop)
    indx = np.logical_and(spks['t'] < tend, spks['t'] > tbegin)
    t_exc, id_exc = spks['t'][indx], spks['i'][indx]
    unique = np.unique(spks['i'][indx])
    T = tend - tbegin
    fr_exc = [np.sum(id_exc == i) / (T / second) for i in unique]
    return fr_exc


def calc_isi_rate(analysis, builds):
    import itertools

    nsp = analysis.get_nsp(builds[0])
    tsearly = 0.0 * ms, nsp['T1']
    tslate = nsp['T1'] + nsp['T2'], nsp['T1'] + nsp['T2'] + nsp['T3']
    print("early", *tsearly, "late", *tslate)

    spks = {"exc": {"early": {"isi": [], "rate": []}, "late": {"isi": [], "rate": []}},
            "inh": {"early": {"isi": [], "rate": []}, "late": {"isi": [], "rate": []}}}
    for build in builds:
        print(build[1])
        nsp = analysis.get_nsp(build)
        tsearly = 0.0 * ms, nsp['T1']
        # tslate = nsp['T1']+nsp['T2']+nsp['T3']+nsp['T4'], nsp['T1']+nsp['T2']+nsp['T3']+nsp['T4']+nsp['T5']
        tslate = nsp['T1'] + nsp['T2'], nsp['T1'] + nsp['T2'] + nsp['T3']
        tss = {"early": tsearly, "late": tslate}
        for pop, tskey in itertools.product(spks.keys(), tss.keys()):
            ts = tss[tskey]
            spks[pop][tskey]['isi'].append(calc_isi_all_neurons(build, nsp, pop, *ts))
            spks[pop][tskey]['rate'].append(calc_fr_all_neurons(build, nsp, pop, *ts))
    return spks

def print_hierarchy(d, n=0):
    if type(d) is not dict:
        return
    for key, value in d.items():
        print("".join(["|" if n > 0 else ""] + ["-" for _ in range(n)] + [key]))
        print_hierarchy(value, n+1)

def calc_isi_rate_means_stds(spks, bins):
    (isi_bins, isi_binc), (rate_bins, rate_binc) = bins
    hists = {kpop: {ktime: {'isi': [np.histogram(build, bins=isi_bins, density=True)[0] for build in builds['isi']],
                            'rate': [np.histogram(build, bins=rate_bins, density=True)[0] for build in builds['rate']]
                            } for ktime, builds in vpop.items()} for kpop, vpop in spks.items()}
    means, stds = {'isi': {}, 'rate': {}}, {'isi': {}, 'rate': {}}
    for key in means.keys():
        means[key]['rev'] = {kpop: {ktime: np.mean(build_hists[key][:3], axis=0) for ktime, build_hists in vpop.items()}
                             for kpop, vpop in hists.items()}
        stds[key]['rev'] = {kpop: {ktime: np.std(build_hists[key][:3], axis=0) for ktime, build_hists in vpop.items()}
                            for kpop, vpop in hists.items()}
        means[key]['sub'] = {kpop: {ktime: np.mean(build_hists[key][3:], axis=0) for ktime, build_hists in vpop.items()}
                             for kpop, vpop in hists.items()}
        stds[key]['sub'] = {kpop: {ktime: np.std(build_hists[key][3:], axis=0) for ktime, build_hists in vpop.items()}
                            for kpop, vpop in hists.items()}
    return means, stds


def plot_isi_rate_data(ax, binc, means, stds, path, **kwargs):
    ax.plot(binc, means[path[0]][path[1]], **kwargs)
    ax.fill_between(binc, means[path[0]][path[1]]-stds[path[0]][path[1]], means[path[0]][path[1]]+stds[path[0]][path[1]], color=kwargs['color'], alpha=0.1)


def plot_isi_rate_data_regime(ax, binc, title, means, stds, titleargs={}):
    ax.set_title(title, **titleargs)
    plot_isi_rate_data(ax, binc, means, stds, ['exc', 'early'], color="blue", label="exc. start")
    plot_isi_rate_data(ax, binc, means, stds, ['exc', 'late'], color="darkblue", label="exc. end")
    plot_isi_rate_data(ax, binc, means, stds, ['inh', 'early'], color="red", label="inh. start")
    plot_isi_rate_data(ax, binc, means, stds, ['inh', 'late'], color="darkred", label="inh. end")


def plot_2x2_isi_rate_regime(axs, bins, means, stds):
    import itertools
    (_, isi_binc), (_, rate_binc) = bins
    (ax1, ax), (ax3, ax2) = axs
    plot_isi_rate_data_regime(ax, isi_binc, "ISI reverberating", means['isi']['rev'], stds['isi']['rev'])
    plot_isi_rate_data_regime(ax1, isi_binc, "ISI sub-critical", means['isi']['sub'], stds['isi']['sub'])
    plot_isi_rate_data_regime(ax2, rate_binc, "firing rates reverberating", means['rate']['rev'], stds['rate']['rev'])
    plot_isi_rate_data_regime(ax3, rate_binc, "firing rates sub-critical", means['rate']['sub'], stds['rate']['sub'])
    for ax in axs[0]:
        ax.set_xlabel("ISI [ms]")
    for ax in axs[1]:
        ax.set_xlabel("firing rate [Hz]")

    ax1.set_ylabel("density")
    ax3.set_ylabel("density")

    for ax in list(itertools.chain(*axs)):
        ax.legend()


def get_isi_rate_bins(isi_count=100):
    isi_bins = np.linspace(0.0, 1000, isi_count)
    isi_binc = (isi_bins[:-1] + isi_bins[1:]) / 2
    rate_bins = np.arange(0, np.max(10.0), 0.25)
    rate_binc = (rate_bins[:-1] + rate_bins[1:]) / 2
    return (isi_bins, isi_binc), (rate_bins, rate_binc)


def plot_surviving_synapses(fig, ax, builds, fontsize):
    from multisim.branching_surviving import SurvivingSynapses
    directories = [f"{build[0]}/builds/{build[1]}" for build in builds]
    plotterin = SurvivingSynapses(in_notebook=True)
    plotterin.directories = directories
    plotterin._prepare()
    nsps = plotterin._get_nsps(plotterin.directories)
    plotterin.plot(plotterin.directories, nsps, fig, [[ax]], fontsize=fontsize)
    ax.set_title("Permanent synapses", fontsize=fontsize)
    ax.set_ylabel("fraction", fontsize=fontsize)
    # ax.set_yticklabels(["0.00", "0.01"], fontsize=fontsize)
    ax.set_xticklabels(["sub-critical", "reverberating"], fontsize=fontsize)


def plot_survival_density(fig, ax, builds, smallfontsize, basefontsize,
                          inset_sizepos = None, plot_inset=True, no_subcritical=False):
    from analysis.multisim.srvprb_branching import SrvPrbBranching
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    directories = [f"{build[0]}/builds/{build[1]}" for build in builds]

    plotter = SrvPrbBranching(in_notebook=True, no_subcritical=no_subcritical)
    plotter.directories = directories
    plotter._prepare()
    nsps = plotter._get_nsps(plotter.directories)

    ax = [[ax]]

    plotter.plot(plotter.directories, nsps, fig, ax, fontsize=basefontsize)

    if plot_inset:
        #                                                        left bottom width height
        inargs = dict(width="100%", height="100%", bbox_to_anchor=[0.12, 0.1, 0.63, 0.36])
        if inset_sizepos is not None:
            inargs.update(inset_sizepos)
        axin = inset_axes(ax[0][0], bbox_transform=ax[0][0].transAxes, **inargs)
        plot_surviving_synapses(fig, axin, builds, smallfontsize)

    ax[0][0].set_title("Survival times of EE synapses", fontsize=basefontsize)
    ax[0][0].legend(loc='upper right', handlelength=0.75, fontsize=basefontsize)
    ax[0][0].set_ylim(10 ** -7.1, 10 ** -2.3)


def plot_kesten_survival_density(fig, ax, dir_groups, basefontsize, colors=None):
    from analysis.kesten_sim.srvprb import SrvPrbBranching
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    plotter = SrvPrbBranching(in_notebook=True, colors=colors)
    from argparse import Namespace
    plotter._process_args(Namespace(group=dir_groups))
    plotter._prepare()

    ax = [[ax]]

    plotter.plot(plotter.directories, None, fig, ax, fontsize=basefontsize)

    ax[0][0].set_title("Survival times of EE synapses", fontsize=basefontsize)
    ax[0][0].legend(loc='upper right', handlelength=0.75, fontsize=basefontsize)
    ax[0][0].set_ylim(10 ** -7.1, 10 ** -2.3)


def calc_synapse_event_rates(builds,
                             devision_by_regime={"rev": [0, 1, 2], "sub": [3, 4, 5]},
                             timeunit=1000,
                             xfirst = None, xsecond = None,
                             yfirst = None, ysecond = None, ymax = None,
                             Tmax=200_000):
    import pandas as pd

    turnovers = []
    for build in builds:
        turnovers.append(
            pd.DataFrame(pickle_load(build, "turnover", print_attrs=False),
                         columns=["type", "t", "i", "j"]).astype({"type": bool, "t": int, "i": int, "j": int}))

    tbegin, tend = np.min(turnovers[0]['t']), np.max(turnovers[0]['t'])
    bins = np.arange(tbegin, tend, 1)
    binc = (bins[1:] + bins[:-1]) / 2

    eventtype = False
    hists = np.array([np.histogram(turnover.where(turnover['type'] == eventtype)['t'], bins=bins, density=False)[0] for turnover
             in turnovers])
    means = {k : np.mean(hists[devision_by_regime[k], :], axis=0) for k in devision_by_regime.keys()}
    stds = {k: np.std(hists[devision_by_regime[k], :], axis=0) for k in devision_by_regime.keys()}

    T = Tmax
    # xfirst, xsecond = 600/timeunit, 78_800/timeunit
    if xfirst is None:
        xfirst, xsecond = 600 / timeunit, (T - 1200) / timeunit
        yfirst, ysecond, ymax = 50, 300, 6_000

    width_ratio_r = (T / timeunit - xsecond) // xfirst

    return dict(
        eventtype=eventtype, means=means, stds=stds, binc=binc, timeunit=timeunit,
        yfirst=yfirst, ysecond=ysecond, ymax=ymax, xfirst=xfirst, xsecond=xsecond, T=T,
        width_ratio_r=width_ratio_r
    )


def plot_synapse_event_rates(fig, axs, eventtype, means, stds, binc, timeunit, yfirst, ysecond, ymax, xfirst, xsecond, T,
                             width_ratio_r, titlefontsize=40, ylabel_x=0.01, xlabel_y=0.005, title_y=0.93,
                             labels={"rev": "rev. reg.", "sub": "sub. reg."},
                             colors={"rev": "green", "sub": "blue"}):
    import itertools

    ((axol, axor), (axbl, axbr)) = axs

    eventtypestr = "creation" if eventtype else "pruning"
    for reg, mean in means.items():
        for ax in itertools.chain(*axs):
            ax.plot(binc / timeunit, mean, color=colors[reg], label=labels[reg], lw=0.75)
            ax.fill_between(binc / timeunit, mean - stds[reg], mean + stds[reg], color=colors[reg], alpha=0.2)
        ax = None

    axol.set_ylim(ysecond, ymax)
    axbl.set_ylim(0, yfirst)
    axor.set_ylim(ysecond, ymax)
    axbr.set_ylim(0, yfirst)

    axol.set_xlim(0, xfirst)
    axbl.set_xlim(0, xfirst)
    axor.set_xlim(xsecond, T / timeunit)
    axbr.set_xlim(xsecond, T / timeunit)

    axol.spines['bottom'].set_visible(False)
    axor.spines['bottom'].set_visible(False)
    axbl.spines['top'].set_visible(False)
    axbr.spines['top'].set_visible(False)
    axol.xaxis.tick_top()
    axor.xaxis.tick_top()
    axol.tick_params(labeltop=False)
    axor.tick_params(labeltop=False)
    axbl.xaxis.tick_bottom()
    axbr.xaxis.tick_bottom()

    axol.spines['right'].set_visible(False)
    axbl.spines['right'].set_visible(False)
    axor.spines['left'].set_visible(False)
    axbr.spines['left'].set_visible(False)
    axor.yaxis.tick_right()
    axbr.yaxis.tick_right()
    axor.tick_params(labelright=False)
    axbr.tick_params(labelright=False)

    # looks like tick marks are necessarily not in order of display
    ticklabels = axbr.get_xticklabels()
    # for ticklabel in ticklabels:
    #     ticklabel.set_ha('right')
    # ticklabels[len(ticklabels)-1].set_ha('right')
    # ticklabels[1].set_ha('left')
    ticklabels = axbl.get_xticklabels()
    # ticklabels[0].set_ha('left')

    fig.text(0.5, xlabel_y, "time [ks]", ha='center')
    fig.text(ylabel_x, 0.5, "events [per second]", va='center', rotation='vertical')
    fig.text(0.5, title_y, f"synaptic {eventtypestr} events", ha='center', fontsize=titlefontsize, weight='normal')

    axor.legend(loc='upper right')

    # break lines
    d = .035  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them

    axol.plot((-d, +d), (-d, +d), transform=axol.transAxes, color='k', clip_on=False)
    axol.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=axol.transAxes, color='k', clip_on=False)
    axbl.plot((-d, +d), (1 - d, 1 + d), transform=axbl.transAxes, color='k', clip_on=False)
    axbl.plot((1 - d, 1 + d), (-d, +d), transform=axbl.transAxes, color='k', clip_on=False)
    c = d / width_ratio_r
    axor.plot((-c, +c), (1 - d, 1 + d), transform=axor.transAxes, color='k', clip_on=False)
    axor.plot((1 - c, 1 + c), (-d, +d), transform=axor.transAxes, color='k', clip_on=False)
    axbr.plot((-c, +c), (-d, +d), transform=axbr.transAxes, color='k', clip_on=False)
    axbr.plot((1 - c, 1 + c), (1 - d, 1 + d), transform=axbr.transAxes, color='k', clip_on=False)


# STDP weight change

def stdp_calc_poisson(build):
    from brian2 import start_scope, PoissonGroup, Synapses, run

    amin = build[2]['amin']
    syn = pickle_load(build, "synee_a", print_attrs=False)
    indices = syn['a'][-1] > 0
    a_sim = syn['a'][-1][indices]

    a_init = a_sim[:452 * 451]
    # %%
    np.random.seed(42)  # TODO do not hardcode

    start_scope()

    # TODO do not hard-code
    stdp_eta = 0.01
    Aplus = 1.6253 * stdp_eta
    Aminus = -0.8127 * stdp_eta
    amin = 0.0363
    amax = 0.1622 * 2
    taupre = 15 * ms
    taupost = 30 * ms

    G = PoissonGroup(452, 2.5 * Hz, dt=0.1 * ms)

    S = Synapses(G, G, model='''
        dApre  /dt = -Apre/taupre  : 1 (event-driven)
        dApost /dt = -Apost/taupost : 1 (event-driven)
        a : 1
    ''', on_pre="""
        Apre = Aplus
        a = clip(a+Apost, amin, amax)
    """,
                 on_post="""
        Apost = Aminus
        a = clip(a+Apre, amin, amax)
     """)
    S.connect('i!=j')
    S.a = a_init
    # %%
    run(1 * second)
    # %%
    a_end = S.a[:]
    # %%

    return a_init, a_end

def stdp_calc_poisson_diff(a_init, a_end):
    bins = np.linspace(-0.04, 0.06, 75)
    centers = (bins[1:] + bins[:-1]) / 2
    # calculate for poisson
    wdw_poisson = np.histogram(a_end - a_init, bins=bins, density=True)[0]
    return bins, centers, wdw_poisson


def stdp_calc_for_builds(builds, bins):
    from notebookutils.stdp import calc_stdp_weight_diff
    # calculate for builds
    wdw = [calc_stdp_weight_diff(build, return_weights=True) for build in builds]
    wdw_dist = [np.histogram(dw['dw'], bins=bins, density=True)[0] for dw in wdw]
    rev_mean, rev_std = np.mean(wdw_dist[:3], axis=0), np.std(wdw_dist[:3], axis=0)
    sub_mean, sub_std = np.mean(wdw_dist[3:], axis=0), np.std(wdw_dist[3:], axis=0)

    return rev_mean, rev_std, sub_mean, sub_std


def plot_stdp_weight_change(ax1, centers, wdw_poisson, rev_mean, rev_std, sub_mean, sub_std, inargs=None, labels=None, legendargs=None):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.patches as mpatches

    xlim = (-0.02, 0.04)
    ylim = (0, 40)
    width = xlim[1] - xlim[0]
    height = ylim[1] - ylim[0]

    if labels is None:
        labels = {'rev. reg.': 'rev. reg.', 'sub-critical': 'sub-critical'}

    ax1.set_title("STDP induced weight change")
    if rev_mean is not None:
        ax1.plot(centers, rev_mean, color="green", label=labels["rev. reg."])
        ax1.fill_between(centers, rev_mean - rev_std, rev_mean + rev_std, color="lightgreen")
    if sub_mean is not None:
        ax1.plot(centers, sub_mean, color="blue", label=labels["sub-critical"])
        ax1.fill_between(centers, sub_mean - sub_std, sub_mean + sub_std, color="lightblue")
    if wdw_poisson is not None:
        ax1.plot(centers, wdw_poisson, color="orange", label="Poisson")
    ax1.set_ylim(*ylim)
    ax1.set_xlim(*xlim)
    ax1.set_xlabel("$\Delta$w/second")
    ax1.set_ylabel("density")
    if legendargs is None:
        legendargs = dict(bbox_to_anchor=(1.02, 0.08), handlelength=0.75)
    ax1.legend(loc='lower right', **legendargs)  # , handletextpad=0.3, borderpad=0.28

    #                                                          left bottom width height
    if inargs is None:
        inargs = dict(width="100%", height="100%", bbox_to_anchor=[0.67, 0.63, 0.35, 0.40])
    ax0 = inset_axes(ax1, bbox_transform=ax1.transAxes, **inargs)

    # ax0.set_title("all synapses")
    lw = 0.5
    if rev_mean is not None:
        ax0.plot(centers, rev_mean, color="green", label=labels["rev. reg."], lw=lw)
        ax0.fill_between(centers, rev_mean - rev_std, rev_mean + rev_std, color="lightgreen")
    if sub_mean is not None:
        ax0.plot(centers, sub_mean, color="blue", label=labels["sub-critical"], lw=lw)
        ax0.fill_between(centers, sub_mean - sub_std, sub_mean + sub_std, color="lightblue")
    if wdw_poisson is not None:
        ax0.plot(centers, wdw_poisson, color="orange", label="Poisson", lw=lw)
    ax0.tick_params(labelleft=False, labelbottom=False)
    ax0.add_patch(
        mpatches.Rectangle((xlim[0], ylim[0]), width, height, edgecolor='black', facecolor='none', zorder=100, lw=1.5))
    ax0.set_xlim(-0.03, 0.06)

    ixlim = ax0.get_xlim()
    iylim = ax0.get_ylim()
    # print("inset aspect ratio", (iylim[1]-iylim[0])/(ixlim[1]-ixlim[0]))

    oxlim = ax1.get_xlim()
    oylim = ax1.get_ylim()
    # print("outside aspect ratio", (oylim[1]-oylim[0])/(oxlim[1]-oxlim[0]))


def bin_by_ms(binned, bin_size):
    N_per_bin = int((bin_size) / (0.1 * ms))
    N_bins = int(len(binned) / N_per_bin)
    return np.sum(np.reshape(binned, (N_bins, N_per_bin)), axis=1)

def indexms(T: Quantity, bin_size: Quantity):
    return int((T / ms) / (bin_size / ms))


def spike_binning(builds, bin_size = 1 * second):
    N_e, N_i = 1600, 320
    binned_builds = []
    binned_builds_i = []
    for build in builds:
        binned = pickle_load(build, 'gexc_binned')['spk_count']
        binned_by_ms = bin_by_ms(binned, bin_size)
        binned_builds.append(binned_by_ms / N_e)

        binned = pickle_load(build, 'ginh_binned')['spk_count']
        binned_by_ms = bin_by_ms(binned, bin_size)
        binned_builds_i.append(binned_by_ms / N_i)
    return binned_builds, binned_builds_i


def mre_for_range(Tstart, Tend, binned_by_ms, bin_size):
    import mrestimator
    from mrestimator import utility as ut
    ut.log.setLevel('ERROR')
    print(Tstart, Tend, indexms(Tstart, bin_size), indexms(Tend, bin_size))
    spiket = binned_by_ms[indexms(Tstart, bin_size):indexms(Tend, bin_size)]
    coeff = mrestimator.coefficients(spiket, dt=bin_size/ms, dtunit='ms')
    ft = mrestimator.fit(coeff)
    return coeff, ft


def calc_for_bin_size(builds, bin_size, times, T=250*second):
    """
    :param builds:
    :param bin_size:
    :param times: beginning of sections to calculate m from
    :param T: size of section to calculate m from
    :return:
    """
    import os.path
    path = os.path.join(builds[0][0], f'movertime_{bin_size}.p')
    print("storage", path)
    if os.path.exists(path):
        print("from cache")
        return path

    print("times", times/ksecond)
    times_regime_mre = [ [] for _ in times ]
    for build in builds:
        binned = pickle_load(build, 'gexc_binned')['spk_count']
        binned_by_ms = bin_by_ms(binned, bin_size)
        for i, Tstart in enumerate(times):
            Tend = Tstart + T
            _, fit = mre_for_range(Tstart, Tend, binned_by_ms, bin_size=bin_size)
            times_regime_mre[i].append(fit.mre)
    with open(path, "wb") as f:
        pickle.dump({'times': times, 'times_regime_mre': times_regime_mre}, f)
    return path


def plot_mre(ax, path, analysis, builds, colors=None, labels=None):
    if colors is None:
        colors = ['darkgreen', 'lightgreen', 'green', 'royalblue', 'darkblue', 'blue']
    with open(path, "rb") as f:
        d = pickle.load(f)
    times, times_regime_mre = d['times'], d['times_regime_mre']
    times_regime_mre = np.array(times_regime_mre)
    nsp0 = analysis.get_nsp(builds[0])

    from matplotlib import cycler

    ax: plt.Axes

    if labels is None:
        labels = [build[1] for build in builds]

    tunit = ksecond
    ax.set_prop_cycle(cycler('color', colors))
    ax.set_title("branching factor over time")
    lines = ax.plot(times/tunit, times_regime_mre)
    ax.axhspan(0.90, 0.995, color="lightgreen", alpha=0.2, label="reverberating regime")
    # ax.axvline(Tfreeze/tunit, color="black")
    ax.set_ylabel("branching factor $m$")
    ax.set_xlabel("time [1000 seconds]")
    ax.set_ylim(0.0, 1.0)
    # ax.set_xlim(0*second/tunit, 201_000*second/tunit)
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.legend(lines, labels, ncol=2)
    # ax.set_xticks(times/tunit)


def plot_mre_for_binsize(simfigname, builds, analysis, bin_size, times, T, colors_lst, labels_lst):
    for (fig, ax, style) in plot_styles((1, 1), styles=[style_MA], name=f'{simfigname}/branching_factor_over_time_{bin_size}'):
        path = calc_for_bin_size(builds, bin_size=bin_size, times=times, T=T)
        plot_mre(ax, path, analysis, builds, colors=colors_lst, labels=labels_lst)
        ax.set_title(f"branching factor over time [dt=${bin_size/ms}$ ms]")

def eibalance(builds, binned_builds, binned_builds_i, buildi = 0):
    N_e, N_i = 1600, 320
    bin_size = 1*second
    esyninfo = pickle_load(builds[buildi], "synee_a", print_attrs=False)
    isyninfo = pickle_load(builds[buildi], "synei_a", print_attrs=False)
    eaa, iaa = esyninfo['a'], isyninfo['a']
    for tindex, t in enumerate(esyninfo['t'][:-1]):
        print(f'{t / second:3.0f} ks', end=': ')
        findex = indexms(t, bin_size)
        iincoming, eincoming = (np.sum(iaa[tindex]) / N_e), (np.sum(eaa[tindex]) / N_e)
        irate, erate = binned_builds_i[buildi][findex], binned_builds[buildi][findex]
        print(f'inh {irate:0.2f} Hz, exc {erate:0.2f} Hz', end=' - ')
        ibalance, ebalance = binned_builds_i[0][findex] * iincoming, binned_builds[0][findex] * eincoming
        print(f'inh {ibalance:7.4f} / second vs exc {ebalance:7.4f} / second')