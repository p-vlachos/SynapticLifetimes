import itertools
import os
from brian2.units import *
import numpy as np

import h5py
import matplotlib
from typing import Dict

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle

rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [
    r'\usepackage{tgheros}',
    r'\usepackage{sansmath}',
    r'\sansmath'               
    r'\usepackage{siunitx}',
    r'\sisetup{detect-all}',
]

def ddcon_figure(bpath):
    nsp = None
    try:
        with open(bpath+'/raw/namespace.p', 'rb') as pfile:
            nsp=pickle.load(pfile)
    except FileNotFoundError:
        print(bpath[-4:], "reports: No namespace data. Skipping.")
        return

    hdf5 = h5py.File('data/hdf5_data.hdf5', 'r')
    print(bpath[-4:], bpath)
    run_results = hdf5['tr1']['results']['runs'][f'run_0000{bpath[-4:]}']
    GExc_x, GExc_y = run_results["GExc_x"]["GExc_x"]*meter, run_results["GExc_y"]["GExc_y"]*meter
    GInh_x, GInh_y = run_results["GInh_x"]["GInh_x"]*meter, run_results["GInh_y"]["GInh_y"]*meter
    EE_src, EE_tar = np.array(run_results["sEE_src"]["sEE_src"]), np.array(run_results["sEE_tar"]["sEE_tar"])
    EI_src, EI_tar = np.array(run_results["sEI_src"]["sEI_src"]), np.array(run_results["sEI_tar"]["sEI_tar"])

    with open(bpath + '/raw/synee_a.p', 'rb') as pfile:
        synee_a = pickle.load(pfile)
        syn_active_ee = synee_a['syn_active'][0]
        print(np.sum(synee_a['syn_active'][-1]), synee_a['syn_active'].shape)
        matrix = np.zeros(shape=(nsp['N_e'], nsp['N_e']))
        matrix[EE_src, EE_tar] = syn_active_ee
        EE_src, EE_tar = np.where(matrix == 1)

    with open(bpath + '/raw/synei_a.p', 'rb') as pfile:
        synei_a = pickle.load(pfile)
        syn_active_ei = synei_a['syn_active'][0]
        print(np.sum(synei_a['syn_active'][-1]), synei_a['syn_active'].shape)
        matrix = np.zeros(shape=(nsp['N_i'], nsp['N_e']))
        matrix[EI_src, EI_tar] = syn_active_ei
        EI_src, EI_tar = np.where(matrix == 1)

    fig: plt.Figure = plt.figure()
    ax_lines, ax_cols = 1,1
    axs: Dict[str, plt.Axes] = {}
    for x,y in itertools.product(range(ax_lines),range(ax_cols)):
        axs['%d,%d'%(x+1,y+1)] = plt.subplot2grid((ax_lines, ax_cols), (x, y))

    plt.gca().set_aspect('equal', adjustable='box')

    circle_index = 260  # TODO runtime parameter
    axs['1,1'].add_patch(plt.Circle((GExc_x[circle_index]/um, GExc_y[circle_index]/um), nsp['half_width']/um, color="lightgreen"))
    axs['1,1'].plot(GExc_x[circle_index]/um, GExc_y[circle_index]/um, color="black", ls="None", marker="x")

    axs['1,1'].plot(GExc_x/um, GExc_y/um, color="blue", ls="None", marker=".", markersize="0.5")
    axs['1,1'].plot(GInh_x/um, GInh_y/um, color="red", ls="None", marker=".", markersize="0.5")

    EE_src_onto_index = EE_src[EE_tar == circle_index]
    print(len(EE_src_onto_index))
    in_one_sigma = 0
    for src in EE_src_onto_index:
        axs['1,1'].plot(GExc_x[src]/um, GExc_y[src]/um,
                        color="blue",
                        linestyle="None",
                        marker="o",
                        markersize="1",
                        )
        if np.sqrt((GExc_x[src]/um - GExc_x[circle_index]/um)**2 + (GExc_y[src]/um - GExc_y[circle_index]/um)**2) <= 200:
            in_one_sigma += 1
    print("in_one_sigma", in_one_sigma)

    EI_src_onto_index = EI_src[EI_tar == circle_index]
    for src in EI_src_onto_index:
        axs['1,1'].plot(GInh_x[src]/um, GInh_y[src]/um,
                        color="red",
                        linestyle="None",
                        marker="o",
                        markersize="1",
                        )

    axs['1,1'].set_xlabel("x [$\\mu \\text{m}$]")
    axs['1,1'].set_ylabel("y [$\\mu \\text{m}$]")

    plt.tight_layout()

    directory = "figures/ddcon"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(f"{directory}/{bpath[-4:]}.png", dpi=600, bbox_inches='tight')


if __name__ == "__main__":
    # return a list of each build (simulations run)
    # e.g. build_dirs = ['builds/0003', 'builds/0007', ...]
    # sorted to ensure expected order
    build_dirs = sorted(['builds/' + pth for pth in next(os.walk("builds/"))[1]])

    for bpath in build_dirs:
        ddcon_figure(bpath)