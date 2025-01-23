"""
Computes box of velocity offsets between each voxel and the nearest 
neutral voxel Earthward along the LoS

Samuel Gagnon-Hartman, 2024
Scuola Normale Superiore, Pisa
"""

import time, os

import numpy as np
import py21cmfast as p21c

from numba import njit
from scipy import integrate, interpolate

from astropy import units as u
from astropy import constants as c
from astropy.cosmology import Planck18, z_at_value

def plot_LC_slice(d_transverse, xHI, density, vz):
    """
    Plot a slice of the LC cube.
    """
    # plot slice of LC cube
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True, constrained_layout=True)

    IDX = 50
    axs[0].set_aspect('equal')
    axs[1].set_aspect('equal')
    axs[2].set_aspect('equal')

    mesh0 = axs[0].pcolormesh(d_transverse, d_transverse, xHI[IDX], cmap='Purples')
    cb = plt.colorbar(mesh0, ax=axs[0])
    cb.set_label(r'$x_{\rm HI}$', fontsize=16)
    axs[0].set_xlabel('Mpc', fontsize=16)
    axs[0].set_ylabel('Mpc', fontsize=16)

    mesh1 = axs[1].pcolormesh(d_transverse, d_transverse, density[IDX], cmap='inferno')
    cb = plt.colorbar(mesh1, ax=axs[1])
    cb.set_label(r'$\delta$', fontsize=16)
    axs[1].set_xlabel('Mpc', fontsize=16)

    mesh2 = axs[2].pcolormesh(d_transverse, d_transverse, vz[IDX], cmap='magma')
    cb = plt.colorbar(mesh2, ax=axs[2])
    cb.set_label(r'$\Delta v_z$ [proper km/s]', fontsize=16)
    axs[2].set_xlabel('Mpc', fontsize=16)

    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/LC_slice.png', dpi=300)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", \
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})

    
    # import the LC cube and extract the relevant quantities
    LC = p21c.LightCone.read('./data/lightcones/LC.h5')
    xHI = LC.xH_box
    vz = LC.velocity_z
    density = LC.density

    DIM = xHI.shape[0]
    LoS_DIM = xHI.shape[-1]
    L = 300*u.Mpc
    res = L/DIM

    d_LoS = np.linspace(0, 1.5*LoS_DIM, int(LoS_DIM))*u.Mpc + Planck18.comoving_distance(5.00)
    d_LoS_edges = np.linspace(0, 1.5*LoS_DIM + 1.5, int(LoS_DIM)+1)*u.Mpc + Planck18.comoving_distance(5.00) - 1.5*u.Mpc/2

    z_LoS = np.array([z_at_value(Planck18.comoving_distance, d) for d in d_LoS])
    z_LoS_edges = np.array([z_at_value(Planck18.comoving_distance, d) for d in d_LoS_edges])
    d_transverse = np.linspace(0, 300, 200)

    z_LoS_cube = np.array([[z_LoS]*DIM]*DIM)
    vz_proper = (vz*u.Mpc/u.s).to('km/s').value/(1+z_LoS_cube)

    # get peculiar velocities in ionized regions
    QHI = 1 - xHI
    QHI = np.where(QHI < 0.1, 0.0, QHI)
    QHI = np.where(QHI > 0.0, 1.0, QHI)

    # make box of "last neutrals": bool True indicates a "Last" neutral voxel
    # e.g. the first neutral voxel encountered by a photon moving Earthward
    # from an ionized region
    states = np.where(QHI == 0.0, 1.0, 2.0)
    # states = np.where(QHI > 0.0, 2.0, QHI)
    transitions = states[...,1:] / states[...,:-1]
    last_neutral = np.where(transitions == 2.0, True, False)
    vz_last_neutral = np.where(last_neutral, vz_proper[...,1:], 0.0)

    # now scroll through and make sure each cell takes the peculiar
    # velocity of the last neutral cell encountered Earthward
    nearest_neutral_vz = np.copy(vz_proper)
    last_slice = vz_last_neutral.T[0]
    for i, vz_slice in enumerate(vz_last_neutral.T[1:]):
        last_slice = np.where(vz_slice != 0.0, vz_slice, last_slice)
        nearest_neutral_vz[...,i] = last_slice

    # get velocity offset between each voxel and the nearest neutral voxel
    # Earthward along the LoS
    dvz = vz_proper - nearest_neutral_vz
    dvz *= QHI

    # flat_dvz = dvz.flatten()
    # nonzero_dvz = flat_dvz[flat_dvz != 0.0]
    # plt.hist(nonzero_dvz, bins=100, histtype='step', color='k')
    # plt.show()

    plot_LC_slice(d_transverse, xHI, density, dvz)
    