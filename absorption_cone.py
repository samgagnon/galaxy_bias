"""
A code to calculate the rest-frame absorption spectrum
in each voxel of a light cone. 
"""

import time

import numpy as np
import py21cmfast as p21c

from numba import njit, prange
from scipy import interpolate

from astropy import units as u
from astropy import constants as c
from astropy.cosmology import Planck18, z_at_value

from utils import *

def delta_nu(T):
    """
    Returns the Doppler width of a Lya line with temperature T
    """
    return ((freq_Lya/c.c) * np.sqrt(2*c.k_B*T/c.m_p.to('kg'))).to('Hz')

def get_a(dnu):
    """
    Returns the damping parameter of a Lya line with Doppler width dnu
    """
    return (decay_factor/(4*np.pi*dnu)).to('')

def voigt_tasitsiomi(x, dnu):
    """
    Returns the Voigt profile of a line with damping parameter a
    """
    dL = 9.936e7*u.Hz
    a = 0.5*(dL/dnu)
    xt = x**2
    z = (xt - 0.855)/(xt + 3.42)
    q = np.zeros(len(x))
    IDX = (z > 0.0)
    q[IDX] = z[IDX]*(1 + 21/xt[IDX])*(a/np.pi/(xt[IDX] + 1.0))*\
        (0.1117 + z[IDX]*(4.421 + z[IDX]*(-9.207 + 5.674*z[IDX])))
    return (q + np.exp(-xt)/1.77245385)*np.sqrt(np.pi)

# change decorator to include parallel=True
@njit(parallel=True)
def main_loop(NLINES, xHI, density, z_LoS, DIM, LoS_DIM, wavelength_range,\
              voigt_tables, rel_idcs_list, z_LoS_highres_list, z_table_list, dz_list,\
              prefactor, n_HI):
    """
    Computes the absorption spectrum in each voxel of the light cone.
    """

    # use nlines when you only want to do a partial lightcone
    tau_IGM = np.zeros((DIM, DIM, LoS_DIM, len(wavelength_range)))
    # change outer loops to prange
    for i in prange(DIM):
        for j in prange(DIM):
            for k in prange(LoS_DIM):
                rel_idcs = rel_idcs_list[k]
                
                # extract relevant quantities
                xHI_rel = xHI[i, j, rel_idcs]
                density_rel = density[i, j, rel_idcs]
                z_LoS_rel = z_LoS[rel_idcs]
                n_HI_rel = n_HI[rel_idcs]

                z_LoS_highres = z_LoS_highres_list[k]
                xHI_rel = np.interp(z_LoS_highres, z_LoS_rel, xHI_rel)
                density_rel = np.interp(z_LoS_highres, z_LoS_rel, density_rel)
                n_HI_rel = np.interp(z_LoS_highres, z_LoS_rel, n_HI_rel)
                n_HI_rel *= (1 + density_rel)*xHI_rel

                # load voigt table and multiply by relevant quantities
                voigt_table = voigt_tables[k]
                crosssec_table = voigt_table*prefactor
                
                n_HI_table = np.zeros_like(crosssec_table)
                for l in range(len(n_HI_rel)):
                    n_HI_table[l] = n_HI_rel[l]

                z_table = z_table_list[k]
                
                # print(n_HI_table.dtype, crosssec_table.dtype, z_table.dtype)
                # print(n_HI_table.mean(), crosssec_table.mean(), z_table.mean())
                integrand = crosssec_table * n_HI_table * z_table
                dz_table = dz_list[k]
                tau_IGM[i,j,k] = np.sum((integrand[:-1] + integrand[1:]) * dz_table / 2, axis=0)
    return tau_IGM

if __name__ == "__main__":

    LC = p21c.LC.read('./data/lightcones/LC.h5')
    xHI = LC.xH_box
    vz = LC.velocity_z
    density = LC.density

    DIM = xHI.shape[0]
    LoS_DIM = xHI.shape[-1]
    L = 300*u.Mpc
    res = L/DIM

    # this file contains properties which are common to all lightcones 
    # of the same size, regardless of astrophysical model
    # it should be said that the z-dependent properties DO 
    # depend on the cosmology, I should be careful about this
    rel_idcs_list = np.loadtxt('./data/absorption_properties/rel_idcs_list.txt')
    log_voigt_tables = np.load('./data/absorption_properties/voigt_tables.npy')
    z_LoS_highres_list = np.load('./data/absorption_properties/z_LoS_highres_list.npy')
    z_table_constructor_list = np.load('./data/absorption_properties/z_table_list.npy')
    dz_constructor_list = np.load('./data/absorption_properties/dz_list.npy')
    z_LoS = np.load('./data/absorption_properties/z_LoS.npy')
    z_LoS_edges = np.load('./data/absorption_properties/z_LoS_edges.npy')
    d_LoS_edges = np.load('./data/absorption_properties/d_LoS_edges.npy')
    prefactor = np.load('./data/absorption_properties/prefactor.npy')
    n_HI = np.load('./data/absorption_properties/n_HI.npy')

    # from plot import plot_neutral_density
    # plot_neutral_density(z_LoS, n_HI)

    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", \
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})

    # plt.imshow(log_voigt_tables[50], origin='lower')
    # plt.colorbar()
    # plt.show()
    # quit()

    voigt_tables = np.zeros_like(log_voigt_tables, dtype=np.float64)
    for i in range(len(z_LoS)):
        # print(n_HI[i], voigt_tables[i].mean(), z_table_constructor_list[i].mean())
        voigt_tables[i] = 10**(log_voigt_tables[i].astype(np.float64))
        z_table_constructor_list[i] = 10**z_table_constructor_list[i]

    rel_idcs_list = rel_idcs_list.astype(int)
    indices = np.where(rel_idcs_list == -1)[0]    
    subarrays = np.split(rel_idcs_list, indices)
    rel_idcs_list = [subarray[subarray != -1] for subarray in subarrays]

    z_table_list = [None]*len(z_LoS)
    for i, jacobian_rel in enumerate(z_table_constructor_list):
        z_table_list[i] = np.array([jacobian_rel]*len(wavelength_range)).T
    
    dz_list = [None]*len(z_LoS)
    for i, dz in enumerate(dz_constructor_list):
        dz_list[i] = np.array([dz]*len(wavelength_range)).T

    tau_IGM = main_loop(DIM, xHI, density, z_LoS, DIM, LoS_DIM, wavelength_range,\
            voigt_tables, rel_idcs_list, z_LoS_highres_list, z_table_list, dz_list,\
            prefactor, n_HI)

    np.save('./data/absorption_properties/tau_IGM.npy', tau_IGM)

    # a = time.time()
    # wavelength_range, tau_IGM_list = main_loop(1, xHI, density, z_LoS, DIM, LoS_DIM, wavelength_range,\
    #         voigt_tables, rel_idcs_list, z_LoS_highres_list, z_table_list, dz_list,\
    #         prefactor, n_HI)
    # for tau_IGM in tau_IGM_list:
    #     plt.plot(wavelength_range, np.exp(-1*tau_IGM), color='black', alpha=0.1)
    # plt.ylabel(r'$\tau_{\mathrm{IGM}}$')
    # plt.xlabel(r'$\lambda$')
    # plt.show()
    # print(f'Finished main loop in {time.time() - a} seconds')

    # from plot import plot_transmissivity
    # plot_transmissivity(wavelength_range, np.exp(-1*tau_IGM))
    # plot_transmissivity(wavelength_range, tau_IGM)

    # calculate the absorption spectrum in each voxel
    # make a separate file for this to troubleshoot
    # apparently can pretty much process an entire lightcone in one second????
    # NLINES_list = [1, 10, 20, 50, 100]
    # execution_time = []
    # for NLINES in NLINES_list:
    #     a = time.time()
    #     main_loop(NLINES, xHI, density, z_LoS, DIM, LoS_DIM, wavelength_range,\
    #             voigt_tables, rel_idcs_list, z_LoS_highres_list, z_table_list, dz_list,\
    #             prefactor, n_HI)
    #     execution_time.append(time.time() - a)
    #     print(f'Finished main loop in {time.time() - a} seconds')
    # plt.plot(NLINES_list, execution_time)
    # plt.show()
    # print(f'Finished main loop in {time.time() - a} seconds')
