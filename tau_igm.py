"""
Functions for calculating the rest-frame absorption spectrum
in a voxel of a light cone. 
"""

import numpy as np
import py21cmfast as p21c

def get_absorption_properties():
    LC = p21c.LightCone.read('./data/lightcones/sgh25.h5')
    xHI = LC.xH_box
    density = LC.density
    vz = LC.velocity_z

    # wavelength range
    wavelength_range = np.linspace(1215, 1220, 1000)

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
    prefactor = np.load('./data/absorption_properties/prefactor.npy')
    n_HI = np.load('./data/absorption_properties/n_HI.npy')

    voigt_tables = np.zeros_like(log_voigt_tables, dtype=np.float64)
    for i in range(len(z_LoS)):
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
    return xHI, density, vz, z_LoS, voigt_tables, rel_idcs_list, \
            z_LoS_highres_list, z_table_list, dz_list, prefactor, n_HI

def get_tau_igm(i, j, k, xHI, density, z_LoS, voigt_tables, rel_idcs_list, \
            z_LoS_highres_list, z_table_list, dz_list, prefactor, n_HI):
    """
    Computes the absorption spectrum in a voxel.
    """
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

    integrand = crosssec_table * n_HI_table * z_table
    dz_table = dz_list[k]
    tau = np.sum((integrand[:-1] + integrand[1:]) * dz_table / 2, axis=0)
    return tau

if __name__ == "__main__":

    xHI, density, vz, z_LoS, voigt_tables, rel_idcs_list, \
        z_LoS_highres_list, z_table_list, dz_list, prefactor, \
        n_HI = get_absorption_properties()

    tau_igm_table = np.zeros((1000, len(z_LoS)//2))
    for k in range(len(z_LoS)//2):
        tau_igm = get_tau_igm(0, 0, k, xHI, density, z_LoS, voigt_tables, rel_idcs_list, \
                z_LoS_highres_list, z_table_list, dz_list, prefactor, n_HI)
        tau_igm_table[:,k] = tau_igm

    np.save('./data/tau_igm_table.npy', tau_igm_table)
    np.save('./data/xHI_pencil.npy', xHI[0,0])
    np.save('./data/density_pencil.npy', density[0,0])
    np.save('./data/vz_pencil.npy', vz[0,0])

    dvz = np.load('./data/absorption_properties/dvz.npy')
    np.save('./data/dvz_pencil.npy', dvz[0,0])
