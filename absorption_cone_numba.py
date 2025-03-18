"""
A parallelized code to calculate the rest-frame absorption spectrum
in each voxel of a light cone. 
"""

import os
import numpy as np
from joblib import Parallel, delayed

def compute_tau_IGM_for_voxel(i, j, xHI, density, z_LoS, LoS_DIM, voigt_tables, \
                              rel_idcs_list, z_LoS_highres_list, z_table_list, \
                              dz_list, prefactor, n_HI):

    if os.path.exists(f'./data/absorption_properties/tau_IGM/tau_IGM_{i}_{j}_0.npy'):
        pass

    for k in range(1000): #LoS_DIM

        rel_idcs = rel_idcs_list[k]
        rel_idcs = rel_idcs[rel_idcs != -1]
        
        xHI_rel = xHI[i, j, rel_idcs]
        density_rel = density[i, j, rel_idcs]
        z_LoS_rel = z_LoS[rel_idcs]
        n_HI_rel = n_HI[rel_idcs]

        z_LoS_highres = z_LoS_highres_list[k]
        xHI_rel = np.interp(z_LoS_highres, z_LoS_rel, xHI_rel)
        density_rel = np.interp(z_LoS_highres, z_LoS_rel, density_rel)
        n_HI_rel = np.interp(z_LoS_highres, z_LoS_rel, n_HI_rel)
        n_HI_rel *= (1 + density_rel) * xHI_rel

        voigt_table = voigt_tables[k]
        crosssec_table = voigt_table * prefactor
        
        n_HI_table = np.zeros_like(crosssec_table, dtype=np.float32)
        for l in range(n_HI_rel.shape[0]):
            n_HI_table[l] = n_HI_rel[l]

        z_table = z_table_list[k]
        
        integrand = crosssec_table * n_HI_table * z_table
        dz_table = dz_list[k]
        tau_IGM_ijk = np.sum((integrand[:-1] + integrand[1:]) * dz_table / 2, axis=0)

        np.save(f'./data/absorption_properties/tau_IGM/tau_IGM_{i}_{j}_{k}.npy', tau_IGM_ijk)

def main_loop(xHI, density, z_LoS, DIM, LoS_DIM, voigt_tables, rel_idcs_list, \
              z_LoS_highres_list, z_table_list, dz_list, prefactor, n_HI):
    """
    Computes the absorption spectrum in each voxel of the light cone and saves each to its own file.
    """
    os.makedirs('./data/absorption_properties/tau_IGM', exist_ok=True)
    Parallel(n_jobs=-1)(delayed(compute_tau_IGM_for_voxel)(i, j, xHI, density, z_LoS, LoS_DIM,\
                                        voigt_tables, rel_idcs_list, z_LoS_highres_list, z_table_list, \
                                        dz_list, prefactor, n_HI) for i in range(DIM) for j in range(DIM))

if __name__ == "__main__":

    xHI = np.load('./data/absorption_properties/xHI.npy').astype(np.float32)
    vz = np.load('./data/absorption_properties/vz.npy').astype(np.float32)
    density = np.load('./data/absorption_properties/density.npy').astype(np.float32)

    DIM = xHI.shape[0]
    LoS_DIM = xHI.shape[-1]
    L = 300
    res = L / DIM

    # wavelength range
    wavelength_range = np.linspace(1215, 1220, 1000).astype(np.float32)

    # this file contains properties which are common to all lightcones 
    # of the same size, regardless of astrophysical model
    # it should be said that the z-dependent properties DO 
    # depend on the cosmology, I should be careful about this
    rel_idcs_list = np.loadtxt('./data/absorption_properties/rel_idcs_list.txt').astype(np.int32)
    log_voigt_tables = np.load('./data/absorption_properties/voigt_tables.npy').astype(np.float32)
    z_LoS_highres_list = np.load('./data/absorption_properties/z_LoS_highres_list.npy').astype(np.float32)
    z_table_constructor_list = np.load('./data/absorption_properties/z_table_list.npy').astype(np.float32)
    dz_constructor_list = np.load('./data/absorption_properties/dz_list.npy').astype(np.float32)
    z_LoS = np.load('./data/absorption_properties/z_LoS.npy').astype(np.float32)
    z_LoS_edges = np.load('./data/absorption_properties/z_LoS_edges.npy').astype(np.float32)
    d_LoS_edges = np.load('./data/absorption_properties/d_LoS_edges.npy').astype(np.float32)
    prefactor = np.load('./data/absorption_properties/prefactor.npy').astype(np.float32)
    n_HI = np.load('./data/absorption_properties/n_HI.npy').astype(np.float32)

    voigt_tables = np.zeros_like(log_voigt_tables, dtype=np.float32)
    for i in range(len(z_LoS)):
        voigt_tables[i] = 10**(log_voigt_tables[i].astype(np.float32))
        z_table_constructor_list[i] = 10**z_table_constructor_list[i]

    rel_idcs_list = rel_idcs_list.astype(np.int32)
    indices = np.where(rel_idcs_list == -1)[0]
    subarrays = np.split(rel_idcs_list, indices)
    rel_idcs_list = [subarray[subarray != -1] for subarray in subarrays]

    max_len = max([len(subarray) for subarray in rel_idcs_list])
    rel_idcs_list = [np.pad(subarray, (0, max_len - len(subarray)), 'constant', \
                        constant_values=(-1, -1)) for subarray in rel_idcs_list]

    z_table_list = [None] * len(z_LoS)
    for i, jacobian_rel in enumerate(z_table_constructor_list):
        z_table_list[i] = np.array([jacobian_rel] * len(wavelength_range)).T.astype(np.float32)
    
    dz_list = [None] * len(z_LoS)
    for i, dz in enumerate(dz_constructor_list):
        dz_list[i] = np.array([dz] * len(wavelength_range)).T.astype(np.float32)

    xHI = np.asarray(xHI, dtype=np.float32)
    density = np.asarray(density, dtype=np.float32)
    z_LoS = np.asarray(z_LoS, dtype=np.float32)
    DIM = np.int32(DIM)
    LoS_DIM = np.int32(LoS_DIM)
    wavelength_range = np.asarray(wavelength_range, dtype=np.float32)
    voigt_tables = np.asarray(voigt_tables, dtype=np.float32)
    rel_idcs_list = np.asarray(rel_idcs_list, dtype=np.int32)
    z_LoS_highres_list = np.asarray(z_LoS_highres_list, dtype=np.float32)
    z_table_list = np.asarray(z_table_list, dtype=np.float32)
    dz_list = np.asarray(dz_list, dtype=np.float32)
    prefactor = np.asarray(prefactor, dtype=np.float32)
    n_HI = np.asarray(n_HI, dtype=np.float32)

    main_loop(xHI, density, z_LoS, DIM, LoS_DIM, voigt_tables, \
        rel_idcs_list, z_LoS_highres_list, z_table_list, dz_list,\
        prefactor, n_HI)
