"""
A parallelized code to calculate the rest-frame absorption spectrum
in each voxel of a light cone. 
"""

import numpy as np
from numba import njit, prange

# change decorator to include parallel=True
@njit(parallel=True)
def main_loop(xHI, density, z_LoS, DIM, LoS_DIM, wavelength_range,\
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
                rel_idcs = rel_idcs[rel_idcs != -1]
                
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
                for l in range(n_HI_rel.shape[0]):
                    n_HI_table[l] = n_HI_rel[l]

                z_table = z_table_list[k]
                
                integrand = crosssec_table * n_HI_table * z_table
                dz_table = dz_list[k]
                tau_IGM[i,j,k] = np.sum((integrand[:-1] + integrand[1:]) * dz_table / 2, axis=0)
    return tau_IGM

if __name__ == "__main__":

    DIM = 10
    LoS_DIM = 100

    xHI = np.zeros((DIM, DIM, LoS_DIM))
    density = np.zeros_like(xHI)
    z_LoS = np.zeros(LoS_DIM)
    DIM = np.int32(DIM)
    LoS_DIM = np.int32(LoS_DIM)
    wavelength_range = np.zeros_like(z_LoS)
    voigt_tables = np.zeros((LoS_DIM, LoS_DIM))
    rel_idcs_list = np.zeros_like(voigt_tables)
    z_LoS_highres_list = np.zeros_like(z_LoS)
    z_table_list = np.zeros_like(voigt_tables)
    dz_list = np.zeros_like(z_table_list)
    prefactor = np.int32(10)
    n_HI = np.zeros_like(xHI)

    tau_IGM = main_loop(xHI, density, z_LoS, DIM, LoS_DIM, wavelength_range,\
            voigt_tables, rel_idcs_list, z_LoS_highres_list, z_table_list, dz_list,\
            prefactor, n_HI)
