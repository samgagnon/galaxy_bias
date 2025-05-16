"""
This script reads in the halo fields from the cache directory and extracts the relevant fields.

Samuel Gagnon-Hartman, 7 February 2025
"""

import os

import numpy as np
import py21cmfast as p21c

from tqdm import tqdm
from astropy.cosmology import Planck18

def get_stellar_mass(halo_masses, stellar_rng):
    sigma_star = 0.5
    mp1 = 1e10
    mp2 = 2.8e11
    M_turn = 10**(8.7)
    a_star = 0.5
    a_star2 = -0.61
    f_star10 = 0.05
    omega_b = Planck18.Ob0
    omega_m = Planck18.Om0
    baryon_frac = omega_b/omega_m
    
    high_mass_turnover = ((mp2/mp1)**a_star + (mp2/mp1)**a_star2)/((halo_masses/mp2)**(-1*a_star)+(halo_masses/mp2)**(-1*a_star2))
    stoc_adjustment_term = 0.5*sigma_star**2
    low_mass_turnover = np.exp(-1*M_turn/halo_masses + stellar_rng*sigma_star - stoc_adjustment_term)
    stellar_mass = f_star10 * baryon_frac * halo_masses * (high_mass_turnover * low_mass_turnover)
    return stellar_mass

def get_sfr(stellar_mass, sfr_rng, z):
    sigma_sfr_lim = 0.19
    sigma_sfr_idx = -0.12
    t_h = 1/Planck18.H(z).to('s**-1').value
    t_star = 0.5
    sfr_mean = stellar_mass / (t_star * t_h)
    sigma_sfr = sigma_sfr_idx * np.log10(stellar_mass/1e10) + sigma_sfr_lim
    sigma_sfr[sigma_sfr < sigma_sfr_lim] = sigma_sfr_lim
    stoc_adjustment_term = sigma_sfr * sigma_sfr / 2. # adjustment to the mean for lognormal scatter
    sfr_sample = sfr_mean * np.exp(sfr_rng*sigma_sfr - stoc_adjustment_term)
    return sfr_sample

if __name__ == "__main__":
    
    # TODO reconfigure, currently hardcoded

    cache_dir = './data/lc_cache/sgh25/'

    # create savedir for halo fields
    os.makedirs('./data/halo_fields/sgh25/', exist_ok=True)

    # call these in a separate loop
    # relevant_redshifts = [4.9, 5.7, 6.0, 6.6, 7.0, 7.3, 8.0
    
    min_redshift=5.0
    max_redshift=35.0

    node_redshifts = p21c.get_logspaced_redshifts(min_redshift=min_redshift,
                            max_redshift=max_redshift,
                            z_step_factor=1.02)
    
    inputs = p21c.InputParameters.from_template('./sgh25.toml',\
            random_seed=1,\
            node_redshifts=node_redshifts)
    
    lc = p21c.LightCone.from_file('./data/lightcones/sgh25.h5')
    los_z = lc.lightcone_redshifts

    cache_dir = './data/lc_cache/sgh25/'
    cache = p21c.OutputCache(cache_dir)
    runcache = p21c.RunCache.from_inputs(cache=cache,inputs=inputs)

    for i, _z in enumerate(tqdm(node_redshifts)):
        # read in halo field object
        pth_z = runcache.get_output_struct_at_z(
            kind="PerturbHaloField",
            z=_z,
            match_z_within=0.01,
        )

        x, y, z = pth_z.get('halo_coords').T

        if np.sum(x**2+y**2+z**2) == 0:
            continue
        # save the indices of all "inhabited" voxels

        DIM = pth_z.simulation_options.DIM
        HII_DIM = pth_z.simulation_options.HII_DIM
        BOX_LEN = pth_z.simulation_options.BOX_LEN

        redshift = pth_z.get('redshift')
        halo_masses = pth_z.get('halo_masses')
        sfr_rng = pth_z.get('sfr_rng')[halo_masses > 0]
        stellar_rng = pth_z.get('star_rng')[halo_masses > 0]

        x = x[halo_masses > 0]
        y = y[halo_masses > 0]
        z = z[halo_masses > 0]
        halo_masses = halo_masses[halo_masses > 0]
        
        stellar_masses = get_stellar_mass(halo_masses, stellar_rng)
        sfr = get_sfr(stellar_masses, sfr_rng, redshift)

        # coeval coords to LC coords
        z_adjust = np.argmin(np.abs(los_z - _z))
        z += z_adjust - int(HII_DIM/2)

        # save the halo field
        halo_field = np.zeros((6, len(x)), dtype=np.float32)
        halo_field[0] = x
        halo_field[1] = y
        halo_field[2] = z
        halo_field[3] = halo_masses
        halo_field[4] = stellar_masses
        halo_field[5] = sfr

        np.save(f'./data/halo_fields/sgh25/halo_field_{redshift}.npy', halo_field)
        # print(f"Saved halo field at redshift {np.around(redshift, 2)}")