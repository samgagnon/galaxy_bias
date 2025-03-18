"""
This script reads in the halo fields from the cache directory and extracts the relevant fields.

Samuel Gagnon-Hartman, 7 February 2025
"""

import os

import numpy as np
import py21cmfast as p21c

from astropy.cosmology import Planck18

def plot(halo_masses, stellar_masses, sfr):
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc) 
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')

    alpha = 0.1

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    axs[0].scatter(halo_masses, stellar_masses, s=1, alpha=alpha, rasterized=True)
    axs[0].set_xlabel(r'$M_{\rm halo}$ [$M_{\odot}$]')
    axs[0].set_ylabel(r'$M_{\rm star}$ [$M_{\odot}$]')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlim(1e8, 1e9)

    axs[1].scatter(halo_masses, sfr, s=1, alpha=alpha, rasterized=True)
    axs[1].set_xlabel(r'$M_{\rm halo}$ [$M_{\odot}$]')
    axs[1].set_ylabel(r'SFR [$M_{\odot}$ yr$^{-1}$]')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlim(1e8, 1e9)

    axs[2].scatter(halo_masses, sfr/stellar_masses, s=1, alpha=alpha, rasterized=True)
    axs[2].set_xlabel(r'$M_{\rm halo}$ [$M_{\odot}$]')
    axs[2].set_ylabel(r'sSFR [yr$^{-1}$]')
    axs[2].set_xscale('log')
    axs[2].set_yscale('log')
    axs[2].set_xlim(1e8, 1e9)

    plt.show()
    quit()

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
    
    cache_dir = './data/lc_cache/sgh25/'

    # LC = p21c.LightCone.read('./data/lightcones/LC.h5')

    all_cache_files = os.listdir(cache_dir)

    # create savedir for halo fields
    os.makedirs('./data/halo_fields', exist_ok=True)

    halo_fields = []
    for file in all_cache_files:
        if file.startswith("HaloField"):
            halo_fields.append(file)

    relevant_redshifts = [4.9, 5.7, 6.0, 6.6, 7.0, 7.3, 8.0]
    cache_redshifts = []

    for halo_field_fn in halo_fields:
        halo_field = p21c.HaloField.from_file(fname=halo_field_fn, direc=cache_dir)
        redshift = halo_field.redshift
        cache_redshifts.append(redshift)

    relevant_redshifts = np.asarray(relevant_redshifts)
    cache_redshifts = np.asarray(cache_redshifts)

    rel_indices = [np.argmin(np.abs(rel_z - cache_redshifts)) for rel_z in relevant_redshifts]

    halo_fields = [halo_fields[i] for i in rel_indices]

    for halo_field_fn in halo_fields:
        halo_field = p21c.HaloField.from_file(fname=halo_field_fn, direc=cache_dir)
        if halo_field.random_seed == 1:
            x, y, z = halo_field.halo_coords.T

            if np.sum(x**2+y**2+z**2) != 0:
                # save the indices of all "inhabited" voxels

                DIM = halo_field.user_params.DIM
                HII_DIM = halo_field.user_params.HII_DIM
                BOX_LEN = halo_field.user_params.BOX_LEN

                # convert from voxel coords to comoving coords
                x, y, z = (x*HII_DIM/DIM).astype(np.int32), \
                    (y*HII_DIM/DIM).astype(np.int32), (z*HII_DIM/DIM).astype(np.int32)

                redshift = halo_field.redshift
                halo_masses = halo_field.halo_masses
                sfr_rng = halo_field.sfr_rng[halo_masses > 0]
                stellar_rng = halo_field.star_rng[halo_masses > 0]

                x = x[halo_masses > 0]
                y = y[halo_masses > 0]
                z = z[halo_masses > 0]
                halo_masses = halo_masses[halo_masses > 0]
                
                stellar_masses = get_stellar_mass(halo_masses, stellar_rng)
                sfr = get_sfr(stellar_masses, sfr_rng, redshift)

                # coeval coords to LC coords
                d_z = Planck18.comoving_distance(redshift).value - Planck18.comoving_distance(5.0).value
                d_z = (d_z*HII_DIM/BOX_LEN).astype(np.int32) - HII_DIM//2
                z += d_z

                # save the halo field
                halo_field = np.zeros((6, len(x)), dtype=np.float32)
                halo_field[0] = x
                halo_field[1] = y
                halo_field[2] = z
                halo_field[3] = halo_masses
                halo_field[4] = stellar_masses
                halo_field[5] = sfr

                np.save(f'./data/halo_fields/halo_field_{np.around(redshift,2)}.npy', halo_field)
                print(f"Saved halo field at redshift {np.around(redshift, 2)}")