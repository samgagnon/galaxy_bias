"""
Generates and saves a PerturbHaloField object.

Samuel Gagnon-Hartman, 6 December 2024
Scuola Normale Superiore, Pisa, Italy
"""


import argparse
import numpy as np

import py21cmfast as p21c
from py21cmfast import cache_tools
from py21cmfast.c_21cmfast import ffi, lib

import logging, os, json
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('21cmFAST')

from timeit import default_timer as timer

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
    
    high_mass_turnover = ((mp1/mp2)**a_star)/((halo_masses/mp2)**(-1*a_star)+(halo_masses/mp2)**a_star2)
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

    import argparse

    parser = argparse.ArgumentParser(description='Generate a perturbed halo field')
    parser.add_argument('-m', type=float, default=1e8, help='Minimum halo mass')
    parser.add_argument('-z', type=float, default=6.6, help='Maximum halo mass')

    args = parser.parse_args()

    MIN_MASS = args.m
    redshift = args.z

    inputs = p21c.InputParameters.from_template('./sgh24.toml',\
                random_seed=1).evolve_input_structs(SAMPLER_MIN_MASS=MIN_MASS)

    initial_conditions = p21c.compute_initial_conditions(inputs=inputs)

    # how is this so fast????
    pth_field = p21c.determine_halo_list(
        redshift = redshift,
        inputs = inputs,
        initial_conditions = initial_conditions
    )

    os.makedirs(f'./data/halo_fields/z{np.around(redshift, 1)}', exist_ok=True)
    pth_field.write(direc=f'./data/halo_fields/z{np.around(redshift, 1)}/', fname=f'sgh24_MIN_MASS_{np.around(np.log10(MIN_MASS), 1)}.h5')