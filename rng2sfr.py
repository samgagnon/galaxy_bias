"""
Generates a single coeval cube and saves relevant cache files.

Samuel Gagnon-Hartman, 16 October 2024
Scuola Normale Superiore, Pisa, Italy
"""


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
    a_star2 = -0.6
    f_star10 = 0.05
    omega_b = Planck18.Ob0
    omega_m = Planck18.Om0
    baryon_frac = omega_b/omega_m
    
    high_mass_turnover = ((mp2/mp1)**a_star)/((halo_masses/mp2)**(-1*a_star)+(halo_masses/mp2)**(-1*a_star2))
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

def get_stellar_mass_no_stoch(halo_masses):
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
    
    high_mass_turnover = ((mp2/mp1)**a_star)/((halo_masses/mp2)**(-1*a_star)+(halo_masses/mp2)**(-1*a_star2))
    stoc_adjustment_term = 0.5*sigma_star**2
    low_mass_turnover = np.exp(-1*M_turn/halo_masses + 1*sigma_star - stoc_adjustment_term)
    stellar_mass = f_star10 * baryon_frac * halo_masses * (high_mass_turnover * low_mass_turnover)
    return stellar_mass

def get_sfr_no_stoch(stellar_mass, z):
    sigma_sfr_lim = 0.19
    sigma_sfr_idx = -0.12
    t_h = 1/Planck18.H(z).to('s**-1').value
    t_star = 0.5
    sfr_mean = stellar_mass / (t_star * t_h)
    sigma_sfr = sigma_sfr_idx * np.log10(stellar_mass/1e10) + sigma_sfr_lim
    sigma_sfr[sigma_sfr < sigma_sfr_lim] = sigma_sfr_lim
    stoc_adjustment_term = sigma_sfr * sigma_sfr / 2. # adjustment to the mean for lognormal scatter
    sfr_sample = sfr_mean * np.exp(sigma_sfr - stoc_adjustment_term)
    return sfr_sample