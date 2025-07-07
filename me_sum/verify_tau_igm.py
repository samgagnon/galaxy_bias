import h5py
import os

import numpy as np
from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, k_B, m_p

from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

import matplotlib.pyplot as plt
rc = {"font.family" : "serif", 
    "mathtext.fontset" : "stix"}
plt.rcParams.update(rc) 
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({'font.size': 14})
plt.style.use('dark_background')
import matplotlib as mpl
label_size = 20
font_size = 30
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size

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

def get_muv(sfr):
    """
    Converts star formation rate to
    absolute UV magnitude.
    """
    kappa = 1.15e-28
    luv = sfr * 3.1557e7 / kappa
    muv = 51.64 - np.log10(luv) / 0.4
    return muv

def I(x):
    return x**(9/2)/(1 - x) + (9/7)*x**(7/2) + (9/5)*x**(5/2) + \
        3*x**(3/2) + 9*x**(1/2) - np.log(1 + x**(1/2)) + np.log(1 - x**(1/2))

def delta_nu(T):
    """
    Returns the Doppler width of a Lya line with temperature T
    """
    return ((freq_Lya/c) * np.sqrt(2*k_B*T/m_p.to('kg'))).to('Hz')

def get_a(dnu):
    """
    Returns the damping parameter of a Lya line with Doppler width dnu
    """
    return (decay_factor/(4*np.pi*dnu)).to('')

hdf = h5py.File('../data/id_2001.h5', 'r')

z = 5.7
mh_min = 5e11 # Msol

density_lightcone = hdf['lightcones']['density'][:]
x_HI_lightcone = hdf['lightcones']['x_HI'][:]
lightcone_redshifts = dict(hdf['lightcones'].attrs)['lightcone_redshifts']
halo_data_redshifts = np.array(list(hdf['halo_data'].keys()), dtype=float)

z = halo_data_redshifts[np.argmin(halo_data_redshifts - z)]

masses = hdf['halo_data'][str(z)]['halo_masses'][:]
coords = hdf['halo_data'][str(z)]['halo_coords'][:]
sfr_rng_indv = hdf['halo_data'][str(z)]['sfr_rng'][:]
stellar_rng_indv = hdf['halo_data'][str(z)]['star_rng'][:]

coords = coords[masses > mh_min]
sfr_rng_indv = sfr_rng_indv[masses > mh_min]
stellar_rng_indv = stellar_rng_indv[masses > mh_min]
masses = masses[masses > mh_min]

los_dim = density_lightcone.shape[-1]

dc_los = Planck18.comoving_distance(5.0).to('Mpc') + \
    np.linspace(0, 1.5*los_dim, los_dim) * u.Mpc
z_los = np.array([z_at_value(Planck18.comoving_distance, d).value for d in dc_los])
central_index = np.argmin(np.abs(z_los - z))
coords[:,-1] = coords[:,-1] + central_index - 100
rel_idcs = np.arange(5) - 2 + central_index
lc_rel_idcs = np.arange(55) - 52 + central_index

_x = coords.T[0]
_y = coords.T[1]
_z = coords.T[2]

select = (_z>=rel_idcs[0])*(_z<=rel_idcs[-1])
masses = masses[select]
sfr_rng_indv = sfr_rng_indv[select]
stellar_rng_indv = stellar_rng_indv[select]
_x = _x[select]
_y = _y[select]
_z = _z[select]

sfr = get_sfr(get_stellar_mass(masses, stellar_rng_indv), sfr_rng_indv, z)
muv = get_muv(sfr)

density_rel = density_lightcone[..., lc_rel_idcs]
x_HI_rel = x_HI_lightcone[..., lc_rel_idcs]
local_range = np.arange(55)

# physical constants
wave_Lya = 1215.67*u.AA
freq_Lya = (c.to(u.AA/u.s)/wave_Lya).to('Hz')
omega_Lya = 2*np.pi*freq_Lya
c_kps = c.to('km/s').value
decay_factor = 6.25e8*u.s**-1
wavelength_range = np.linspace(1200, 1220, 1000)*u.AA
velocity_range = c_kps*(wavelength_range/wave_Lya - 1)

res_abs_list = []

for __x, __y, __z, sfr, stellar_rng, muv in zip(_x, _y, _z, sfr, stellar_rng_indv, muv):
    print(f'({__x}, {__y}, {__z}) SFR: {sfr*3.14e7:.2f} Msun/yr MUV: {muv:.2f}')
    tau = np.zeros(len(velocity_range))
    
    sightline_idcs = local_range[(lc_rel_idcs<=__z)]
    
    sightline_density = density_rel[__x,  __y, sightline_idcs] + 1
    sightline_x_HI = x_HI_rel[__x,  __y, sightline_idcs]
    
    sightline_z = z_los[sightline_idcs]
    z_diff = sightline_z[1] - sightline_z[0]
    sightline_z_edges = np.zeros(len(sightline_z) + 1)
    sightline_z_edges[:-1] = sightline_z - z_diff/2
    sightline_z_edges[-1] = sightline_z[-1]
    zs = sightline_z[-1]  # systemic redshift

    ra = (decay_factor/(4*np.pi*delta_nu(1e4*u.K))).to('').value
    prefactor = ra/np.pi
    
    z_rel = wavelength_range/wave_Lya - 1 + zs
    for zb, ze, _dens, _x_HI in zip(sightline_z_edges[1:], sightline_z_edges[:-1], \
                                    sightline_density, sightline_x_HI):
        tau +=  _dens * _x_HI * (1+zb)/(1+z_rel)**(3/2) * (I((1+zb)/(1+z_rel)) - I((1+ze)/(1+z)))
     
    tau *= prefactor

    if sightline_x_HI[-1] < 0.01:
        plt.plot(velocity_range, np.exp(-1*tau), color='cyan', alpha=0.5)
    else:
        plt.plot(velocity_range, np.exp(-1*tau), color='orange', alpha=0.5)

    res_abs_list.append(tau[0])

plt.xlabel(r'$\Delta v$ [km/s]', fontsize=font_size)
plt.ylabel(r'$\exp(-\tau_{\rm IGM})$', fontsize=font_size)
plt.show()

res_abs_list = np.array(res_abs_list)
np.save('/mnt/c/Users/sgagn/Downloads/res_tau_igm.npy', res_abs_list)