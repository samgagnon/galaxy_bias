import os

import numpy as np
import py21cmfast as p21c

from tqdm import tqdm

from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, k_B, m_p, e, m_e

from scipy.integrate import trapezoid
from scipy.special import gamma, erf
from scipy.optimize import differential_evolution

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

# from Bouwens 2021 https://arxiv.org/pdf/2102.07775
phi_5 = 0.79
muv_star_5 = -21.1
alpha_5 = -1.74

lum_flux_factor = 4*np.pi*(Planck18.luminosity_distance(5.0).to('cm').value)**2

def schechter(muv, phi, muv_star, alpha):
    return (0.4*np.log(10))*phi*(10**(0.4*(muv_star - muv)))**(alpha+1)*\
        np.exp(-10**(0.4*(muv_star - muv)))

def normal_cdf(x, mu=0):
    """
    Cumulative distribution function for a normal distribution.
    """
    return 0.5 * (1 + erf((x - mu + mu/5) / (mu/5 * np.sqrt(2))))

def mh(muv):
    """
    Returns log10 Mh in solar masses as a function of MUV.
    """
    redshift = 5.0
    muv_inflection = -20.0 - 0.26*redshift
    gamma = 0.4*(muv >= muv_inflection) - 0.7
    return gamma * (muv - muv_inflection) + 11.75

def vcirc(muv):
    """
    Returns circular velocity in km/s as a function of MUV 
    at redshift 5.0.
    """
    log10_mh = mh(muv)
    return (log10_mh - 5.62)/3

def p_obs(lly, dv, lha, muv, mode='wide'):
    """
    Probability of observing a galaxy with given Lya luminosity, H-alpha luminosity, and UV magnitude.
    """
    # Convert luminosities to fluxes
    f_lya = lly / lum_flux_factor
    f_ha = lha / lum_flux_factor
    luv = 10**(0.4*(51.64 - muv))
    w_emerg = (1215.67/2.47e15)*(lly/luv)
    f_ha_lim = 2e-18  # H-alpha flux limit in erg/s/cm^2
    v_lim = 10**vcirc(muv)
    if mode == 'wide':
        w_lim = 80
        f_lya_lim = 2e-17
    elif mode == 'deep':
        # tang has data down to 8 but it does not seem complete
        # to that level
        w_lim = 25
        f_lya_lim = 2e-18
    # https://arxiv.org/pdf/2202.06642
    # https://arxiv.org/pdf/2003.12083
    muv_lim = -17.75

    p_v = normal_cdf(dv, (6/5)*v_lim)
    p_lya = normal_cdf(f_lya, f_lya_lim)
    p_ha = normal_cdf(f_ha, f_ha_lim)
    p_w = normal_cdf(w_emerg, w_lim)
    p_muv = 1 - normal_cdf(10**muv, 6*(10**muv_lim))
    
    return p_lya * p_ha * p_w * p_muv * p_v

def line(x, m, b):
    """
    Linear function.
    """
    return m * (x + 18.5) + b

# measured lya properties from https://arxiv.org/pdf/2402.06070
MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, fescA, \
    fescA_err, fescB, fescB_err, ID = np.load('../data/tang24.npy').T

lum_lya = (ew_lya/1215.67) * 2.47e15 * 10**(0.4*(51.6 - MUV))
lum_lya_err = lum_lya*np.sqrt((ew_lya_err/ew_lya)**2 + (np.exp(-0.4*np.log(10)*MUV_err)/MUV)**2)

lum_ha = lum_lya / 11.4 / fescA
lum_ha_err = lum_ha*np.sqrt((lum_lya_err/lum_lya)**2 + (fescA_err/fescA)**2)

lum_flux_factor = 4*np.pi*(Planck18.luminosity_distance(5.0).to('cm').value)**2

wide = ID==0
_muv_wide = MUV[wide]
_muv_err_wide = MUV_err[wide]
_dv_lya_wide = dv_lya[wide]
_dv_lya_err_wide = dv_lya_err[wide]
_ew_lya_wide = ew_lya[wide]
_ew_lya_err_wide = ew_lya_err[wide]
_fesca_wide = fescA[wide]
_fesca_err_wide = fescA_err[wide]
_lum_lya_wide = lum_lya[wide]
_lum_lya_err_wide = lum_lya_err[wide]
_lum_ha_wide = lum_ha[wide]
_lum_ha_err_wide = lum_ha_err[wide]

deep = ID==1
_muv_deep = MUV[deep]
_muv_err_deep = MUV_err[deep]
_dv_lya_deep = dv_lya[deep]
_dv_lya_err_deep = dv_lya_err[deep]
_ew_lya_deep = ew_lya[deep]
_ew_lya_err_deep = ew_lya_err[deep]
_lum_lya_deep = lum_lya[deep]
_lum_lya_err_deep = lum_lya_err[deep]
_lum_ha_deep = lum_ha[deep]
_lum_ha_err_deep = lum_ha_err[deep]
_fesca_deep = fescA[deep]
_fesca_err_deep = fescA_err[deep]

import os

import numpy as np
import py21cmfast as p21c

from tqdm import tqdm

from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, k_B, m_p, e, m_e

from scipy.integrate import trapezoid
from scipy.special import gamma, erf
from scipy.optimize import differential_evolution

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

# from Bouwens 2021 https://arxiv.org/pdf/2102.07775
phi_5 = 0.79
muv_star_5 = -21.1
alpha_5 = -1.74

lum_flux_factor = 4*np.pi*(Planck18.luminosity_distance(5.0).to('cm').value)**2

def schechter(muv, phi, muv_star, alpha):
    return (0.4*np.log(10))*phi*(10**(0.4*(muv_star - muv)))**(alpha+1)*\
        np.exp(-10**(0.4*(muv_star - muv)))

def line(x, m, b):
    """
    Linear function.
    """
    return m * (x + 18.5) + b

fig, axs = plt.subplots(3, 3, figsize=(10, 15), constrained_layout=True)


axs[0,0].errorbar(_muv_wide, _lum_lya_wide, xerr=_muv_err_wide, \
                  yerr=_lum_lya_err_wide, fmt='o', color='cyan')
axs[0,0].errorbar(_muv_deep, _lum_lya_deep, xerr=_muv_err_deep, \
                 yerr=_lum_lya_err_deep, fmt='o', color='white')
axs[0,0].axhline(y=2e-18*lum_flux_factor, color='white', linestyle='--', label='Lya Luminosity Limit')
axs[0,0].axhline(y=1e-17*lum_flux_factor, color='cyan', linestyle='--', label='Lya Luminosity Limit')
axs[0,0].set_yscale('log')
axs[0,0].set_ylabel(r'$\log_{10}L_{\rm Ly\alpha}$ [erg/s]', fontsize=font_size)
axs[0,0].set_xticklabels([])

axs[0,1].errorbar(_dv_lya_wide, _lum_lya_wide, xerr=_dv_lya_err_wide, \
                  yerr=_lum_lya_err_wide, fmt='o', color='cyan')
axs[0,1].errorbar(_dv_lya_deep, _lum_lya_deep, xerr=_dv_lya_err_deep, \
                 yerr=_lum_lya_err_deep, fmt='o', color='white')
axs[0,1].axhline(y=2e-18*lum_flux_factor, color='white', linestyle='--', label='Lya Luminosity Limit')
axs[0,1].axhline(y=1e-17*lum_flux_factor, color='cyan', linestyle='--', label='Lya Luminosity Limit')
axs[0,1].set_yscale('log')
axs[0,1].set_yticklabels([])
axs[0,1].set_xticklabels([])

axs[0,2].errorbar(_lum_ha_wide, _lum_lya_wide, xerr=_lum_ha_err_wide, \
                  yerr=_lum_lya_err_wide, fmt='o', color='cyan')
axs[0,2].errorbar(_lum_ha_deep, _lum_lya_deep, xerr=_lum_ha_err_deep, \
                 yerr=_lum_lya_err_deep, fmt='o', color='white')
axs[0,2].axhline(y=2e-18*lum_flux_factor, color='white', linestyle='--', label='Lya Luminosity Limit')
axs[0,2].axhline(y=1e-17*lum_flux_factor, color='cyan', linestyle='--', label='Lya Luminosity Limit')
axs[0,2].axvline(x=1e-18*lum_flux_factor, color='white', linestyle='--', label='H-alpha Luminosity Limit')
axs[0,2].set_yscale('log')
axs[0,2].set_xscale('log')
axs[0,2].set_yticklabels([])
axs[0,2].set_xticklabels([])

axs[1,0].errorbar(_muv_wide, _dv_lya_wide, xerr=_muv_err_wide, \
                  yerr=_dv_lya_err_wide, fmt='o', color='cyan')
axs[1,0].errorbar(_muv_deep, _dv_lya_deep, xerr=_muv_err_deep, \
                 yerr=_dv_lya_err_deep, fmt='o', color='white')
axs[1,0].set_ylabel(r'$\Delta v$ [km/s]', fontsize=font_size)
axs[1,0].set_xticklabels([])

axs[1,1].errorbar(_dv_lya_wide, _dv_lya_wide, xerr=_dv_lya_err_wide, \
                  yerr=_dv_lya_err_wide, fmt='o', color='cyan')
axs[1,1].errorbar(_dv_lya_deep, _dv_lya_deep, xerr=_dv_lya_err_deep, \
                 yerr=_dv_lya_err_deep, fmt='o', color='white')
axs[1,1].set_yticklabels([])
axs[1,1].set_xticklabels([])

axs[1,2].errorbar(_lum_ha_wide, _dv_lya_wide, xerr=_lum_ha_err_wide, \
                  yerr=_dv_lya_err_wide, fmt='o', color='cyan')
axs[1,2].errorbar(_lum_ha_deep, _dv_lya_deep, xerr=_lum_ha_err_deep, \
                 yerr=_dv_lya_err_deep, fmt='o', color='white')
axs[1,2].axvline(x=1e-18*lum_flux_factor, color='white', linestyle='--', label='H-alpha Luminosity Limit')
axs[1,2].set_xscale('log')
axs[1,2].set_yticklabels([])
axs[1,2].set_xticklabels([])

axs[2,0].errorbar(_muv_wide, _lum_ha_wide, xerr=_muv_err_wide, \
                  yerr=_lum_ha_err_wide, fmt='o', color='cyan')
axs[2,0].errorbar(_muv_deep, _lum_ha_deep, xerr=_muv_err_deep, \
                 yerr=_lum_ha_err_deep, fmt='o', color='white')
axs[2,0].axhline(y=1e-18*lum_flux_factor, color='white', linestyle='--', label='H-alpha Luminosity Limit')
axs[2,0].set_yscale('log')
axs[2,0].set_ylabel(r'$\log_{10}L_{\rm H\alpha}$ [erg/s]', fontsize=font_size)
axs[2,0].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)

axs[2,1].errorbar(_dv_lya_wide, _lum_ha_wide, xerr=_dv_lya_err_wide, \
                  yerr=_lum_ha_err_wide, fmt='o', color='cyan')
axs[2,1].errorbar(_dv_lya_deep, _lum_ha_deep, xerr=_dv_lya_err_deep, \
                 yerr=_lum_ha_err_deep, fmt='o', color='white')
axs[2,1].axhline(y=1e-18*lum_flux_factor, color='white', linestyle='--', label='H-alpha Luminosity Limit')
axs[2,1].set_yscale('log')
axs[2,1].set_yticklabels([])
axs[2,1].set_xlabel(r'$\Delta v$ [km/s]', fontsize=font_size)

axs[2,2].errorbar(_lum_ha_wide, _lum_ha_wide, xerr=_lum_ha_err_wide, \
                  yerr=_lum_ha_err_wide, fmt='o', color='cyan')
axs[2,2].errorbar(_lum_ha_deep, _lum_ha_deep, xerr=_lum_ha_err_deep, \
                 yerr=_lum_ha_err_deep, fmt='o', color='white')
axs[2,2].axhline(y=1e-18*lum_flux_factor, color='white', linestyle='--', label='H-alpha Luminosity Limit')
axs[2,2].axvline(x=1e-18*lum_flux_factor, color='white', linestyle='--', label='H-alpha Luminosity Limit')
axs[2,2].set_yscale('log')
axs[2,2].set_xscale('log')
axs[2,2].set_yticklabels([])
axs[2,2].set_xlabel(r'$\log_{10}L_{\rm H\alpha}$ [erg/s]', fontsize=font_size)

plt.show()