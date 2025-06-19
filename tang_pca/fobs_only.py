"""
Uses the PCA basis and treats the evolution of mu, sigma 
for each basis vector as a linear function of Muv. Optimizes 
the parameters of each linear relation by fitting the observed 
fraction of LyA+Ha emitters quantity.
"""

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

def schechter(muv, phi, muv_star, alpha):
    return (0.4*np.log(10))*phi*(10**(0.4*(muv_star - muv)))**(alpha+1)*\
        np.exp(-10**(0.4*(muv_star - muv)))

def p_muv(muv, phi, muv_star, alpha):
    return schechter(muv, phi, muv_star, alpha)/gamma(alpha_5+2)/(0.4*np.log(10))/phi_5

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
_lum_lya_wide = lum_lya[wide]
_lum_lya_err_wide = lum_lya_err[wide]
_lum_ha_wide = lum_ha[wide]
_lum_ha_err_wide = lum_ha_err[wide]

deep = ID==1
_muv_deep = MUV[deep]
_muv_err_deep = MUV_err[deep]
_dv_lya_deep = dv_lya[deep]
_dv_lya_err_deep = dv_lya_err[deep]
_lum_lya_deep = lum_lya[deep]
_lum_lya_err_deep = lum_lya_err[deep]
_lum_ha_deep = lum_ha[deep]
_lum_ha_err_deep = lum_ha_err[deep]

muv_wide = np.zeros(len(_muv_wide)*1000)
lum_lya_wide = np.zeros_like(muv_wide)
dv_wide = np.zeros_like(muv_wide)
lum_ha_wide = np.zeros_like(muv_wide)

muv_deep = np.zeros(len(_muv_deep)*1000)
lum_lya_deep = np.zeros_like(muv_deep)
dv_deep = np.zeros_like(muv_deep)
lum_ha_deep = np.zeros_like(muv_deep)

for i, (muv, muve, lly, llye, dv, dve, lha, lhae) in enumerate(zip(_muv_wide, _muv_err_wide, _lum_lya_wide, _lum_lya_err_wide, \
                                                 _dv_lya_wide, _dv_lya_err_wide, _lum_ha_wide, _lum_ha_err_wide)):
    muv_wide[1000*i:1000*(i+1)] = np.random.normal(muv, muve, 1000)
    lum_lya_wide[1000*i:1000*(i+1)] = np.random.normal(lly, llye, 1000)
    dv_wide[1000*i:1000*(i+1)] = np.random.normal(dv, dve, 1000)
    lum_ha_wide[1000*i:1000*(i+1)] = np.random.normal(lha, lhae, 1000)
    lum_ha_wide[1000*i:1000*(i+1)][lum_ha_wide[1000*i:1000*(i+1)]<10] = np.mean(lum_ha_wide[1000*i:1000*(i+1)])  # replace with mean
    lum_lya_wide[1000*i:1000*(i+1)][lum_lya_wide[1000*i:1000*(i+1)]<10] = np.mean(lum_lya_wide[1000*i:1000*(i+1)])  # replace with mean

    
for i, (muv, muve, lly, llye, dv, dve, lha, lhae) in enumerate(zip(_muv_deep, _muv_err_deep, _lum_lya_deep, _lum_lya_err_deep, \
                                                 _dv_lya_deep, _dv_lya_err_deep, _lum_ha_deep, _lum_ha_err_deep)):
    muv_deep[1000*i:1000*(i+1)] = np.random.normal(muv, muve, 1000)
    lum_lya_deep[1000*i:1000*(i+1)] = np.random.normal(lly, llye, 1000)
    dv_deep[1000*i:1000*(i+1)] = np.random.normal(dv, dve, 1000)
    lum_ha_deep[1000*i:1000*(i+1)] = np.random.normal(lha, lhae, 1000)
    lum_ha_deep[1000*i:1000*(i+1)][lum_ha_deep[1000*i:1000*(i+1)]<10] = np.mean(lum_ha_deep[1000*i:1000*(i+1)])  # replace with mean
    lum_lya_deep[1000*i:1000*(i+1)][lum_lya_deep[1000*i:1000*(i+1)]<10] = np.mean(lum_lya_deep[1000*i:1000*(i+1)])  # replace with mean

def normal_cdf(x, mu=0):
    """
    Cumulative distribution function for a normal distribution.
    """
    return 0.5 * (1 + erf((x - mu + mu/5) / (mu/5 * np.sqrt(2))))

def p_obs(lly, lha, muv, mode='wide'):
    """
    Probability of observing a galaxy with given Lya luminosity, H-alpha luminosity, and UV magnitude.
    """
    # Convert luminosities to fluxes
    f_lya = lly / lum_flux_factor
    f_ha = lha / lum_flux_factor
    luv = 10**(0.4*(51.64 - muv))
    w_emerg = (1215.67/2.47e15)*(lly/luv)
    f_ha_lim = 2e-18  # H-alpha flux limit in erg/s/cm^2
    if mode == 'wide':
        w_lim = 80
        f_lya_lim = 2e-17
    elif mode == 'deep':
        w_lim = 8
        f_lya_lim = 2e-18
    # https://arxiv.org/pdf/2202.06642
    # https://arxiv.org/pdf/2003.12083
    muv_lim = -17.75

    p_lya = normal_cdf(f_lya, f_lya_lim)
    p_ha = normal_cdf(f_ha, f_ha_lim)
    p_w = normal_cdf(w_emerg, w_lim)
    p_muv = 1 - normal_cdf(10**muv, 6*(10**muv_lim))
    
    return p_lya * p_ha * p_w * p_muv

def p_f(_p_obs, f, fe):
    _f = np.sum(_p_obs) / len(_p_obs)
    _f_err = _f * np.sqrt((np.sqrt(len(_p_obs)) / len(_p_obs))**2 + (fe/f)**2)
    dist = np.exp(-0.5 * ((f - _f) / _f_err) ** 2)
    if np.isinf(dist):
        dist = 100
    # get Gaussian distance from expected observed fraction
    return dist

def line(x, m, b):
    """
    Linear function.
    """
    return m * x + b

XWIDE = np.array([np.log10(lum_lya_wide), dv_wide, np.log10(lum_ha_wide)])
XDEEP = np.array([np.log10(lum_lya_deep), dv_deep, np.log10(lum_ha_deep)])

# Y = PCA T of X
os.makedirs('../data/pca', exist_ok=True)

# pca = PCA(n_components=3)  # Keeping all 3 dimensions

# transform the wide sample to the PCA basis
XALL = np.concatenate((XWIDE, XDEEP), axis=1)

XC = XALL.mean(axis=1, keepdims=True)
np.save('../data/pca/xc.npy', XC)
XSTD = XALL.std(axis=1, keepdims=True)
np.save('../data/pca/xstd.npy', XSTD)
XALL0 = (XALL - XC) / XSTD

T = np.linalg.inv(np.load('../data/pca/A.npy'))

YALL = T @ XALL0

XWIDE0 = XWIDE - XC
XDEEP0 = XDEEP - XC

def fit():
    """
    Fit the PCA coefficients to the observed fraction of LyA+Ha emitters.
    """

    NBINS = 20

    muv_centers = np.linspace(-20, -17, NBINS)
    f = np.zeros((NBINS, 2))
    f_err = np.zeros((NBINS, 2))
    for i, muv in enumerate(muv_centers):

        muv_range = np.linspace(muv-0.5, muv+0.5, 1000)
        GALDENS = trapezoid(schechter(muv_range, phi_5, muv_star_5, alpha_5), muv_range)

        VOL_WIDE = 146.082  # 10^3 Mpc^3
        n_gal = GALDENS * VOL_WIDE
        n_gal_rel_err = np.sqrt(n_gal)/n_gal
        n_muv_wide = np.sum(np.abs(muv_wide - muv) < 0.5)/1000
        n_muv_wide_rel_err = np.sqrt(n_muv_wide)/n_muv_wide

        f[i,0] = n_muv_wide / n_gal
        f_err[i,0] = f[i,0] * np.sqrt(n_gal_rel_err**2 + n_muv_wide_rel_err**2)
        
        VOL_DEEP = 7.877 # 10^3 Mpc^3
        n_gal = GALDENS * VOL_DEEP
        n_gal_rel_err = np.sqrt(n_gal)/n_gal
        n_muv_deep = np.sum(np.abs(muv_deep - muv) < 0.5)/1000
        n_muv_deep_rel_err = np.sqrt(n_muv_deep)/n_muv_deep

        f[i,1] = n_muv_deep / n_gal
        f_err[i,1] = f[i,1] * np.sqrt(n_gal_rel_err**2 + n_muv_deep_rel_err**2)

    def objective(params):
        m1, m2, m3, b1, b2, b3 = params

        # TODO let's write in the linear function from analyse_pca.py for verification

        # Fit the PCA coefficients to the observed fraction of LyA+Ha emitters
        mu1 = line(muv_centers, m1, b1)
        mu2 = line(muv_centers, m2, b2)
        mu3 = line(muv_centers, m3, b3)

        std = np.ones(NBINS)  # Assuming constant std for simplicity

        y1 = np.random.normal(mu1, std, (1000, NBINS))
        y2 = np.random.normal(mu2, std, (1000, NBINS))
        y3 = np.random.normal(mu3, std, (1000, NBINS))

        Y = np.stack([y1, y2, y3]).reshape((1000, 3, NBINS))
        X = (np.linalg.inv(T) @ Y) * XSTD.reshape((1, 3, 1)) + XC.reshape((1, 3, 1))
        X = X.reshape((3, 1000, NBINS))
        lly, _, lha = X

        plt.scatter(lly, lha)
        plt.show()

        _p_obs_wide = p_obs(lly, lha, muv_centers.reshape((1, 20)), mode='wide')
        _p_obs_deep = p_obs(lly, lha, muv_centers.reshape((1, 20)), mode='deep')

        f_wide = np.sum(_p_obs_wide, axis=0) / 1000
        f_deep = np.sum(_p_obs_deep, axis=0) / 1000

        # plt.scatter(muv_centers, f_wide, s=1, label='Wide Sample', color='red', alpha=0.5)
        # plt.scatter(muv_centers, f_deep, s=1, label='Deep Sample', color='orange', alpha=0.5)
        # plt.errorbar(muv_centers, f[:,0], yerr=f_err[:,0], fmt='o', color='red', markersize=5, label='Wide')
        # plt.errorbar(muv_centers, f[:,1], yerr=f_err[:,1], fmt='o', color='orange', markersize=5, label='Deep')
        # plt.yscale('log')
        # plt.show()

        f_wide_err = f_wide * np.sqrt((np.sqrt(1000) / 1000)**2 + (f_err[:,0]/f[:,0])**2)
        f_deep_err = f_deep * np.sqrt((np.sqrt(1000) / 1000)**2 + (f_err[:,1]/f[:,1])**2)

        p_wide = -0.5*((f_wide - f[i,0])/f_wide_err)**2
        p_deep = -0.5*((f_deep - f[i,1])/f_deep_err)**2

        logp = np.sum(p_wide) + np.sum(p_deep)

        return -1*logp

    bounds = [(-5, 5)] * 6  # bounds for each parameter
    result = differential_evolution(objective, bounds)
    return result.x

fit_params = fit()