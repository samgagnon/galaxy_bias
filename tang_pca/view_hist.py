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
# presentation = False
presentation = True
if presentation:
    plt.style.use('dark_background')
    cmap = 'Blues_r'
    color1 = 'white'
    color2 = 'cyan'
else:
    cmap = 'hot_r'
    color1 = 'black'
    color2 = 'black'

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

def p_obs(lly, dv, lha, muv, theta, mode='wide'):
    """
    Probability of observing a galaxy with given Lya luminosity, H-alpha luminosity, and UV magnitude.
    """
    w1, w2, f1, f2, fh = theta
    # Convert luminosities to fluxes
    f_lya = lly / lum_flux_factor
    f_ha = lha / lum_flux_factor
    luv = 10**(0.4*(51.64 - muv))
    w_emerg = (1215.67/2.47e15)*(lly/luv)
    f_ha_lim = fh*2e-18  # H-alpha flux limit in erg/s/cm^2
    v_lim = 10**vcirc(muv)
    if mode == 'wide':
        w_lim = 80*w1
        f_lya_lim = f1*2e-17
    elif mode == 'deep':
        w_lim = 25*w2
        f_lya_lim = f2*2e-18
    # https://arxiv.org/pdf/2202.06642
    # https://arxiv.org/pdf/2003.12083
    # muv_lim = -18.0
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
_fesca_deep = fescA[deep]
_fesca_err_deep = fescA_err[deep]
_lum_lya_deep = lum_lya[deep]
_lum_lya_err_deep = lum_lya_err[deep]
_lum_ha_deep = lum_ha[deep]
_lum_ha_err_deep = lum_ha_err[deep]

# T = np.load('../data/pca/A.npy')
I = np.array([[1,0,0],[0,1,0],[0,0,1]])
A1 = np.array([[0,0,0],[0,0,-1],[0,1,0]])
A2 = np.array([[0,0,1],[0,0,0],[-1,0,0]])
A3 = np.array([[0,-1,0],[1,0,0],[0,0,0]])
c1, c2, c3, c4 = 1, 1, 1/3, -1
T = c1 * I + c2 * A1 + c3 * A2 + c4 * A3

NSAMPLES = 100000
xc = np.load('../data/pca/xc.npy')
xstd = np.load('../data/pca/xstd.npy')
f = np.load('../data/pca/f.npy')
f_err = np.load('../data/pca/f_err.npy')
m1, m2, m3, b1, b2, b3, std1, std2, std3, w1, w2, f1, f2, fh = np.load('../data/pca/fit_params.npy')
# print(m1, m2, m3, b1, b2, b3, std1, std2, std3)
# print(w1, w2, f1, f2, fh)
# print('xstd', xstd)
# print('xc', xc)
# quit()
theta = [w1, w2, f1, f2, fh]

muv_range = np.linspace(-22, -17, NSAMPLES)
p_muv = schechter(muv_range, phi_5, muv_star_5, alpha_5)
p_muv /= np.sum(p_muv)  # Normalize the probability distribution
muv_sample = np.random.choice(muv_range, size=NSAMPLES, p=p_muv)

# Fit the PCA coefficients to the observed fraction of LyA+Ha emitters
mu1 = line(muv_sample, m1, b1)
mu2 = line(muv_sample, m2, b2)
mu3 = line(muv_sample, m3, b3)

y1 = np.random.normal(mu1, std1, NSAMPLES)
y2 = np.random.normal(mu2, std2, NSAMPLES)
y3 = np.random.normal(mu3, std3, NSAMPLES)

# NOTE something is going wrong with the values here
Y = np.stack([y1, y2, y3], axis=-2)
X0 = T @ Y  # Transform to PCA basis

lly, dv, lha = X0[0], X0[1], X0[2]
lly = lly * xstd[0] + xc[0]
dv = dv * xstd[1] + xc[1]
lha = lha * xstd[2] + xc[2]

_p_obs_wide = p_obs(10**lly, dv, 10**lha, muv_sample, theta, mode='wide')
_p_obs_deep = p_obs(10**lly, dv, 10**lha, muv_sample, theta, mode='deep')

f_wide = np.sum(_p_obs_wide, axis=0) / NSAMPLES
f_deep = np.sum(_p_obs_deep, axis=0) / NSAMPLES

muv_deep = muv_sample[_p_obs_deep > 0.5]
lya_deep = lly[_p_obs_deep > 0.5]
dv_deep = dv[_p_obs_deep > 0.5]
lha_deep = lha[_p_obs_deep > 0.5]

muv_wide = muv_sample[_p_obs_wide > 0.5]
lya_wide = lly[_p_obs_wide > 0.5]
dv_wide = dv[_p_obs_wide > 0.5]
lha_wide = lha[_p_obs_wide > 0.5]

ew_sample = (10**lly)/(10**(0.4*(51.64 - muv_sample))) * 1215.67 / 2.47e15
fesc_sample = (10**lly)/(11.4*10**lha)

ew_deep = (10**lya_deep)/(10**(0.4*(51.64 - muv_deep))) * 1215.67 / 2.47e15
fesc_deep = (10**lya_deep)/(11.4*10**lha_deep)

ew_wide = (10**lya_wide)/(10**(0.4*(51.64 - muv_wide))) * 1215.67 / 2.47e15
fesc_wide = (10**lya_wide)/(11.4*10**lha_wide)

muv_space = np.linspace(-22, -17, 50)
ew_space = np.logspace(0, 3, 50)
dv_space = np.linspace(50, 1000, 50) # this is important!
fesc_space = np.linspace(0, 1.0, 50)
lha_space = np.logspace(41, 43, 50)

vcirc_space = vcirc(muv_space)

h11, _, _ = np.histogram2d(muv_deep, ew_deep, bins=[muv_space, ew_space], density=True)
h12, _, _ = np.histogram2d(muv_deep, dv_deep, bins=[muv_space, dv_space], density=True)
h130, _, _ = np.histogram2d(muv_deep, fesc_deep, bins=[muv_space, fesc_space], density=True)
h13, _, _ = np.histogram2d(muv_deep, 10**lha_deep, bins=[muv_space, lha_space], density=True)

h21, _, _ = np.histogram2d(muv_wide, ew_wide, bins=[muv_space, ew_space], density=True)
h22, _, _ = np.histogram2d(muv_wide, dv_wide, bins=[muv_space, dv_space], density=True)
h230, _, _ = np.histogram2d(muv_wide, fesc_wide, bins=[muv_space, fesc_space], density=True)
h23, _, _ = np.histogram2d(muv_wide, 10**lha_wide, bins=[muv_space, lha_space], density=True)

h31, _, _ = np.histogram2d(muv_sample, ew_sample, bins=[muv_space, ew_space], density=True)
h32, _, _ = np.histogram2d(muv_sample, dv, bins=[muv_space, dv_space], density=True)
h330, _, _ = np.histogram2d(muv_sample, fesc_sample, bins=[muv_space, fesc_space], density=True)
h33, _, _ = np.histogram2d(muv_sample, 10**lha, bins=[muv_space, lha_space], density=True)

fig, axs = plt.subplots(3, 3, figsize=(16, 12), sharex=False, constrained_layout=True)

axs[0,0].pcolormesh(muv_space, ew_space, h11.T, cmap=cmap, shading='auto')
axs[0,0].errorbar(_muv_deep, _ew_lya_deep, xerr=_muv_err_deep, yerr=_ew_lya_err_deep, fmt='o', color=color1, alpha=1.0, label='Observed')
axs[0,0].set_yscale('log')
axs[0,0].set_ylabel(r'${\rm W}_{\rm emerg}^{\rm Ly\alpha}$ [$\AA$]', fontsize=font_size)
axs[0,0].text(0.05, 0.95, r'MUSE-Deep', transform=axs[0,0].transAxes, fontsize=font_size, verticalalignment='top', color=color1)
axs[0,0].set_xlim(-22, -17)
axs[0,0].set_xticklabels([])

axs[1,0].pcolormesh(muv_space, ew_space, h21.T, cmap=cmap, shading='auto')
axs[1,0].errorbar(_muv_wide, _ew_lya_wide, xerr=_muv_err_wide, yerr=_ew_lya_err_wide, fmt='o', color=color2, alpha=1.0, label='Observed (EW)')
axs[1,0].set_yscale('log')
axs[1,0].set_ylabel(r'${\rm W}_{\rm emerg}^{\rm Ly\alpha}$ [$\AA$]', fontsize=font_size)
axs[1,0].text(0.05, 0.95, r'MUSE-Wide', transform=axs[1,0].transAxes, fontsize=font_size, verticalalignment='top', color=color1)
axs[1,0].set_xticklabels([])

def get_wc_mason(m):
    return 31 + 12 * np.tanh(4 * (m + 20.25))
axs[2,0].pcolormesh(muv_space, ew_space, h31.T, cmap=cmap, shading='auto')
# axs[2,0].plot(muv_space, get_wc_mason(muv_space), '--', color=color1, label='Wc Mason')
axs[2,0].set_yscale('log')
axs[2,0].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
axs[2,0].set_ylabel(r'${\rm W}_{\rm emerg}^{\rm Ly\alpha}$ [$\AA$]', fontsize=font_size)
axs[2,0].text(0.05, 0.95, r'no selection', transform=axs[2,0].transAxes, fontsize=font_size, verticalalignment='top', color=color1)

axs[0,1].pcolormesh(muv_space, dv_space, h12.T, cmap=cmap, shading='auto')
axs[0,1].errorbar(_muv_deep, _dv_lya_deep, xerr=_muv_err_deep, yerr=_dv_lya_err_deep, fmt='o', color=color1, alpha=1.0, label='Observed')
# axs[0,1].plot(muv_space, 10**vcirc_space, '--', color=color1, label='Vcirc')
axs[0,1].set_ylim(50, 600)
axs[0,1].set_ylabel(r'$\Delta v$ [km/s]', fontsize=font_size)
axs[0,1].set_xticklabels([])

axs[1,1].pcolormesh(muv_space, dv_space, h22.T, cmap=cmap, shading='auto')
axs[1,1].errorbar(_muv_wide, _dv_lya_wide, xerr=_muv_err_wide, yerr=_dv_lya_err_wide, fmt='o', color=color2, alpha=1.0, label='Observed (DV)')
# axs[1,1].plot(muv_space, 10**vcirc_space, '--', color=color1, label='Vcirc')
axs[1,1].set_ylim(50, 600)
axs[1,1].set_ylabel(r'$\Delta v$ [km/s]', fontsize=font_size)
axs[1,1].set_xticklabels([])

axs[2,1].pcolormesh(muv_space, dv_space, h32.T, cmap=cmap, shading='auto')
# axs[2,1].plot(muv_space, 10**vcirc_space, '--', color=color1, label='Vcirc')
axs[2,1].set_ylim(50, 600)
axs[2,1].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
axs[2,1].set_ylabel(r'$\Delta v$ [km/s]', fontsize=font_size)

axs[0,2].pcolormesh(muv_space, fesc_space, h130.T, cmap=cmap, shading='auto')
# axs[0,2].pcolormesh(muv_space, lha_space, h13.T, cmap=cmap, shading='auto')
axs[0,2].errorbar(_muv_deep, _fesca_deep, xerr=_muv_err_deep, yerr=_fesca_err_deep, fmt='o', color=color1, alpha=1.0, label='Observed')
# axs[0,2].errorbar(_muv_deep, _lum_ha_deep, xerr=_muv_err_deep, yerr=_lum_ha_err_deep, fmt='o', color=color1, alpha=1.0, label='Observed')
# axs[0,2].set_yscale('log')
axs[0,2].set_ylim(1e-3, 1)
axs[0,2].set_ylabel(r'$f_{\rm esc}^{\rm Ly\alpha}$', fontsize=font_size)
# axs[0,2].set_ylabel(r'$L_{\rm H\alpha}$ [erg/s]', fontsize=font_size)
axs[0,2].set_xticklabels([])

axs[1,2].pcolormesh(muv_space, fesc_space, h230.T, cmap=cmap, shading='auto')
# axs[1,2].pcolormesh(muv_space, lha_space, h23.T, cmap=cmap, shading='auto')
axs[1,2].errorbar(_muv_wide, _fesca_wide, xerr=_muv_err_wide, yerr=_fesca_err_wide, fmt='o', color=color2, alpha=1.0, label='Observed (fesc)')
# axs[1,2].errorbar(_muv_wide, _lum_ha_wide, xerr=_muv_err_wide, yerr=_lum_ha_err_wide, fmt='o', color=color2, alpha=1.0, label='Observed (fesc)')
# axs[1,2].set_yscale('log')
axs[1,2].set_ylim(1e-3, 1)
axs[1,2].set_ylabel(r'$f_{\rm esc}^{\rm Ly\alpha}$', fontsize=font_size)
# axs[1,2].set_ylabel(r'$L_{\rm H\alpha}$ [erg/s]', fontsize=font_size)
# axs[1,2].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
axs[1,2].set_xticklabels([])

axs[2,2].pcolormesh(muv_space, fesc_space, h330.T, cmap=cmap, shading='auto')
# axs[2,2].pcolormesh(muv_space, lha_space, h33.T, cmap=cmap, shading='auto')
# axs[2,2].set_yscale('log')
axs[2,2].set_ylim(1e-3, 1)
# axs[2,2].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
axs[2,2].set_ylabel(r'$f_{\rm esc}^{\rm Ly\alpha}$', fontsize=font_size)
axs[2,2].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
# axs[2,2].set_ylabel(r'$L_{\rm H\alpha}$ [erg/s]', fontsize=font_size)

axs[0,2].set_xlim(-22, -17)
axs[0,1].set_xlim(-22, -17)
axs[2,2].set_xlim(-22, -17)

axs[2,0].plot(-21.5, 13, color=color1, marker='*', markersize=20)
axs[2,1].plot(-21.5, 555, color=color1, marker='*', markersize=20)
axs[2,2].plot(-21.5, 0.04, color=color1, marker='*', markersize=20)

figures_dir = '/mnt/c/Users/sgagn/Documents/phd/lyman_alpha/figures/'
# plt.savefig(f'{figures_dir}/prop_hist.pdf', bbox_inches='tight')

plt.show()
