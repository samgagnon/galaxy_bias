import os

import numpy as np
import py21cmfast as p21c

from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, k_B, m_p, e, m_e

from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from scipy.special import gamma, erf
from scipy.optimize import minimize, curve_fit

from sklearn.decomposition import PCA

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

lum_flux_factor = 4*np.pi*(Planck18.luminosity_distance(5.0).to('cm').value)**2

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

def line(x, m, b):
    """
    Linear function.
    """
    return m * x + b

def flat(x, m):
    """
    Linear function.
    """
    return np.ones_like(x)*m

def parabola(x, a, b, c):
    """
    Parabolic function.
    """
    return a * x**2 + b * x + c

def sigmoid(x, a, b, c, d):
    """
    Sigmoid function.
    """
    return d / (1 + np.exp(-a * (x - b))) + c

muv = np.load('../data/pca/muv.npy')
params = np.load('../data/pca/params.npy')
n_points = np.load('../data/pca/n_points.npy')
f = np.load('../data/pca/f.npy')
f_err = np.load('../data/pca/f_err.npy')

std_fixed_mean = np.load('../data/pca_fixed_mean/params.npy')
f_fixed_mean = np.load('../data/pca_fixed_mean/f.npy')
f_fixed_mean_err = np.load('../data/pca_fixed_mean/f_err.npy')

XC = np.load('../data/pca/xc.npy')
XSTD = np.load('../data/pca/xstd.npy')

A = np.linalg.inv(np.load('../data/pca/A.npy'))

NSAMPLES = 1000

fmw = []
fmd = []
log10lya = []
log10lha = []
lye = []
lae = []
dv = []
dve = []
for i, _muv in enumerate(muv):

    mu1, mu2, mu3, std1, std2, std3 = params[i]

    y1 = np.random.normal(mu1, std1, NSAMPLES)
    y2 = np.random.normal(mu2, std2, NSAMPLES)
    y3 = np.random.normal(mu3, std3, NSAMPLES)

    y = np.array([y1, y2, y3])
    x = (np.linalg.inv(A) @ y) * XSTD + XC

    lly, _dv, lha = x
    _p_obs_wide = p_obs(10**lly, 10**lha, _muv, mode='wide')
    _p_obs_deep = p_obs(10**lly, 10**lha, _muv, mode='deep')
    f_wide = np.sum(_p_obs_wide) / NSAMPLES
    f_deep = np.sum(_p_obs_deep) / NSAMPLES
    fmw.append(f_wide)
    fmd.append(f_deep)
    log10lya.append(np.mean(lly))
    log10lha.append(np.mean(lha))
    lye.append(np.std(lly))
    lae.append(np.std(lha))
    dv.append(np.mean(_dv))
    dve.append(np.std(_dv))

fmw = np.array(fmw)
fmd = np.array(fmd)
log10lya = np.array(log10lya)
log10lha = np.array(log10lha)
lye = np.array(lye)
lae = np.array(lae)
dv = np.array(dv)
dve = np.array(dve)

mu1, mu2, mu3, std1, std2, std3 = params.T

std_f1, std_f2, std_f3 = std_fixed_mean.T

popt1, pcov1 = curve_fit(parabola, muv, mu1, p0=[1, 1, 1])
popt2, pcov2 = curve_fit(parabola, muv, mu2, p0=[1, 1, 1])
popt3, pcov3 = curve_fit(parabola, muv, mu3, p0=[1, 1, 1])

print(f'Fit parameters for mu1: {popt1}, std: {np.sqrt(np.diag(pcov1))}')
print(f'Fit parameters for mu2: {popt2}, std: {np.sqrt(np.diag(pcov2))}')
print(f'Fit parameters for mu3: {popt3}, std: {np.sqrt(np.diag(pcov3))}')

fw = []
fd = []
for _muv in muv:
    y1 = np.random.normal(parabola(_muv, *popt1), 1, NSAMPLES)
    y2 = np.random.normal(parabola(_muv, *popt2), 1, NSAMPLES)
    y3 = np.random.normal(parabola(_muv, *popt3), 1, NSAMPLES)
    Y = np.stack([y1, y2, y3])
    X = (np.linalg.inv(A) @ Y) * XSTD + XC
    X = (np.linalg.inv(A) @ Y) * XSTD + XC
    lly, dv, lha = X
    _p_obs_wide = p_obs(10**lly, 10**lha, _muv, mode='wide')
    _p_obs_deep = p_obs(10**lly, 10**lha, _muv, mode='deep')

    f_wide = np.sum(_p_obs_wide) / NSAMPLES
    f_deep = np.sum(_p_obs_deep) / NSAMPLES
    fw.append(f_wide)
    fd.append(f_deep)

fw = np.array(fw)
fd = np.array(fd)

fig, axs = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)

axs[0].plot(muv, mu1, '-', color='red', label=r'$\mu_1$')
axs[0].plot(muv, mu2, '-', color='orange', label=r'$\mu_2$')
axs[0].plot(muv, mu3, '-', color='yellow', label=r'$\mu_3$')
axs[0].plot(muv, parabola(muv, *popt1), '--', color='red')
axs[0].plot(muv, parabola(muv, *popt2), '--', color='orange')
axs[0].plot(muv, parabola(muv, *popt3), '--', color='yellow')
axs[0].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
axs[0].legend(fontsize=font_size)


axs[1].errorbar(muv, f.T[0], yerr=f_err.T[0], fmt='o', color='red', markersize=5, label='Wide')
axs[1].errorbar(muv, f.T[1], yerr=f_err.T[1], fmt='o', color='orange', markersize=5, label='Deep')
# axs[2].errorbar(muv, f_fixed_mean.T[0], yerr=f_fixed_mean_err.T[0], fmt='.', color='red', markersize=5)
# axs[2].errorbar(muv, f_fixed_mean.T[1], yerr=f_fixed_mean_err.T[1], fmt='.', color='orange', markersize=5)
axs[1].plot(muv, fmw, '-', color='red')
axs[1].plot(muv, fmd, '-', color='orange')
axs[1].plot(muv, fw, '--', color='red')
axs[1].plot(muv, fd, '--', color='orange')
axs[1].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
axs[1].set_ylabel(r'$\log_{10}f_{\rm obs}$', fontsize=font_size)
axs[1].legend(fontsize=font_size)
axs[1].set_yscale('log')

plt.show()
