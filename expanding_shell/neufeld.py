"""
Understanding the Neufeld relationship between dv and nhi
"""

import os

import numpy as np
import py21cmfast as p21c

from tqdm import tqdm

from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, k_B, m_p, e, m_e

from scipy.special import gamma, erf
from scipy.optimize import curve_fit, differential_evolution

import matplotlib.pyplot as plt
rc = {"font.family" : "serif", 
    "mathtext.fontset" : "stix"}
plt.rcParams.update(rc) 
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({'font.size': 14})
import matplotlib as mpl
label_size = 20
font_size = 30
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size
def double_power(x, a, b1, b2, c):
    """
    Double power law function.
    """
    x = x - c
    return a + x + np.log10(10**(x*b1) + 10**(x*b2))

def sigmoid(x, L, x0, k, b):
    """
    Sigmoid function.
    """
    return L / (1 + np.exp(-k*(x - x0))) + b

def lorentzian(x, x0, gamma, a, b):
    """
    Lorentzian function.
    """
    return a * (gamma**2) / ((x - x0)**2 + gamma**2) + b

def mh_from_muv(muv):
    """
    Get log10 halo mass from UV magnitude using the fitted skew normal parameters.
    """
    # https://en.wikipedia.org/wiki/Skew_normal_distribution
    # https://stats.stackexchange.com/questions/316314/sampling-from-skew-normal-distribution
    popt_mu = [11.50962787, -1.25471915, -2.12854869, -21.99916644]
    popt_std = [-0.50714459, -20.92567604, 1.72699987, 0.72541845]
    popt_alpha = [-2.13037242e+01, 1.83486155e+00, 2.49700612e+00, 8.04770033e-03]
    mu_val = double_power(muv, *popt_mu)
    std_val = sigmoid(muv, *popt_std)
    alpha_val = lorentzian(muv, *popt_alpha)
    standard_normal_samples = np.random.normal(0, 1, size=len(muv))
    p_flip = 0.5 * (1 + erf(-1*alpha_val*standard_normal_samples / np.sqrt(2)))
    u_samples = np.random.uniform(0, 1, size=len(muv))
    standard_normal_samples[u_samples < p_flip] *= -1
    standard_normal_samples *= std_val
    mh_samples = standard_normal_samples + mu_val
    return mh_samples

def median_nhi(mh):
    """
    Mean NHI as a function of halo mass from Gelli et al. 2025.
    """
    return 10**21.73 * (mh / 1e11)**0.38  # cm^-2

def sigma_nhi(mh):
    """
    Standard deviation of NHI as a function of halo mass from Gelli et al. 2025.
    """
    return 0.29 * np.log10(mh) - 2.18  # dex

def nhi_mh(mh):
    """
    NHI distribution as a function of halo mass.
    """
    med = median_nhi(mh)
    sig = sigma_nhi(mh)
    standard_sample = np.random.normal(0, 1, size=len(mh))
    delta = sig * standard_sample #* np.log10(np.e)
    nhi_samples = med * 10**delta
    return nhi_samples

def nhi_from_muv(muv):
    r = np.random.normal(loc=0.0, scale=5.29, size=len(muv))
    nhi = -4.9e17 * (muv + 11.76 + r)**3
    return nhi

def sgh_mean_nhi_dv(dv):
    return 20.87 + dv/263

def sgh_sigma_nhi_dv(dv):
    return 0.5 + dv/345

def neufeld_nhi_dv(dv):
    return 3*np.log10(dv/160) + 20.0

# NOTE uncertainties are sus
# dv_space = np.linspace(0, 1000, 1000)
# sgh_mean_nhi = sgh_mean_nhi_dv(dv_space)
# sgh_sigma_nhi = sgh_sigma_nhi_dv(dv_space)
# neufeld_nhi = neufeld_nhi_dv(dv_space)

# fig, axs = plt.subplots(1,1, figsize=(8,6), constrained_layout=True)
# axs.plot(dv_space, neufeld_nhi, label='Neufeld (1990)', color='black')
# axs.plot(dv_space, sgh_mean_nhi, label='This Work', color='red')
# axs.fill_between(dv_space, sgh_mean_nhi - sgh_sigma_nhi, sgh_mean_nhi + sgh_sigma_nhi, \
#                  color='red', alpha=0.3)
# axs.set_xlabel(r'$\Delta v$ [km/s]', fontsize=font_size)
# axs.set_ylabel(r'$\log_{10}(N_{\rm HI} \, [{\rm cm}^{-2}])$', fontsize=font_size)
# axs.legend(fontsize=font_size-4)
# axs.set_xlim(0, 1000)
# plt.show()
# quit()

muv_sample = np.random.uniform(-22, -16, size=1000000)
nhi_neufeld = nhi_from_muv(muv_sample)

mh_sample = 10**mh_from_muv(muv_sample)
nhi_gelli = nhi_mh(mh_sample)

nhi_side = np.logspace(18, 24, 50)
muv_side = np.linspace(-22, -16, 50)

fig, axs = plt.subplots(1,2, figsize=(12,6), sharey=True, sharex=True, constrained_layout=True)

im = axs[0].hist2d(muv_sample[nhi_neufeld>1e17], np.log10(nhi_neufeld[nhi_neufeld>1e17]), \
           bins=(muv_side, np.log10(nhi_side)), vmin=0, vmax=0.12, cmap='hot_r', density=True)
axs[0].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
axs[0].set_ylabel(r'$\log_{10}(N_{\rm HI} \, [{\rm cm}^{-2}])$', fontsize=font_size)
axs[0].set_title('Neufeld (1990)', fontsize=font_size)

im = axs[1].hist2d(muv_sample, np.log10(nhi_gelli), \
           bins=(muv_side, np.log10(nhi_side)), vmin=0, vmax=0.12, cmap='hot_r', density=True)
cb = fig.colorbar(im[3], ax=axs[1])
cb.set_label('Number density [arb. units]', fontsize=font_size-4)
axs[1].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
axs[1].set_title('Gelli et al. (2025)', fontsize=font_size)
# plt.show()
plt.savefig('/mnt/c/Users/sgagn/OneDrive/Documents/phd/lyman_alpha/figures/neufeld_nhi_muv.pdf')