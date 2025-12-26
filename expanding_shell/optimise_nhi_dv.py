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

muv_sample = np.random.uniform(-24, -16, 100000)
mh_sample = 10**mh_from_muv(muv_sample)
nhi_gelli = nhi_mh(mh_sample)

dv_sgh = np.random.normal(-27.92*(muv_sample + 18.5) + 197.19, 144.57, size=len(muv_sample))

dv_side = np.linspace(0, 1000, 50)
muv_side = np.linspace(-24, -16, 50)

sgh_hist, xedges, yedges = np.histogram2d(muv_sample, dv_sgh, bins=(muv_side, dv_side), density=True)
sh = sgh_hist.flatten()

def dv_nhi(nhi, a, b):
    return a * np.log10(nhi / b)
    # return a * (nhi/1e20)**b

def objective(params):
    a, b = params
    dv_gelli = dv_nhi(nhi_gelli, a, b)
    gelli_hist, _, _ = np.histogram2d(muv_sample, dv_gelli, bins=(muv_side, dv_side), density=True)
    gh = gelli_hist.flatten()
    kl_div = np.sum(np.log((sh + 1e-10) / (gh + 1e-10))*sh)
    return kl_div

result = differential_evolution(objective, bounds=[(0, 500), (1e18, 1e22)], maxiter=100)
a_opt, b_opt = result.x
print(f'Optimized parameters: a={a_opt}, b={b_opt}')

dv_gelli_opt = dv_nhi(nhi_gelli, a_opt, b_opt)
fig, axs = plt.subplots(1,2, figsize=(12,6), sharey=True, sharex=True, constrained_layout=True)
im = axs[0].hist2d(muv_sample, dv_sgh, \
           bins=(muv_side, dv_side), cmap='hot_r', density=True)
axs[0].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
axs[0].set_ylabel(r'$\Delta v$ [km/s]', fontsize=font_size)
axs[0].set_title('This Work', fontsize=font_size)

im = axs[1].hist2d(muv_sample, dv_gelli_opt, \
           bins=(muv_side, dv_side), cmap='hot_r', density=True)
cb = fig.colorbar(im[3], ax=axs[1])
cb.set_label('Number density [arb. units]', fontsize=font_size-4)
axs[1].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
axs[1].set_title('This Work + Gelli+25', fontsize=font_size)
plt.show()