"""
Drawing up an NHI Mh relation using Gelli's paper
https://ui.adsabs.harvard.edu/abs/2025arXiv251001315G/abstract
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

plt.style.use('dark_background')

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

# Example usage
muv_values = np.ones(10000) * -24
mh_values = 10**mh_from_muv(muv_values)
nhi_values = nhi_mh(mh_values)

print(np.median(nhi_values), np.std(np.log10(nhi_values)))
print("Expected median NHI:", median_nhi(np.mean(mh_values)))
print("Expected sigma NHI (dex):", sigma_nhi(np.mean(mh_values)))

plt.hist(np.log10(nhi_values), bins=30, alpha=0.7)
plt.show()