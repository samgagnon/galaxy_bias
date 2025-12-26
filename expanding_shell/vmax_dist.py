"""
Compute vmax as a function of MUV and Delta v
https://arxiv.org/pdf/2510.18946
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

def vmax_from_nhi_dv(n_hi, dv):
    """
    Get vmax from n_hi and delta v using the fitted logistic parameters.
    """
    # best-fit parameters from read_dv_fesc_tables.py
    a, b, c, d = 0.2281008595883267, 79.0806494487709, 0.009432694809561661, 4.216808122459913
    C, logA, k = 42.681206720549525, 4.609275018390131, 1.7191137540321961
    
    x0 = (a*dv + b) / (c*dv + d)
    x_log = np.log10(n_hi)
    S = np.exp(logA)
    # handle small k ~ 0: treat limit k->0 as S * (x_log - x0)
    if np.isclose(k, 0.0):
        return C - S * (x_log - x0)
    vmax = C - (S / k) * np.exp(k * (x_log - x0))
    return vmax

# n_hi_space = np.logspace(20, 25, 100)  # cm^-2
# dv = 1200 # km s^-1
# vmax_sample = vmax_from_nhi_dv(n_hi_space, dv)
# plt.plot(np.log10(n_hi_space), vmax_sample, '-', color='cyan')
# plt.xlabel(r'$\log_{10} N_{\mathrm{HI}}$ [cm$^{-2}$]', fontsize=font_size)
# plt.ylabel(r'$v_{\mathrm{max}}$ [km s$^{-1}$]', fontsize=font_size)
# plt.title(r'$\Delta v = $' + f' {dv} km s$^{{-1}}$', fontsize=font_size)
# plt.show()

# Example usage
NSAMPLE = 10000
# for muv in [-16, -18, -20, -22, -24]:
for mh in [1e10, 1e11, 1e12, 1e13]:
#     muv_values = np.ones(NSAMPLE) * muv
#     mh_values = 10**mh_from_muv(muv_values)
    mh_values = mh*np.ones(NSAMPLE)
    nhi_values = nhi_mh(mh_values)

    # I = np.array([[1,0,0],[0,1,0],[0,0,1]])
    # A1 = np.array([[0,0,0],[0,0,-1],[0,1,0]])
    # A2 = np.array([[0,0,1],[0,0,0],[-1,0,0]])
    # A3 = np.array([[0,-1,0],[1,0,0],[0,0,0]])
    # c1, c2, c3, c4 = 1, 1, 1/3, -1
    # A = c1 * I + c2 * A1 + c3 * A2 + c4 * A3
    # xc = np.load('../data/pca/xc.npy')
    # xstd = np.load('../data/pca/xstd.npy')

    # m1, m2, m3, b1, b2, b3, std1, std2, std3, w1, w2, f1, f2, fh = np.load('../data/pca/fit_params.npy')

    # u1, u2, u3 = np.random.normal(m1*(muv_values + 18.5) + b1, std1, NSAMPLE), \
    #             np.random.normal(m2*(muv_values + 18.5) + b2, std2, NSAMPLE), \
    #             np.random.normal(m3*(muv_values + 18.5) + b3, std3, NSAMPLE)
    # _, dv, _ = (A @ np.array([u1, u2, u3]))* xstd + xc

    vmax_values = vmax_from_nhi_dv(nhi_values, dv=1200)
    # I should sample dv from P(dv|muv) but for now just fix dv=1200 km/s

    select = (vmax_values > -600) & (vmax_values < 600)
    plt.hist(vmax_values[select], bins=30, alpha=0.7, label=f'Mh={mh:.1e}')
    print(np.median(vmax_values), np.std(vmax_values))

plt.legend()
plt.yscale('log')
plt.xlabel(r'$v_{\mathrm{max}}$ [km s$^{-1}$]', fontsize=font_size)
plt.ylabel('Counts', fontsize=font_size)
plt.show()