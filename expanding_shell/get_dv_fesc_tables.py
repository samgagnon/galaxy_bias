"""
Comparing my LyA model to that of the expanding shell formulation employed by
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

def expanding_shell_waveform(x, v_max, n_hi, t4=1):
    sqrt_t4 = np.sqrt(t4)
    a = 4.702e-4 / sqrt_t4
    tau0 = 5.898e-14 * sqrt_t4 * n_hi
    a_tau0 = a * tau0
    beta = v_max / 12.85 / sqrt_t4
    j = x**2*np.exp(-1*np.sqrt(np.pi)*beta*x**3/3/(a_tau0))\
        /np.cosh(np.sqrt(np.pi**3/54)*x**3/(a_tau0))**2
    return j

# get velocity range
x_range = np.linspace(-100, 100, 5000)
v_range = -1*x_range * 12.85

# the waveform of the expanding shell model is only well-defined up to a maximum beta
beta_max = np.pi*np.sqrt(6)/3
v_max_max = beta_max * 12.85

v_max_range = np.linspace(-v_max_max/1.1, v_max_max/1.1, 100)
n_hi_range = np.logspace(20, 23, 100)
dv_table = np.zeros((len(v_max_range), len(n_hi_range)))
fesc_min_table = np.zeros((len(v_max_range), len(n_hi_range)))

for i, v_max in tqdm(enumerate(v_max_range)):
    for j, n_hi in enumerate(n_hi_range):
        j_expanding_shell = expanding_shell_waveform(x_range, v_max, n_hi)
        delta_v = v_range[v_range >= 0][np.argmax(j_expanding_shell[v_range >= 0])]
        dv_table[i, j] = delta_v

        blue_integral = np.trapz(j_expanding_shell[v_range < 0], v_range[v_range < 0])
        red_integral = np.trapz(j_expanding_shell[v_range >= 0], v_range[v_range >= 0])
        f_esc_min = red_integral / (red_integral + blue_integral)
        fesc_min_table[i, j] = f_esc_min

# make iso-dv contours
contour_levels = np.linspace(0, 1500, 11)
# make fesc contours
fesc_levels = np.linspace(0, 1, 11)

def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"

fig, axs = plt.subplots(1, 2, figsize=(14,6), constrained_layout=True, sharey=True)
cs = axs[0].contour(n_hi_range, v_max_range, dv_table, levels=contour_levels, colors='white', linewidths=0.5)
im = axs[0].pcolormesh(n_hi_range, v_max_range, dv_table, shading='auto', cmap='plasma', vmin=0, vmax=1500)
axs[0].clabel(cs, cs.levels, fmt=fmt, fontsize=10)
cbar = fig.colorbar(im, ax=axs[0])
cbar.set_label(r'$\Delta v$ [km s$^{-1}$]', fontsize=font_size)
axs[0].set_xscale('log')
axs[0].set_xlabel(r'${\rm N}_{\rm HI}$ [cm$^{-2}$]', fontsize=font_size)
axs[0].set_ylabel(r'${\rm V}_{\rm max}$ [km s$^{-1}$]', fontsize=font_size)

cs = axs[1].contour(n_hi_range, v_max_range, fesc_min_table, levels=fesc_levels, colors='white', linewidths=0.5)
im = axs[1].pcolormesh(n_hi_range, v_max_range, fesc_min_table, shading='auto', cmap='plasma', vmin=0, vmax=1)
axs[1].clabel(cs, cs.levels, fmt=fmt, fontsize=10)
cbar = fig.colorbar(im, ax=axs[1])
cbar.set_label(r'${\rm f}_{\rm esc,\;max}^{\rm Ly\alpha}$', fontsize=font_size)
axs[1].set_xscale('log')
axs[1].set_xlabel(r'${\rm N}_{\rm HI}$ [cm$^{-2}$]', fontsize=font_size)
plt.show()

os.makedirs('./data/', exist_ok=True)
np.save('./data/dv_table.npy', dv_table)
np.save('./data/fesc_min_table.npy', fesc_min_table)
np.save('./data/v_max_range.npy', v_max_range)
np.save('./data/n_hi_range.npy', n_hi_range)
# viola has priors for NHI based on Mh https://arxiv.org/pdf/2510.01315