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

n_hi_range = np.load('./data/n_hi_range.npy')  # cm^-2
v_max_range = np.load('./data/v_max_range.npy')  # km s^-
dv_table = np.load('./data/dv_table.npy')  # km s^-1
fesc_min_table = np.load('./data/fesc_min_table.npy')  # unitless

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
# cs = axs[0].contour(n_hi_range, v_max_range, dv_table, levels=contour_levels, colors='white', linewidths=0.5)
im = axs[0].pcolormesh(n_hi_range, v_max_range, dv_table, shading='auto', cmap='plasma', vmin=0, vmax=1500)
# axs[0].clabel(cs, cs.levels, fmt=fmt, fontsize=10)
cbar = fig.colorbar(im, ax=axs[0])
cbar.set_label(r'$\Delta v$ [km s$^{-1}$]', fontsize=font_size)
axs[0].set_xscale('log')
axs[0].set_xlabel(r'${\rm N}_{\rm HI}$ [cm$^{-2}$]', fontsize=font_size)
axs[0].set_ylabel(r'${\rm V}_{\rm max}$ [km s$^{-1}$]', fontsize=font_size)

# cs = axs[1].contour(n_hi_range, v_max_range, fesc_min_table, levels=fesc_levels, colors='white', linewidths=0.5)
im = axs[1].pcolormesh(n_hi_range, v_max_range, fesc_min_table, shading='auto', cmap='plasma', vmin=0, vmax=1)
# axs[1].clabel(cs, cs.levels, fmt=fmt, fontsize=10)
cbar = fig.colorbar(im, ax=axs[1])
cbar.set_label(r'${\rm f}_{\rm esc,\;max}^{\rm Ly\alpha}$', fontsize=font_size)
axs[1].set_xscale('log')
axs[1].set_xlabel(r'${\rm N}_{\rm HI}$ [cm$^{-2}$]', fontsize=font_size)
plt.savefig('/mnt/c/Users/sgagn/OneDrive/Documents/phd/dv_fesc_contours.pdf', dpi=300)
plt.close()

dv_levels = np.array([50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
vmax_dv_table = np.zeros((len(n_hi_range), len(dv_levels)))
for i, nhi in enumerate(n_hi_range):
    dv_vals = dv_table[:, i]
    vmax_dv = np.interp(dv_levels, dv_vals, v_max_range)
    vmax_dv_table[i, :] = vmax_dv

beta_max = np.pi*np.sqrt(6)/3
v_max_max = beta_max * 12.85

def smooth_schechter(x, a, b1, b2, c, x0):
    """
    Smooth Schechter function.
    """
    x = np.log10(x)
    l1 = 1 / (1 + np.exp((x - x0)))
    s1 = 1 / (1 + np.exp(-(x - x0)))
    first_term = (a * l1 + b1 * s1) * (x - x0)

    l2 = 1 / (1 + np.exp(-1*(x - x0)))
    s2 = 1 / (1 + np.exp((x - x0)))
    second_term = (c * l2 + b2 * s2) * np.exp(-1*(x - x0))

    return first_term * second_term


def monotone_schechter(x, C, logA, k, x0):
    """
    Monotonic decreasing model constructed so that
    d/d(log10 x) monotone_schechter < 0 for all x.

    Form:
        f(x) = C - (S / k) * exp(k * (log10(x) - x0)),
    where S = exp(logA) > 0 and k > 0, so
        df/d(log10 x) = -S * exp(k*(log10(x)-x0)) < 0.

    Parameters:
        x: array-like, input in linear space (will take log10 inside)
        C: baseline constant (same units as f)
        logA: log of positive amplitude (real)
        k: positive shape parameter (enforce >0 in bounds)
        x0: pivot in log10(x)

    Returns:
        array of same shape as x
    """
    x_log = np.log10(x)
    S = np.exp(logA)
    # handle small k ~ 0: treat limit k->0 as S * (x_log - x0)
    if np.isclose(k, 0.0):
        return C - S * (x_log - x0)
    return C - (S / k) * np.exp(k * (x_log - x0))

plt.figure(figsize=(8,6))

x0_list = []
dv_list = []

fit_bool = False

for j, dv_level in enumerate(dv_levels[::-1]):

    j = len(dv_levels) - j - 1  # reverse order for plotting

    vbool = (vmax_dv_table[:, j]>-v_max_max/1.1) * (vmax_dv_table[:, j]<v_max_max/1.1)
    plt.plot(n_hi_range[vbool], vmax_dv_table[:, j][vbool], label=f'Δv = {dv_level} km/s', \
             linestyle=':', color='white', alpha=0.7)

    if np.sum(vbool) == 0:
        continue

    elif not fit_bool:
        fit_bool = True

        # perform fit using a monotonic re-parameterization
        # params: C, logA, k, x0  such that df/d(log10 x) = -exp(logA) * exp(k*(log10(x)-x0)) < 0
        def objective_func(params):
            C, logA, k, x0 = params
            # enforce k>0 via bounds, so derivative is negative everywhere
            model_vals = monotone_schechter(n_hi_range[vbool], C, logA, k, x0)
            return np.sum((model_vals - vmax_dv_table[:, j][vbool])**2)

        popt = differential_evolution(objective_func, \
                                      bounds=[(0, 100), (-50, 50), (1e-3, 10), (21, 25)],
                                      maxiter=1000,
                                      popsize=20).x

        C_fit, logA_fit, k_fit, x0_fit = popt
        print(f"Fitted monotone parameters for Δv = {dv_level} km/s: C={C_fit}, logA={logA_fit}, k={k_fit}, x0={x0_fit}")
        dv_list.append(dv_level)
        x0_list.append(x0_fit)

        plt.plot(n_hi_range[vbool], monotone_schechter(n_hi_range[vbool], *popt), \
                 linestyle='-', color='cyan', alpha=0.7)
        
        # quite a rough fit, but lets keep it for now
        # plt.xscale('log')
        # plt.show()
        # quit()
        
        def fixed_monotone_schechter(x, x0):
            return monotone_schechter(x, C_fit, logA_fit, k_fit, x0)
            
            
    else:
        popt, pcov = curve_fit(fixed_monotone_schechter, n_hi_range[vbool] , vmax_dv_table[:, j][vbool], \
                                    p0=[20])
        x0 = popt[0]
        x0_list.append(x0)
        dv_list.append(dv_level)

        plt.plot(n_hi_range[vbool], fixed_monotone_schechter(n_hi_range[vbool], x0), \
                 linestyle='-', color='cyan', alpha=0.7)

plt.xscale('log')
plt.xlabel(r'${\rm N}_{\rm HI}$ [cm$^{-2}$]', fontsize=font_size)
plt.ylabel(r'${\rm V}_{\rm max}$ [km s$^{-1}$]', fontsize=font_size)
# plt.show()
plt.savefig('/mnt/c/Users/sgagn/OneDrive/Documents/phd/dv2.pdf', dpi=300)
plt.close()

dv_array = np.array(dv_list)
x0_array = np.array(x0_list)

# fit x0 vs dv_array with a rational function
def rational_func(x, a, b, c, d):
    """
    Rational function: (a*x + b) / (c*x + d)
    """
    return (a*x + b) / (c*x + d)

def objective_rational(params):
    a, b, c, d = params
    model_vals = rational_func(dv_array, a, b, c, d)
    return np.sum((model_vals - x0_array)**2)

popt_rational = differential_evolution(objective_rational, \
                                      bounds=[(-10, 10), (-100, 100), (1e-3, 10), (1e-3, 100)],
                                      maxiter=1000,
                                      popsize=20).x

a, b, c, d = popt_rational
print(f"Fitted rational function parameters: a={a}, b={b}, c={c}, d={d}")

fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)

ax.plot(dv_array, x0_array, 'x', color='white')
dv_fit = np.linspace(np.min(dv_array), np.max(dv_array), 100)
x0_fit = rational_func(dv_fit, *popt_rational)
ax.plot(dv_fit, x0_fit, '-', color='cyan', alpha=0.7)
ax.set_xlabel(r'$\Delta v$ [km s$^{-1}$]', fontsize=font_size)
ax.set_ylabel(r'$x_0$', fontsize=font_size)

plt.savefig('/mnt/c/Users/sgagn/OneDrive/Documents/phd/dv_3.pdf', dpi=300)
# plt.show()
plt.close()