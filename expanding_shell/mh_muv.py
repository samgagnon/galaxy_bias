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

from halomod import TracerHaloModel

def get_stellar_mass(halo_masses, stellar_rng):
    sigma_star = 0.5
    mp1 = 1e10
    mp2 = 2.8e11
    M_turn = 10**(8.7)
    a_star = 0.5
    a_star2 = -0.61
    f_star10 = 0.05
    omega_b = Planck18.Ob0
    omega_m = Planck18.Om0
    baryon_frac = omega_b/omega_m
    
    high_mass_turnover = ((mp2/mp1)**a_star + (mp2/mp1)**a_star2)/((halo_masses/mp2)**(-1*a_star)+(halo_masses/mp2)**(-1*a_star2))
    stoc_adjustment_term = 0.5*sigma_star**2
    low_mass_turnover = np.exp(-1*M_turn/halo_masses + stellar_rng*sigma_star - stoc_adjustment_term)
    stellar_mass = f_star10 * baryon_frac * halo_masses * (high_mass_turnover * low_mass_turnover)
    return stellar_mass

def get_sfr(stellar_mass, sfr_rng, z):
    sigma_sfr_lim = 0.19
    sigma_sfr_idx = -0.12
    t_h = 1/Planck18.H(z).to('s**-1').value
    t_star = 0.5
    sfr_mean = stellar_mass / (t_star * t_h)
    sigma_sfr = sigma_sfr_idx * np.log10(stellar_mass/1e10) + sigma_sfr_lim
    sigma_sfr[sigma_sfr < sigma_sfr_lim] = sigma_sfr_lim
    stoc_adjustment_term = sigma_sfr * sigma_sfr / 2. # adjustment to the mean for lognormal scatter
    sfr_sample = sfr_mean * np.exp(sfr_rng*sigma_sfr - stoc_adjustment_term)
    return sfr_sample

def change_of_basis(x, mu, sigma):
    """
    Change of basis to standard normal distribution.
    """
    return (x - mu) / sigma

def standard_normal_pdf(x):
    """
    Probability density function for a standard normal distribution.
    """
    return (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)

def standard_normal_cdf(x):
    """
    Cumulative distribution function for a standard normal distribution.
    """
    return 0.5 * (1 + erf(x / np.sqrt(2)))

def skewnormal_pdf(x, mu, sigma, alpha):
    """
    Probability density function for a skew normal distribution.
    """
    x = change_of_basis(x, mu, sigma)
    phi = standard_normal_pdf(x)
    Phi = standard_normal_cdf(alpha * x)
    return 2 * phi * Phi

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

n_halos = 1000000

hm_smt6 = TracerHaloModel(
    z=6.0,  # Redshift
    hmf_model="Tinker08",  # tinker 08 halo mass function
    cosmo_params={"Om0": Planck18.Om0, "H0": Planck18.H0.value},
)

m, dndm = hm_smt6.m, hm_smt6.dndm
dndm = dndm[m>1e10]
m = m[m>1e10]
p_m = dndm/np.sum(dndm)

# mh = np.random.choice(m, size=n_halos, p=p_m)
mh = 10**np.random.uniform(9, 16, size=n_halos)
sfr_rng = np.random.normal(0, 1, size=n_halos)
stellar_rng = np.random.normal(0, 1, size=n_halos)
mstar = get_stellar_mass(mh, stellar_rng)
sfr = get_sfr(mstar, sfr_rng, z=6.0)
muv = -2.5 * (np.log10(sfr) + np.log10(3.1557e7) + np.log10(1.15e28)) + 51.64


fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
ax.hist2d(np.log10(mh), muv, bins=[100, 100], range=[[9, 16], [-24, -16]], density=True, cmap='hot', rasterized=True)
ax.set_xlabel(r'$\log_{10}(M_{\rm h}/M_\odot)$', fontsize=font_size)
ax.set_ylabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
# plt.show()

savedir = '/mnt/c/Users/sgagn/Documents/phd/lyman_alpha/expanding_shell/'
os.makedirs(savedir, exist_ok=True)

plt.savefig(f'{savedir}halo_mass_uv_magnitude_distribution.pdf', dpi=1000)
plt.clf()

hist2d, xedges, yedges = np.histogram2d(np.log10(mh), muv, bins=[100, 16], range=[[9, 16], [-24, -16]], density=True)

muv_centers = 0.5 * (yedges[1:] + yedges[:-1])
mu = []
std = []
alpha = []
row_pdf_norm_list = []
skew_pdf_norm_list = []

fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

for row in hist2d.T:
    xcenters = 0.5 * (xedges[1:] + xedges[:-1])
    row_pdf = row / np.sum(row)

    def objective_func(params):
        mu, sigma, alpha = params
        model_pdf = skewnormal_pdf(xcenters, mu, sigma, alpha)
        model_pdf /= np.max(model_pdf)
        row_pdf_norm = row_pdf / np.max(row_pdf)
        return np.sum((model_pdf - row_pdf_norm)**2)
    
    pvar = differential_evolution(objective_func,
                                    bounds=[(9, 15), (0.1, 5), (0, 10)],
                                    strategy='best1bin',
                                    maxiter=1000,
                                    popsize=15).x
    
    mu.append(pvar[0])
    std.append(pvar[1])
    alpha.append(pvar[2])

    row_pdf_norm = row_pdf / np.max(row_pdf)
    skew_pdf = skewnormal_pdf(xcenters, *pvar)
    skew_pdf_norm = skew_pdf / np.max(skew_pdf)
    row_pdf_norm_list.append(row_pdf_norm)
    skew_pdf_norm_list.append(skew_pdf_norm)
    ax.plot(xcenters, row_pdf_norm, color='white', linestyle=':', alpha=0.7)
    ax.plot(xcenters, skew_pdf_norm, color='cyan', linestyle='-', alpha=0.7)

ax.set_xlabel(r'$\log_{10}(M_{\rm h}/M_\odot)$', fontsize=font_size)
ax.set_ylabel(r'PDF', fontsize=font_size)
# plt.show()
plt.savefig(f'{savedir}halo_mass_uv_magnitude_skewnormal_fits.pdf', dpi=1000)
plt.clf()

mu = np.array(mu)
std = np.array(std)
alpha = np.array(alpha)

# fit mu with double power law
popt_mu, pcov_mu = curve_fit(double_power, muv_centers, mu, p0=[10, -2, -2, -20])
mu_fit = double_power(muv_centers, *popt_mu)
print(f'Fitted mu parameters: {popt_mu}')

# fit std with sigmoid
popt_std, pcov_std = curve_fit(sigmoid, muv_centers, std, p0=[2, -20, 1, 0.5])
std_fit = sigmoid(muv_centers, *popt_std)
print(f'Fitted std parameters: {popt_std}')

# fit alpha with lorentzian
popt_alpha, pcov_alpha = curve_fit(lorentzian, muv_centers, alpha, p0=[-20, 2, 5, 1])
alpha_fit = lorentzian(muv_centers, *popt_alpha)
print(f'Fitted alpha parameters: {popt_alpha}')

fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
axs[0].plot(muv_centers, mu, ':', color='white', label=r'$\mu$')
axs[0].plot(muv_centers, mu_fit, '-', color='cyan', label='Double Power Law Fit')
axs[0].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
axs[0].set_ylabel(r'$\langle \log_{10}(M_{\rm h}/M_\odot) \rangle$', fontsize=font_size)
axs[1].plot(muv_centers, std, ':', color='white', label=r'$\sigma$')
axs[1].plot(muv_centers, std_fit, '-', color='cyan', label='Sigmoid Fit')
axs[1].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
axs[1].set_ylabel(r'$\sigma$', fontsize=font_size)
axs[2].plot(muv_centers, alpha, ':', color='white', label=r'$\alpha$')
axs[2].plot(muv_centers, alpha_fit, '-', color='cyan', label='Lorentz Fit')
axs[2].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
axs[2].set_ylabel(r'$\alpha$', fontsize=font_size)
# plt.show()
plt.savefig(f'{savedir}halo_mass_uv_magnitude_skewnormal_params.pdf', dpi=1000)
plt.clf()

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

fig, axs = plt.subplots(figsize=(8, 6), constrained_layout=True)
for i in range(len(muv_centers)):
    skew_parametric = skewnormal_pdf(xcenters, mu_fit[i], std_fit[i], alpha_fit[i])
    skew_parametric_norm = skew_parametric / np.max(skew_parametric)
    samples = mh_from_muv(np.ones(100000) * muv_centers[i])
    hist, _ = np.histogram(samples, bins=xedges, density=True)
    row_pdf_norm = hist / np.max(hist)

    axs.plot(xcenters, row_pdf_norm_list[i], color='white', linestyle=':', alpha=0.5)
    axs.plot(xcenters, skew_pdf_norm_list[i], color='cyan', linestyle='--', alpha=0.5)
    axs.plot(xcenters, skew_parametric_norm, color='orange', linestyle='-', alpha=1.0)
    axs.plot(xcenters, row_pdf_norm, color='magenta', linestyle='-.', alpha=1.0)
axs.set_xlabel(r'$\log_{10}(M_{\rm h}/M_\odot)$', fontsize=font_size)
axs.set_ylabel(r'PDF', fontsize=font_size)
# plt.show()
plt.savefig(f'{savedir}halo_mass_uv_magnitude_skewnormal_fits_parametric.pdf', dpi=1000)
plt.clf()

# now for the final plot comparing my original hist 2d to the one from the parametric model
fig, ax = plt.subplots(1, 3, figsize=(14, 6), constrained_layout=True, sharey=True)
hist2d, xedges, yedges = np.histogram2d(np.log10(mh), muv, bins=[100, 100], range=[[9, 16], [-24, -16]], density=True)
ax[0].pcolormesh(xedges, yedges, hist2d.T, cmap='hot', shading='auto', rasterized=True)
ax[0].set_xlabel(r'$\log_{10}(M_{\rm h}/M_\odot)$', fontsize=font_size)
ax[0].set_ylabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
ax[0].set_title('Original', fontsize=font_size)

# generate new mh samples from muv
muv_samples = np.random.uniform(-24, -16, size=n_halos)
mh_parametric = mh_from_muv(muv_samples)
hist2d_parametric, xedges, yedges = np.histogram2d(mh_parametric, muv_samples, bins=[100, 100], range=[[9, 16], [-24, -16]], density=True)
hist2d_normalized = hist2d_parametric * np.sum(hist2d, axis=0) / np.sum(hist2d_parametric, axis=0)
ax[1].pcolormesh(xedges, yedges, hist2d_normalized.T, cmap='hot', shading='auto', rasterized=True)
ax[1].set_xlabel(r'$\log_{10}(M_{\rm h}/M_\odot)$', fontsize=font_size)
ax[1].set_title('Parametric', fontsize=font_size)

# residuals
from matplotlib.colors import LinearSegmentedColormap

color_min    = "#4203ff"
color_center = "black"
color_max    = "#ff0342"
cmap = LinearSegmentedColormap.from_list(
    "cmap_name",
    [color_min, color_center, color_max]
)

im = ax[2].pcolormesh(xedges, yedges, (hist2d.T - hist2d_normalized.T), cmap=cmap, shading='auto', rasterized=True)
cb = fig.colorbar(im, ax=ax[2])
cb.set_label('Residuals', fontsize=font_size)
ax[2].set_xlabel(r'$\log_{10}(M_{\rm h}/M_\odot)$', fontsize=font_size)
ax[2].set_title('Residuals', fontsize=font_size)

# plt.show()
plt.savefig(f'{savedir}halo_mass_uv_magnitude_distribution_comparison.pdf', dpi=1000)
plt.clf()