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
import matplotlib as mpl
from matplotlib import patches
label_size = 20
font_size = 30
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size

presentation = False  # Set to True for presentation style
# presentation = True  # Set to True for presentation style
if presentation == True:
    plt.style.use('dark_background')
    color = 'white'
    color1 = 'cyan'
else:
    color = 'black'
    color1 = 'blue'
    color2 = 'red'

np.random.seed(44)

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

lum_flux_factor = 4*np.pi*(Planck18.luminosity_distance(5.0).to('cm').value)**2

def p_obs(lly, dv, lha, muv, theta, mode='wide'):
    """
    Probability of observing a galaxy with given Lya luminosity, H-alpha luminosity, and UV magnitude.
    """
    w1, f1, fh = theta
    # Convert luminosities to fluxes
    f_lya = lly / lum_flux_factor
    f_ha = lha / lum_flux_factor
    luv = 10**(0.4*(51.64 - muv))
    w_emerg = (1215.67/2.47e15)*(lly/luv)
    f_ha_lim = fh*2e-18  # H-alpha flux limit in erg/s/cm^2
    v_lim = 10**vcirc(muv)
    w_lim = w1
    f_lya_lim = f1
    # https://arxiv.org/pdf/2202.06642
    # https://arxiv.org/pdf/2003.12083
    # muv_lim = -18.0
    muv_lim = -17.75 # performing the integral to find the min muv that 
    # results in a mean of -18.7 reveals a value of -17.68, however 
    # we are likely still biased lower by w and f selections, so we take -17.75

    p_v = normal_cdf(dv, (6/5)*v_lim)
    p_lya = normal_cdf(f_lya, f_lya_lim)
    p_ha = normal_cdf(f_ha, f_ha_lim)
    p_w = normal_cdf(w_emerg, w_lim)
    p_muv = 1 - normal_cdf(10**muv, 6*(10**muv_lim))
    
    return p_lya * p_ha * p_w * p_muv * p_v

def get_beta_bouwens14(muv):
    # https://arxiv.org/pdf/1306.2950
    return -2.05 + -0.2*(muv+19.5)

muv_space = np.linspace(-22, -16, 1000)
log10w_space = np.linspace(0.5, 3, 1000)

phi_5 = 0.79
muv_star_5 = -21.1
alpha_5 = -1.74
def schechter(muv, phi, muv_star, alpha):
    return (0.4*np.log(10))*phi*(10**(0.4*(muv_star - muv)))**(alpha+1)*\
        np.exp(-10**(0.4*(muv_star - muv)))

NSAMPLES = 1250
STD_W = 0.36
w_min = 25
f_min = 2.2e-18  # erg/s/cm2
# muv_samples = np.random.normal(MEAN_MUV, STD_MUV, NSAMPLES)
muv_space = np.linspace(-24, -16, NSAMPLES)
ewpdf = True
p_muv = schechter(muv_space, phi_5, muv_star_5, alpha_5)
n_gal = np.trapezoid(p_muv, x=muv_space)*1e-3 # galaxy number density in Mpc^-3
EFFECTIVE_VOLUME = NSAMPLES/n_gal  # Mpc3, for normalization

p_muv /= np.sum(p_muv)
muv_samples = np.random.choice(muv_space, size=NSAMPLES, p=p_muv)
# muv_samples = np.random.uniform(-22, -16, NSAMPLES)
# log10w_samples = np.random.normal(MEAN_W, STD_W, NSAMPLES)
# MEAN_W = 1.5*np.ones_like(muv_samples)
MEAN_W = 0.12*(muv_samples + 18.5) + 1.49
log10w_samples = np.random.normal(MEAN_W, STD_W, NSAMPLES)

beta_samples = get_beta_bouwens14(muv_samples)
f_samples = ((10**log10w_samples) / \
             (4*np.pi*Planck18.luminosity_distance(5.0).to('cm').value**2)) * \
             (2.47e15/1215.67) * (10 ** (0.4*(51.6 - muv_samples))) * \
             (1215.67/1500)**(beta_samples + 2)

w_select = log10w_samples > np.log10(w_min)
muv_select = muv_samples < -18
f_select = f_samples > f_min
# select = muv_select & f_select
select = w_select & muv_select & f_select

print(f'Number of objects detected: {np.sum(select)}')

w_minf = (1215.67/2.47e15) * (4*np.pi*Planck18.luminosity_distance(5.0).to('cm').value**2) * \
          f_min * (10 ** (-0.4*(51.6 - muv_space))) * \
          (1215.67/1500)**(-1*get_beta_bouwens14(muv_space) - 2)

def line(x, m, b):
    return m*x + b

from scipy.optimize import curve_fit

print(f'N={np.sum(select)}')

# popt, pcov = curve_fit(line, muv_samples[select], log10w_samples[select])
# m_fit, b_fit = popt

fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)

ax.plot(muv_space, 10**np.log10(w_minf), '--', color=color, linewidth=2)
ax.axvline(-18, color=color, linestyle='--', linewidth=2)
ax.axhline(10**np.log10(w_min), color=color, linestyle='--', linewidth=2)
ax.set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
ax.set_ylabel(r'$W_{\rm emerg}^{\rm Ly\alpha}$ [$\AA$]', fontsize=font_size)
# ax.set_ylabel(r'$\log_{10}W_{\rm emerg}^{\rm Ly\alpha}$ [$\AA$]', fontsize=font_size)
ax.set_xlim(-22, -16)
ax.set_ylim(2, 600)

muv_bin_centers = np.array([-21.5, -20.2, -19])

for i, s in enumerate([True, False]):

    mean_ws = []
    std_ws = []

    for muv_bin in muv_bin_centers:
        bin_width = 2.0
        bin_select = (muv_samples >= muv_bin - bin_width/2) & (muv_samples < muv_bin + bin_width/2)
        f_obs = len(muv_samples[bin_select & select]) / len(muv_samples[bin_select])
        mean_w = np.log10(np.mean(10**log10w_samples[bin_select & select]))
        if s is True:
            mean_w += np.log10(f_obs)
        std_w = 10**mean_w / np.sqrt(len(muv_samples[bin_select & select]))
        mean_ws.append(mean_w)
        std_ws.append(std_w)
        if std_w > mean_w:
            lower_w = mean_w
        if s is True:
            ax.errorbar(muv_bin, 10**mean_w, yerr=std_w, fmt='*', color='red', markersize=20,
                        label='Schenker-like estimation' if muv_bin == -19 else None)
        else:
            ax.errorbar(muv_bin, 10**mean_w, yerr=std_w, fmt='*', color='green', markersize=20,
                        label='Naive estimation' if muv_bin == -19 else None)

    mean_ws = np.array(mean_ws)
    std_ws = np.array(std_ws)

    popt, pcov = curve_fit(line, muv_bin_centers, mean_ws, sigma=std_ws)
    m_fit, b_fit = popt

    if s is True:
        ax.plot(muv_space, 10**line(muv_space, m_fit, b_fit), '--', color='red', linewidth=3)
    else:
        ax.plot(muv_space, 10**line(muv_space, m_fit, b_fit), '--', color='green', linewidth=3)

low_select = muv_samples < -21
high_select = (muv_samples >= -19) & (muv_samples < -18)
mean_low = np.mean(10**log10w_samples[low_select & select])
mean_high = np.mean(10**log10w_samples[high_select & select])
mean_mean = (mean_low + mean_high) / 2
mean_diff = np.abs(mean_low - mean_high)
fobs_low = np.sum(select & low_select) / np.sum(low_select)
fobs_high = np.sum(select & high_select) / np.sum(high_select)
fobs_mean = (fobs_low + fobs_high) / 2
fobs_diff = np.abs(fobs_low - fobs_high)

print(mean_mean, mean_diff)
print(fobs_mean, fobs_diff)

def get_a(m):
            return fobs_mean + fobs_diff * np.tanh(3 * (m + 20.75))

def get_wc(m):
    return mean_mean + mean_diff * np.tanh(4 * (m + 20.25))

ax.plot(muv_space, get_wc(muv_space)*get_a(muv_space), '--', color='magenta', linewidth=3,
        label='Mason-like estimation')

ax.plot(muv_space, 10**(0.12*(muv_space + 18.5) + 1.49), '-', color='blue', linewidth=3, label='True relation')

ax.plot(muv_samples[select], 10**log10w_samples[select], 'o',
             markersize=5, alpha=1.0, color=color, label='Observed')
ax.plot(muv_samples[~select], 10**log10w_samples[~select], 'x',
             markersize=5, alpha=0.3, color=color, label='Not observed')

ax.legend(fontsize=16, loc='lower right')
ax.set_yscale('log')

savedir = '/mnt/c/Users/sgagn/OneDrive/Documents/phd/lyman_alpha/figures/'
savedir += 'pedagogy/'
os.makedirs(savedir, exist_ok=True)
plt.savefig(f'{savedir}spurious_vs_true_correlation.pdf', dpi=300)
plt.show()