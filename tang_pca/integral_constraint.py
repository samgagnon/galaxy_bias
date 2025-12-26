"""
Compares scatter in the SGH model with the Mason et al. (2018) model
and the exponential functions of Tang et al. (2024).
"""

import os

import numpy as np
import py21cmfast as p21c

from tqdm import tqdm

from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, k_B, m_p, e, m_e

from scipy.special import gamma, erf
from scipy.integrate import trapezoid

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

presentation = False  # Set to True for presentation style
# presentation = True
if presentation == True:
    plt.style.use('dark_background')
    color1 = 'cyan'
    color2 = 'lime'
    color3 = 'orange'
    textcolor = 'white'
else:
    color1 = 'black'
    color2 = 'blue'
    color3 = 'red'
    color4 = 'orange'
    textcolor = 'black'

# from Bouwens 2021 https://arxiv.org/pdf/2102.07775
phi_5 = 0.79
muv_star_5 = -21.1
alpha_5 = -1.74

def get_beta_bouwens14(muv):
    # https://arxiv.org/pdf/1306.2950
    return -2.05 + -0.2*(muv+19.5)

def schechter(muv, phi, muv_star, alpha):
    return (0.4*np.log(10))*phi*(10**(0.4*(muv_star - muv)))**(alpha+1)*\
        np.exp(-10**(0.4*(muv_star - muv)))

def get_a(m):
            return 0.65 + 0.1 * np.tanh(3 * (m + 20.75))

def get_wc(m):
    return 31 + 12 * np.tanh(4 * (m + 20.25))

def mason2018(Muv):
    """
    Samples EW and emission probability from the
    fit functions obtained by Mason et al. 2018.
    """
    A = get_a(Muv)
    rv_A = np.random.uniform(0, 1, len(Muv))
    emit_bool = rv_A < A
    Wc = get_wc(Muv[emit_bool])
    rv_W = np.random.uniform(0, 1, len(Wc))
    W = -1*Wc*np.log(rv_W)
    return W, emit_bool

def get_silverrush_laelf(z):
    if z==4.9:
        # SILVERRUSH XIV z=4.9 LAELF
        lum_silver = np.array([42.75, 42.85, 42.95, 43.05, 43.15, 43.25, 43.35, 43.45, 43.55, 43.65])
        logphi_silver = -1*np.array([2.91, 3.17, 3.42, 3.78, 3.88, 4.00, 4.75, 4.93, 5.23, 4.93])
        logphi_up_silver = 1e-2*np.array([5, 5, 6, 9, 10, 12, 29, 36, 52, 36])
        logphi_low_silver = 1e-2*np.array([5, 5, 6, 9, 10, 12, 34, 45, 77, 45])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    elif z==5.7:
        # SILVERRUSH XIV z=5.7 LAELF
        lum_silver = np.array([42.85, 42.95, 43.05, 43.15, 43.25, 43.35, 43.45, 43.55, 43.65, 43.75, 43.85, 43.95])
        logphi_silver = -1*np.array([3.05, 3.27, 3.56, 3.85, 4.15, 4.41, 4.72, 5.15, 5.43, 6.03, 6.33, 6.33])
        logphi_up_silver = 1e-2*np.array([4, 2, 2, 3, 4, 5, 7, 12, 17, 36, 52, 52])
        logphi_low_silver = 1e-2*np.array([4, 2, 2, 3, 4, 5, 7, 13, 18, 45, 77, 77])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    elif z==6.6:
        # SILVERRUSH XIV z=6.6 LAELF
        lum_silver = np.array([42.95, 43.05, 43.15, 43.25, 43.35, 43.45, 43.55, 43.65, 43.75, 43.95, 44.05])
        logphi_silver = -1*np.array([3.71, 4.11, 4.37, 4.65, 4.83, 5.28, 5.89, 5.9, 5.9, 6.38, 6.38])
        logphi_up_silver = 1e-2*np.array([9, 5, 6, 7, 8, 14, 29, 29, 29, 52, 52])
        logphi_low_silver = 1e-2*np.array([9, 5, 6, 7, 8, 15, 34, 34, 34, 77, 77])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    elif z==7.0:
        # wip
        # SILVERRUSH XIV z=7.0 LAELF
        lum_silver = np.array([43.25, 43.35])
        logphi_silver = -1*np.array([4.4, 4.95])
        logphi_up_silver = 1e-2*np.array([29, 52])
        logphi_low_silver = 1e-2*np.array([34, 77])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    elif z==7.3:
        # wip
        # SILVERRUSH XIV z=7.3 LAELF
        lum_silver = np.array([43.45])
        logphi_silver = -1*np.array([4.81])
        logphi_up_silver = 1e-2*np.array([36])
        logphi_low_silver = 1e-2*np.array([45])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    
lum, logphi, logphi_up, logphi_low = get_silverrush_laelf(4.9)
bin_edges = np.zeros(len(lum) + 1)
bin_edges[0] = lum[0] - 0.5*(lum[1] - lum[0])
bin_edges[1:-1] = 0.5*(lum[1:] + lum[:-1])
bin_edges[-1] = lum[-1] + 0.5*(lum[-1] - lum[-2])

# from Bouwens 2021 https://arxiv.org/pdf/2102.07775
phi_5 = 0.79
muv_star_5 = -21.1
alpha_5 = -1.74

def schechter(muv, phi, muv_star, alpha):
    return (0.4*np.log(10))*phi*(10**(0.4*(muv_star - muv)))**(alpha+1)*\
        np.exp(-10**(0.4*(muv_star - muv)))

# muv_range = np.linspace(-24, -16, 1000)
# median_muv = np.zeros(len(muv_range))
# for i, muv in enumerate(muv_range):
#     weights = schechter(muv_range[muv_range<muv], phi_5, muv_star_5, alpha_5)
#     weights /= np.sum(weights)
#     median_muv[i] = np.sum(muv_range[muv_range<muv] * weights)

# print(muv_range[np.argmin(np.abs(median_muv + 18.7))])
# quit()

def schechter_lya(l, phi, l_star, alpha):
    return phi*(l/l_star)**alpha*np.exp(-l/l_star)

phi_5 = 10**-2.79
l_star_5 = 10**42.69
alpha_5 = -1.61

lrange = 10**np.linspace(41, 44, 1000)
flux_to_lum = 4 * np.pi * Planck18.luminosity_distance(5.0).to('cm').value**2
wide_lim = 3e-17 * flux_to_lum
deep_lim = 7e-18 * flux_to_lum

laelf_5 = schechter_lya(lrange, phi_5, l_star_5, alpha_5)

num_laes_wide = trapezoid(laelf_5[lrange>wide_lim], lrange[lrange>wide_lim]/l_star_5) * 256.7e3
num_laes_deep = trapezoid(laelf_5[lrange>deep_lim], lrange[lrange>deep_lim]/l_star_5) * 28.8e3

print("Number of LAEs in wide survey:", num_laes_wide)
print("Number of LAEs in deep survey:", num_laes_deep)


lrange = 10**np.linspace(41, 44, 1000)
flux_to_lum = 4 * np.pi * Planck18.luminosity_distance(5.0).to('cm').value**2
lim_range = 10**np.linspace(-18, -16, 100) * flux_to_lum
laelf_5 = schechter_lya(lrange, phi_5, l_star_5, alpha_5)
num_laes = np.zeros(len(lim_range))

for i, lim in enumerate(lim_range):
    num_laes[i] = trapezoid(laelf_5[lrange>lim], lrange[lrange>lim]/l_star_5)

flux_wide = lim_range[np.argmin(np.abs(num_laes*256.7e3 - 24))]/flux_to_lum
flux_deep = lim_range[np.argmin(np.abs(num_laes*28.8e3 - 36))]/flux_to_lum

print(flux_wide, flux_deep)
quit()
