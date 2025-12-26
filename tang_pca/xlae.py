"""
Compare the fraction of LAEs with W>25 A to the JWST average from Table 1 of 
https://arxiv.org/abs/2508.14171
"""

import os

import numpy as np
import py21cmfast as p21c

from tqdm import tqdm

from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, k_B, m_p, e, m_e

from scipy.special import gamma, erf

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

def lum2mag(lum, z):
    lyman_alpha = 1215.67e-7 #m
    lyman_alpha = c.cgs.value/lyman_alpha
    freq = lyman_alpha/(1+z)
    flux = lum/(4*np.pi*(Planck18.luminosity_distance(z).to('cm').value)**2)
    intensity = flux/freq
    return -2.5*np.log10(intensity) - 48.6

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

def phi_z(z):
    phi = 4e-1*10**(-0.33*(z - 6) - 0.024*(z - 6)**2)
    return phi

def muv_star_z(z):
    z_t = 2.46
    if z < z_t:
        muv_star = -20.89 - 1.09 * (z - z_t)
    else:
        muv_star = -21.03 - 0.04 * (z - 6)
    return muv_star

def alpha_z(z):
    alpha = -1.94 + -0.11*(z - 6)
    return alpha

redshifts = [5, 6, 7, 8, 9]
xLAE_ref = [15.3, 21.8, 14.8, 21, 10]
xLAE_ref_err = [2.1, 3.5, 2.7, 9, 6]
NSAMPLES = 100000
muv_space = np.linspace(-24, -17, NSAMPLES)
p_muv = schechter(muv_space, phi_5, muv_star_5, alpha_5)

p_muv /= np.sum(p_muv)
muv_sample = np.random.choice(muv_space, size=NSAMPLES, p=p_muv)

# Gagnon-Hartman et al. (2025) model
I = np.array([[1,0,0],[0,1,0],[0,0,1]])
A1 = np.array([[0,0,0],[0,0,-1],[0,1,0]])
A2 = np.array([[0,0,1],[0,0,0],[-1,0,0]])
A3 = np.array([[0,-1,0],[1,0,0],[0,0,0]])
c1, c2, c3, c4 = 1, 1, 1/3, -1
c1, c2, c3, c4 = np.load('../data/pca/coefficients.npy')
A = c1 * I + c2 * A1 + c3 * A2 + c4 * A3
xc = np.load('../data/pca/xc.npy')
xstd = np.load('../data/pca/xstd.npy')
m1, m2, m3, b1, b2, b3, std1, std2, std3, w1, w2, f1, f2, fh = np.load('../data/pca/fit_params.npy')
u1, u2, u3 = np.random.normal(m1*(muv_sample + 18.5) + b1, std1, NSAMPLES), \
    np.random.normal(m2*(muv_sample + 18.5) + b2, std2, NSAMPLES), \
    np.random.normal(m3*(muv_sample + 18.5) + b3, std3, NSAMPLES)
log10lya, dv, log10ha = (A @ np.array([u1, u2, u3]))* xstd + xc

lum_to_flux = 4*np.pi*(Planck18.luminosity_distance(6).to(u.cm).value)**2
f_lya = log10lya - np.log10(lum_to_flux)

# Mason et al. (2018) model
w_m18, emit_bool_m18 = mason2018(muv_sample)
f10_m18 = np.sum(w_m18 > 10) / NSAMPLES
f25_m18 = np.sum(w_m18 > 25) / NSAMPLES
log10lya_m18 = np.log10(w_m18*(2.47e15/1215.67)*(1500/1215.67)**(get_beta_bouwens14(muv_sample[emit_bool_m18])+2)*\
    10**(0.4*(51.6-muv_sample[emit_bool_m18])))

f10lya_m18 = log10lya_m18 - np.log10(lum_to_flux)



mab = lum2mag(10**log10lya_m18, 6)
plt.plot(mab, f10lya_m18, '.', color='blue', alpha=0.1, label='Mason+18 model')
plt.show()

# plt.hist2d(muv_sample[emit_bool_m18], f10lya_m18, bins=50, range=[[-24,-17], [-19, -15]], cmap='Blues',)
# plt.show()