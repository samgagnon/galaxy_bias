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
    color1 = 'blue'
    color2 = 'black'
    color3 = 'red'
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

muv_t24, muv_emu, muv_dex = np.array([[-19.5, -18.5, -17.5], \
                                      [10, 16, 27], [1.75, 1.51, 0.99]])
# muv_dex /= np.log(10)

from scipy.optimize import curve_fit
emu_popt, _ = curve_fit(lambda x, a, b: a * x + b, muv_t24, muv_emu)
dex_popt, _ = curve_fit(lambda x, a, b: a * x + b, muv_t24, muv_dex)

def mean_t24(muv):
    mean = np.zeros_like(muv)
    mean[muv<=-19.5] = muv_emu[0]
    mean[muv>-17.5] = muv_emu[2]
    mean[mean==0] = muv_emu[1]
    return mean

def sigma_t24(muv):
    sigma = np.zeros_like(muv)
    sigma[muv<=-19.5] = muv_dex[0]
    sigma[muv>-17.5] = muv_dex[2]
    sigma[sigma==0] = muv_dex[1]
    return sigma/np.log(10)

NSAMPLES = 1000000
WMIN = 1

muv_space = np.linspace(-20.25, -18.75, NSAMPLES)
p_muv = schechter(muv_space, phi_5, muv_star_5, alpha_5)
p_muv /= np.sum(p_muv)
muv_sample = np.random.choice(muv_space, size=NSAMPLES, p=p_muv)
w_m18, emit_bool_m18 = mason2018(muv_sample)
bool_m18 = w_m18 > WMIN
bool_10_m18 = w_m18 > 10
bool_25_m18 = w_m18 > 25
f10_m18 = np.sum(emit_bool_m18) / NSAMPLES - np.sum(~bool_10_m18) / NSAMPLES
f25_m18 = np.sum(emit_bool_m18) / NSAMPLES - np.sum(~bool_25_m18) / NSAMPLES
w_m18 = w_m18[bool_m18]

# w_t24 = np.random.lognormal(mean=np.log(emu_popt[0]*muv_sample + emu_popt[1]),
#                             sigma=dex_popt[0]*muv_sample + dex_popt[1],
#                             size=NSAMPLES)
w_t24 = np.random.lognormal(mean=np.log(mean_t24(muv_sample)),
                            sigma=sigma_t24(muv_sample),
                            size=NSAMPLES)
bool_t24 = w_t24 > 1
bool_10_t24 = w_t24 > 10
bool_25_t24 = w_t24 > 25
f10_t24 = np.sum(bool_10_t24) / NSAMPLES
f25_t24 = np.sum(bool_25_t24) / NSAMPLES
w_t24 = w_t24[bool_t24]

I = np.array([[1,0,0],[0,1,0],[0,0,1]])
A1 = np.array([[0,0,0],[0,0,-1],[0,1,0]])
A2 = np.array([[0,0,1],[0,0,0],[-1,0,0]])
A3 = np.array([[0,-1,0],[1,0,0],[0,0,0]])
c1, c2, c3, c4 = 1, 1, 1/3, -1
A = c1 * I + c2 * A1 + c3 * A2 + c4 * A3
xc = np.load('../data/pca/xc.npy')
xstd = np.load('../data/pca/xstd.npy')
m1, m2, m3, b1, b2, b3, std1, std2, std3, w1, w2, f1, f2, fh = np.load('../data/pca/fit_params.npy')
u1, u2, u3 = np.random.normal(m1*(muv_sample + 18.5) + b1, std1, NSAMPLES), \
              np.random.normal(m2*(muv_sample + 18.5) + b2, std2, NSAMPLES), \
              np.random.normal(m3*(muv_sample + 18.5) + b3, std3, NSAMPLES)
log10lya, dv, log10ha = (A @ np.array([u1, u2, u3]))* xstd + xc
w_sgh = (1215.67/2.47e15)*(10**log10lya)*10**(-0.4*(51.6-muv_sample))*\
    (1215.67/1500)**(-1*get_beta_bouwens14(muv_sample)-2)
bool_sgh = w_sgh > WMIN
bool_10_sgh = w_sgh > 10
bool_25_sgh = w_sgh > 25
f10_sgh = np.sum(bool_10_sgh) / NSAMPLES
f25_sgh = np.sum(bool_25_sgh) / NSAMPLES
w_sgh = w_sgh[bool_sgh]

print(f'MUV: {np.min(muv_sample):.2f} to {np.max(muv_sample):.2f}')
print(f'Fraction of emitters with W > 10 A: {f10_sgh:.2f} (SGH), {f10_t24:.2f} (T24), {f10_m18:.2f} (M18)')
print(f'Fraction of emitters with W > 25 A: {f25_sgh:.2f} (SGH), {f25_t24:.2f} (T24), {f25_m18:.2f} (M18)')

# measured lya properties from https://arxiv.org/pdf/2402.06070
MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, fescA, \
    fescA_err, fescB, fescB_err, ID = np.load('../data/tang24.npy').T

ew_data = np.zeros(1000*len(ew_lya))
select = (MUV>-20.25)*(MUV<-18.25)
for i, (ew, ewe) in enumerate(zip(ew_lya[select], \
                                  ew_lya_err[select])):
    ew_data[i*1000:(i+1)*1000] = np.random.normal(ew, ewe, 1000)
ew_data = ew_data[ew_data > 1]

fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
ax.hist(np.log10(ew_data), bins=100, density=True, histtype='step', label='SGH model', color=color1, linestyle='--')
ax.hist(np.log10(w_sgh), bins=100, density=True, histtype='step', label='SGH model', color=color1)
ax.hist(np.log10(w_t24), bins=100, density=True, histtype='step', label='Tang et al. 2024', color=color2)
ax.hist(np.log10(w_m18), bins=100, density=True, histtype='step', label='Mason et al. 2018', color=color3)
ax.set_yscale('log')
ax.set_xlabel(r'$\log_{10} W^{\rm Ly\alpha}_{\rm emerg}$ [$\AA$]', fontsize=font_size)
ax.set_ylabel(r'$P(W^{\rm Ly\alpha}_{\rm emerg})$', fontsize=font_size)
plt.show()