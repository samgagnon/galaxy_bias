"""
Load in PCA basis and parameters from find_basis.py and likelihood.py 
and tweak the parameters to better fit the Umeda+24 LAELF
"""

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
label_size = 20
font_size = 30
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size

presentation = False  # Set to True for presentation style
# presentation = True
if presentation == True:
    plt.style.use('dark_background')

# from Bouwens 2021 https://arxiv.org/pdf/2102.07775
phi_5 = 0.79
muv_star_5 = -21.1
alpha_5 = -1.74

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

def get_beta_bouwens14(muv):
    # https://arxiv.org/pdf/1306.2950
    return -2.05 + -0.2*(muv+19.5)

def schechter(muv, phi, muv_star, alpha):
    return (0.4*np.log(10))*phi*(10**(0.4*(muv_star - muv)))**(alpha+1)*\
        np.exp(-10**(0.4*(muv_star - muv)))

def p_muv(muv, phi, muv_star, alpha):
    return schechter(muv, phi, muv_star, alpha)/gamma(alpha_5+2)/(0.4*np.log(10))/phi_5

# measured lya properties from https://arxiv.org/pdf/2402.06070
MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, fescA, \
    fescA_err, fescB, fescB_err, ID = np.load('../data/tang24.npy').T

beta = get_beta_bouwens14(MUV)
lum_lya = (ew_lya/1215.67) * 2.47e15 * 10**(0.4*(51.6 - MUV)) * (1215.6/1500) ** (beta + 2)
lum_lya_err = lum_lya*np.sqrt((ew_lya_err/ew_lya)**2 + (np.exp(-0.4*np.log(10)*MUV_err)/MUV)**2)

lum_ha = lum_lya / 11.4 / fescA
lum_ha_err = lum_ha*np.sqrt((lum_lya_err/lum_lya)**2 + (fescA_err/fescA)**2)

lum_flux_factor = 4*np.pi*(Planck18.luminosity_distance(5.0).to('cm').value)**2

wide = ID==0
# print(np.sum(wide), 'wide galaxies')
_muv_wide = MUV[wide]
_muv_err_wide = MUV_err[wide]
_dv_lya_wide = dv_lya[wide]
_dv_lya_err_wide = dv_lya_err[wide]
_lum_lya_wide = lum_lya[wide]
_lum_lya_err_wide = lum_lya_err[wide]
_lum_ha_wide = lum_ha[wide]
_lum_ha_err_wide = lum_ha_err[wide]
_fesc_wide = fescA[wide]
_fesc_err_wide = fescA_err[wide]
_ew_wide = ew_lya[wide]
_ew_err_wide = ew_lya_err[wide]

deep = ID==1
# print(np.sum(deep), 'deep galaxies')
# quit()
_muv_deep = MUV[deep]
_muv_err_deep = MUV_err[deep]
_dv_lya_deep = dv_lya[deep]
_dv_lya_err_deep = dv_lya_err[deep]
_lum_lya_deep = lum_lya[deep]
_lum_lya_err_deep = lum_lya_err[deep]
_lum_ha_deep = lum_ha[deep]
_lum_ha_err_deep = lum_ha_err[deep]
_fesc_deep = fescA[deep]
_fesc_err_deep = fescA_err[deep]
_ew_deep = ew_lya[deep]
_ew_err_deep = ew_lya_err[deep]

muv_wide = np.zeros(len(_muv_wide)*1000)
lum_lya_wide = np.zeros_like(muv_wide)
dv_wide = np.zeros_like(muv_wide)
lum_ha_wide = np.zeros_like(muv_wide)
fesc_wide = np.zeros_like(muv_wide)
ew_wide = np.zeros_like(muv_wide)

muv_deep = np.zeros(len(_muv_deep)*1000)
lum_lya_deep = np.zeros_like(muv_deep)
dv_deep = np.zeros_like(muv_deep)
lum_ha_deep = np.zeros_like(muv_deep)
fesc_deep = np.zeros_like(muv_deep)
ew_deep = np.zeros_like(muv_deep)

for i, (muv, muve, ew, ewe, lly, llye, dv, dve, lha, lhae, fe, fee) in enumerate(zip(_muv_wide, _muv_err_wide, \
                                                 _ew_wide, _ew_err_wide, _lum_lya_wide, _lum_lya_err_wide, \
                                                 _dv_lya_wide, _dv_lya_err_wide, _lum_ha_wide, _lum_ha_err_wide, \
                                                 _fesc_wide, _fesc_err_wide)):
    muv_wide[1000*i:1000*(i+1)] = np.random.normal(muv, muve, 1000)
    lum_lya_wide[1000*i:1000*(i+1)] = np.random.normal(lly, llye, 1000)
    dv_wide[1000*i:1000*(i+1)] = np.random.normal(dv, dve, 1000)
    lum_ha_wide[1000*i:1000*(i+1)] = np.random.normal(lha, lhae, 1000)
    fesc_wide[1000*i:1000*(i+1)] = np.random.normal(fe, fee, 1000)
    ew_wide[1000*i:1000*(i+1)] = np.random.normal(ew, ewe, 1000)
    ew_wide[1000*i:1000*(i+1)][ew_wide[1000*i:1000*(i+1)]<=0] = np.mean(ew_wide[1000*i:1000*(i+1)])  # replace with mean
    lum_lya_wide[1000*i:1000*(i+1)][lum_lya_wide[1000*i:1000*(i+1)]<10] = np.mean(lum_lya_wide[1000*i:1000*(i+1)])  # replace with mean
    dv_wide[1000*i:1000*(i+1)][dv_wide[1000*i:1000*(i+1)]<10] = np.mean(dv_wide[1000*i:1000*(i+1)])  # replace with mean
    lum_ha_wide[1000*i:1000*(i+1)][lum_ha_wide[1000*i:1000*(i+1)]<10] = np.mean(lum_ha_wide[1000*i:1000*(i+1)])  # replace with mean

    
for i, (muv, muve, ew, ewe, lly, llye, dv, dve, lha, lhae, fe, fee) in enumerate(zip(_muv_deep, _muv_err_deep, \
                                                 _ew_deep, _ew_err_deep, _lum_lya_deep, _lum_lya_err_deep, \
                                                 _dv_lya_deep, _dv_lya_err_deep, _lum_ha_deep, _lum_ha_err_deep,\
                                                 _fesc_deep, _fesc_err_deep)):
    muv_deep[1000*i:1000*(i+1)] = np.random.normal(muv, muve, 1000)
    lum_lya_deep[1000*i:1000*(i+1)] = np.random.normal(lly, llye, 1000)
    dv_deep[1000*i:1000*(i+1)] = np.random.normal(dv, dve, 1000)
    lum_ha_deep[1000*i:1000*(i+1)] = np.random.normal(lha, lhae, 1000)
    fesc_deep[1000*i:1000*(i+1)] = np.random.normal(fe, fee, 1000)
    ew_deep[1000*i:1000*(i+1)] = np.random.normal(ew, ewe, 1000)
    ew_deep[1000*i:1000*(i+1)][ew_deep[1000*i:1000*(i+1)]<=0] = np.mean(ew_deep[1000*i:1000*(i+1)])  # replace with mean
    lum_lya_deep[1000*i:1000*(i+1)][lum_lya_deep[1000*i:1000*(i+1)]<10] = np.mean(lum_lya_deep[1000*i:1000*(i+1)])  # replace with mean
    dv_deep[1000*i:1000*(i+1)][dv_deep[1000*i:1000*(i+1)]<10] = np.mean(dv_deep[1000*i:1000*(i+1)])  # replace with mean
    lum_ha_deep[1000*i:1000*(i+1)][lum_ha_deep[1000*i:1000*(i+1)]<10] = np.mean(lum_ha_deep[1000*i:1000*(i+1)])  # replace with mean

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

def p_obs(lly, dv, lha, muv, theta, mode='wide'):
    """
    Probability of observing a galaxy with given Lya luminosity, H-alpha luminosity, and UV magnitude.
    """
    w1, w2, f1, f2, fh = theta
    # Convert luminosities to fluxes
    f_lya = lly / lum_flux_factor
    f_ha = lha / lum_flux_factor
    luv = 10**(0.4*(51.64 - muv))
    w_emerg = (1215.67/2.47e15)*(lly/luv)
    f_ha_lim = fh*2e-18  # H-alpha flux limit in erg/s/cm^2
    v_lim = 10**vcirc(muv)
    if mode == 'wide':
        w_lim = 80*w1 # 80
        f_lya_lim = f1*2e-17 # 2e-17
    elif mode == 'deep':
        w_lim = 25*w2
        f_lya_lim = f2*2e-18 # 2e-18
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

def p_f(_p_obs, f, fe):
    _f = np.sum(_p_obs) / len(_p_obs)
    _f_err = _f * np.sqrt((np.sqrt(len(_p_obs)) / len(_p_obs))**2 + (fe/f)**2)
    dist = np.exp(-0.5 * ((f - _f) / _f_err) ** 2)
    if np.isinf(dist):
        dist = 100
    # get Gaussian distance from expected observed fraction
    return dist

def line(x, m, b):
    """
    Linear function.
    """
    return m * (x + 18.5) + b

XWIDE = np.array([np.log10(lum_lya_wide), dv_wide, np.log10(lum_ha_wide)])
XDEEP = np.array([np.log10(lum_lya_deep), dv_deep, np.log10(lum_ha_deep)])

# Y = PCA T of X
os.makedirs('../data/pca', exist_ok=True)

# pca = PCA(n_components=3)  # Keeping all 3 dimensions

# transform the wide sample to the PCA basis
XALL = np.concatenate((XWIDE, XDEEP), axis=1)

# XC = XALL.mean(axis=1, keepdims=True)
# XSTD = XALL.std(axis=1, keepdims=True)
XC = np.load('../data/pca/xc.npy')
XSTD = np.load('../data/pca/xstd.npy')
XALL0 = (XALL - XC) / XSTD

# T = np.load('../data/pca/A.npy')
# coeff = np.load('../data/pca/coefficients.npy')
I = np.array([[1,0,0],[0,1,0],[0,0,1]])
A1 = np.array([[0,0,0],[0,0,-1],[0,1,0]])
A2 = np.array([[0,0,1],[0,0,0],[-1,0,0]])
A3 = np.array([[0,-1,0],[1,0,0],[0,0,0]])
c1, c2, c3, c4 = 1, 1, 1/3, -1
# c1, c2, c3, c4 = np.load('../data/pca/coefficients.npy')
T = c1 * I + c2 * A1 + c3 * A2 + c4 * A3

YALL = np.linalg.inv(T) @ XALL0

XWIDE0 = XWIDE - XC
XDEEP0 = XDEEP - XC

YWIDE = np.linalg.inv(T) @ XWIDE0
YDEEP = np.linalg.inv(T) @ XDEEP0


lum, logphi, logphi_up, logphi_low = get_silverrush_laelf(4.9)
bin_edges = np.zeros(len(lum) + 1)
bin_edges[0] = lum[0] - 0.5*(lum[1] - lum[0])
bin_edges[1:-1] = 0.5*(lum[1:] + lum[:-1])
bin_edges[-1] = lum[-1] + 0.5*(lum[-1] - lum[-2])
phi_err_up = (10**(logphi + logphi_up) - 10**logphi)
phi_err_low = (10**logphi - 10**(logphi - logphi_low))
phi_err = 0.5 * (phi_err_up + phi_err_low)
lum = np.array(lum)
logphi = np.array(logphi)
logphi_up = np.array(logphi_up)
logphi_low = np.array(logphi_low)
phi_err = np.array(phi_err)

NSAMPLES = 1000000
ewpdf = False  # Set to True to compute the EW PDF
muv_space = np.linspace(-24, -16, NSAMPLES)
p_muv = schechter(muv_space, phi_5, muv_star_5, alpha_5)
n_gal = np.trapezoid(p_muv, x=muv_space)*1e-3 # galaxy number density in Mpc^-3
EFFECTIVE_VOLUME = NSAMPLES/n_gal  # Mpc3, for normalization
p_muv /= np.sum(p_muv)
muv_sample = np.random.choice(muv_space, size=NSAMPLES, p=p_muv)

# Gagnon-Hartman et al. (2025) model
I = np.array([[1,0,0],[0,1,0],[0,0,1]])
A1 = np.array([[0,0,0],[0,0,-1],[0,1,0]])
A2 = np.array([[0,0,1],[0,0,0],[-1,0,0]])
A3 = np.array([[0,-1,0],[1,0,0],[0,0,0]])
# c1, c2, c3, c4 = np.load('../data/pca/coefficients.npy')
A = c1 * I + c2 * A1 + c3 * A2 + c4 * A3
xc = np.load('../data/pca/xc.npy')
xstd = np.load('../data/pca/xstd.npy')
m1, m2, m3, b1, b2, b3, std1, std2, std3, w1, w2, f1, f2, fh = np.load('../data/pca/fit_params.npy')
stannorm1, stannorm2, stannorm3 = np.random.normal(0, 1, NSAMPLES), \
                                    np.random.normal(0, 1, NSAMPLES), \
                                    np.random.normal(0, 1, NSAMPLES)

def fit():

    def objective(params):
        m1, m2, m3, b1, b2, b3, std1, std2, std3 = params

        u1, u2, u3 = stannorm1*std1, stannorm2*std2, stannorm3*std3
        u1 += line(muv_sample, m1, b1)
        u2 += line(muv_sample, m2, b2)
        u3 += line(muv_sample, m3, b3)

        log10lya, _, _ = (A @ np.array([u1, u2, u3]))* xstd + xc

        heights_sgh, bins_sgh = np.histogram(log10lya, bins=bin_edges, density=False)
        bin_widths = bins_sgh[1:] - bins_sgh[:-1]
        height_err_sgh = np.sqrt(heights_sgh) / bin_widths / EFFECTIVE_VOLUME
        heights_sgh = heights_sgh / bin_widths / EFFECTIVE_VOLUME

        # compute chi2
        sigma = np.sqrt(phi_err ** 2 + height_err_sgh**2)
        chi2 = -0.5 * (heights_sgh - 10**logphi)**2 / sigma**2

        return -1*chi2.sum()

    x0 = m1, m2, m3, b1, b2, b3, std1, std2, std3
    bounds = []
    for i, x in enumerate(x0):
        if x < 0:
            lower_bound = 1.1*x
            upper_bound = 0.9*x
        else:
            lower_bound = 0.9*x
            upper_bound = 1.1*x
        bounds.append((lower_bound, upper_bound))
    result = differential_evolution(objective, bounds, x0=x0, maxiter=1000, popsize=1, tol=1e-7, disp=True)
    print("Optimization Result:")
    print(result)
    return result.x

fit()