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
presentation = True
# presentation = False
if presentation:
    plt.style.use('dark_background')
    cmap = 'Blues_r'
else:
    cmap = 'hot_r'

# from Bouwens 2021 https://arxiv.org/pdf/2102.07775
phi_5 = 0.79
muv_star_5 = -21.1
alpha_5 = -1.74

lum_flux_factor = 4*np.pi*(Planck18.luminosity_distance(5.0).to('cm').value)**2

def schechter(muv, phi, muv_star, alpha):
    return (0.4*np.log(10))*phi*(10**(0.4*(muv_star - muv)))**(alpha+1)*\
        np.exp(-10**(0.4*(muv_star - muv)))

def line(x, m, b):
    """
    Linear function.
    """
    return m * (x + 18.5) + b

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
        w_lim = 80*w1
        f_lya_lim = f1*2e-17
    elif mode == 'deep':
        w_lim = 25*w2
        f_lya_lim = f2*2e-18
    # https://arxiv.org/pdf/2202.06642
    # https://arxiv.org/pdf/2003.12083
    # muv_lim = -18.0
    muv_lim = -17.75

    p_v = normal_cdf(dv, (6/5)*v_lim)
    p_lya = normal_cdf(f_lya, f_lya_lim)
    p_ha = normal_cdf(f_ha, f_ha_lim)
    p_w = normal_cdf(w_emerg, w_lim)
    p_muv = 1 - normal_cdf(10**muv, 6*(10**muv_lim))
    
    return p_lya * p_ha * p_w * p_muv * p_v

def get_beta_bouwens14(muv):
    # https://arxiv.org/pdf/1306.2950
    return -2.05 + -0.2*(muv+19.5)

# A = np.load('../data/pca/A.npy')
I = np.array([[1,0,0],[0,1,0],[0,0,1]])
A1 = np.array([[0,0,0],[0,0,-1],[0,1,0]])
A2 = np.array([[0,0,1],[0,0,0],[-1,0,0]])
A3 = np.array([[0,-1,0],[1,0,0],[0,0,0]])
c1, c2, c3, c4 = 1, 1, 1/3, -1
A = c1 * I + c2 * A1 + c3 * A2 + c4 * A3
xc = np.load('../data/pca/xc.npy')
xstd = np.load('../data/pca/xstd.npy')
m1, m2, m3, b1, b2, b3, std1, std2, std3, w1, w2, f1, f2, fh = np.load('../data/pca/fit_params.npy')
theta = [w1, w2, f1, f2, fh]
NSAMPLES = 100000
muv_space = np.linspace(-22, -16, NSAMPLES)
p_muv = schechter(muv_space, phi_5, muv_star_5, alpha_5)
p_muv /= np.sum(p_muv)  # Normalize the probability distribution
muv_sample = np.random.choice(muv_space, p=p_muv, size=NSAMPLES)
muv_space = muv_sample

mu1 = line(muv_space, m1, b1)
mu2 = line(muv_space, m2, b2)
mu3 = line(muv_space, m3, b3)
y1 = np.random.normal(mu1, std1, NSAMPLES)
y2 = np.random.normal(mu2, std2, NSAMPLES)
y3 = np.random.normal(mu3, std3, NSAMPLES)
Y = np.vstack((y1, y2, y3))
X = (A @ Y) * xstd + xc

log10lya, dv, log10ha = X[0], X[1], X[2]
fesc = 10**(log10lya - log10ha - np.log10(8.7))
mean_fesc = np.mean(fesc)

beta = get_beta_bouwens14(muv_space)

flux_lya =  10**log10lya/lum_flux_factor

w_lya = (1215.67/2.47e15) * (10**log10lya / (10**(0.4*(51.6 - muv_space)))) * (1215.67/1500)**(-1*beta - 2)

from scipy.optimize import curve_fit

muv_select = muv_space < -18
heights, edges = np.histogram(w_lya[muv_select][(w_lya[muv_select]>40)*(w_lya[muv_select]<200)], bins=100)
bin_centers = 0.5 * (edges[1:] + edges[:-1])
popt, _ = curve_fit(lambda x, a, b: a*x + b, bin_centers, np.log10(heights))
print(popt)

w_range = np.linspace(0, 200, 1000)
muv_range = np.linspace(-22, -16, 20)

plt.hist(w_lya[muv_select], bins=100, color='blue', alpha=0.7, label='Lya Flux')
plt.plot(w_range, 10**(popt[0]*w_range + popt[1]), color='red', linestyle='--', label='Fit')
plt.yscale('log')
plt.show()
quit()

a_range = []
wc_range = []
for muv in muv_range:
    flya = flux_lya[np.abs(muv_space - muv) < 0.5]
    select = flya > 2e-18
    wc_range.append(np.mean(w_lya[np.abs(muv_space - muv) < 0.5][select]))
    a_range.append(len(flya[select])/len(flya))
    # print(muv, len(flya[select])/len(flya))
a_range = np.array(a_range)
wc_range = np.array(wc_range)

def tanh_fit(x, a, b, c):
    return a  + b*np.tanh(c*(x + 16.5))

def exponential_fit(x, a, b, c):
    return a + b * np.exp(c * (x + 16.5))

# popt, _ = curve_fit(tanh_fit, muv_range, a_range)
# print(popt)

# popt, _ = curve_fit(tanh_fit, muv_range, wc_range)
# print(popt)

popt, _ = curve_fit(exponential_fit, muv_range, wc_range)
print(popt)

# plt.plot(muv_range, a_range, marker='o', linestyle='-', color='blue')
plt.plot(muv_range, wc_range, marker='o', linestyle='-', color='orange')
plt.plot(muv_range, exponential_fit(muv_range, *popt), color='red', linestyle='--', label='Exponential Fit')
# plt.plot(muv_range, tanh_fit(muv_range, *popt)/wc_range - 1, color='red', linestyle='--', label='Tanh Fit')
plt.show()

quit()

print(theta)
theta = np.ones_like(theta)
_pobs = p_obs(10**log10lya, dv, 10**log10ha, muv_space, theta, mode='deep')
_pobs = np.ones_like(_pobs)  # For testing purposes, set all probabilities to 1

popt, _ = curve_fit(lambda x, a, b: a*x + b, beta[_pobs>0.5], np.log10(fesc[_pobs>0.5]))
print(popt)
beta_space = np.linspace(-3, 0, 1000)

# measured lya properties from https://arxiv.org/pdf/2402.06070
MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, fescA, \
    fescA_err, fescB, fescB_err, ID = np.load('../data/tang24.npy').T

beta = get_beta_bouwens14(MUV)

wide = ID == 0
deep = ID == 1
plt.errorbar(beta[wide], np.log10(fescA[wide]), xerr=0.2*MUV_err[wide], yerr=fescA_err[wide]/fescA[wide]/np.log(10), fmt='o', color='cyan', label='fesc A')
plt.errorbar(beta[deep], np.log10(fescB[deep]), xerr=0.2*MUV_err[deep], yerr=fescB_err[deep]/fescB[deep]/np.log(10), fmt='o', color='orange', label='fesc B')
plt.plot(beta_space, popt[0]*beta_space + popt[1], color='blue', linestyle='--', label='Fit')
plt.plot(beta_space, np.log10(0.54)-0.71*(beta_space+2.5), color='red', linestyle='--', label='Shifted Fit')
plt.show()
quit()


plt.hist2d(beta[_pobs>0.5], np.log10(fesc[_pobs>0.5]), bins=[50, 50], cmap='hot')

plt.colorbar(label='Counts')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$f_{\mathrm{esc}}$')
plt.show()

quit()
plt.hist(fesc, bins=100, color='blue', alpha=0.7)
plt.title(f'median fesc = {0.693*mean_fesc:.3f}, w/ selection = {np.mean(0.693*fesc[_pobs>0.5]):.3f}')
plt.show()

quit()


hist_res = 50
muv_side = np.linspace(-24, -16, hist_res)
lya_side = np.linspace(40, 45, hist_res)
lha_side = np.linspace(40, 45, hist_res)
dv_side = np.linspace(-250, 600, hist_res)

sides = [lya_side, dv_side, lha_side]
idc_pairs = [(0, 1), (0, 2), (1, 2)]

for pair in idc_pairs:
    idc1, idc2 = pair
    side1 = sides[idc1]
    side2 = sides[idc2]

    hist, _, _ = np.histogram2d(X[idc1], X[idc2], bins=[side1, side2])
    space1 = 0.5*(side1[1:] + side1[:-1])
    space2 = 0.5*(side2[1:] + side2[:-1])
    mean_arr = []
    lower_arr = []
    upper_arr = []
    for i in range(hist.shape[0]):
        weights = hist[i] / np.sum(hist[i])
        weight_dist = np.cumsum(weights)
        mean = space2[np.argmin(np.abs(weight_dist - 0.5))]
        lower_lim = space2[np.argmin(np.abs(weight_dist - 0.32))]
        upper_lim = space2[np.argmin(np.abs(weight_dist - 0.68))]
        mean_arr.append(mean)
        lower_arr.append(lower_lim)
        upper_arr.append(upper_lim)

    mean_arr = np.array(mean_arr)
    lower_arr = np.array(lower_arr)
    upper_arr = np.array(upper_arr)

    # how do I determine the start and end indices for the fit?
    unif_sample_per_pix = NSAMPLES / hist_res**2
    hist_sum = np.sum(hist, axis=1)
    select = hist_sum > unif_sample_per_pix
    space1 = space1[select]
    mean_arr = mean_arr[select]
    lower_arr = lower_arr[select]
    upper_arr = upper_arr[select]

    popt_mean, _ = curve_fit(lambda x, a, b: a*x + b, space1, mean_arr)
    popt_lower, _ = curve_fit(lambda x, a, b: a*x + b, space1, lower_arr)
    popt_upper, _ = curve_fit(lambda x, a, b: a*x + b, space1, upper_arr)

    print(popt_mean)
    print(np.mean(popt_upper[0]*space1 + popt_upper[1] - popt_mean[0]*space1 - popt_mean[1]))

    plt.hist2d(X[idc1], X[idc2], bins=[space1, space2], cmap=cmap,
            cmin=1, cmax=10000, norm=mpl.colors.LogNorm())
    plt.plot(space1, mean_arr, color='red', label='Mean')
    plt.fill_between(space1, lower_arr, upper_arr,
                    color='red', alpha=0.3, label='Std Dev')
    plt.plot(space1, popt_mean[0]*space1 + popt_mean[1], color='blue', linestyle='--', label='Fit Mean')
    plt.plot(space1, popt_lower[0]*space1 + popt_lower[1], color='green', linestyle='--', label='Fit Lower')
    plt.plot(space1, popt_upper[0]*space1 + popt_upper[1], color='orange', linestyle='--', label='Fit Upper')
    plt.show()