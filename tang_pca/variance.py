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
plt.style.use('dark_background')
import matplotlib as mpl
label_size = 20
font_size = 30
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size

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
# print(A, xc, xstd, m1, m2, m3, b1, b2, b3, std1, std2, std3)
# print((A[0,0]*m1 + A[0,1]*m2 + A[0,2]*m3)*xstd[0], \
#     np.sqrt(np.abs(A[0,0])*std1**2 + np.abs(A[0,1])*std2**2 + np.abs(A[0,2])*std3**2)*xstd[0], \
#     (A[0,0]*b1 + A[0,1]*b2 + A[0,2]*b3)*xstd[0] + xc[0])
# print((A[1,0]*m1 + A[1,1]*m2 + A[1,2]*m3)*xstd[1], \
#     np.sqrt(np.abs(A[1,0])*std1**2 + np.abs(A[1,1])*std2**2 + np.abs(A[1,2])*std3**2)*xstd[1], \
#     (A[1,0]*b1 + A[1,1]*b2 + A[1,2]*b3)*xstd[1] + xc[1])
# print((A[2,0]*m1 + A[2,1]*m2 + A[2,2]*m3)*xstd[2], \
#     np.sqrt(np.abs(A[2,0])*std1**2 + np.abs(A[2,1])*std2**2 + np.abs(A[2,2])*std3**2)*xstd[2], \
#     (A[2,0]*b1 + A[2,1]*b2 + A[2,2]*b3)*xstd[2] + xc[2])
# quit()
NSAMPLES = 10000
muv_space = np.linspace(-24, -16, NSAMPLES)
p_muv = schechter(muv_space, phi_5, muv_star_5, alpha_5)
p_muv /= np.sum(p_muv)  # Normalize the probability distribution
muv_sample = np.random.choice(muv_space, p=p_muv, size=NSAMPLES)
muv_space = muv_sample
hist_res = 25
muv_side = np.linspace(-24, -16, hist_res)
lya_side = np.linspace(40, 45, hist_res)
lha_side = np.linspace(40, 45, hist_res)
dv_side = np.linspace(-250, 600, hist_res)

mu1 = line(muv_space, m1, b1)
mu2 = line(muv_space, m2, b2)
mu3 = line(muv_space, m3, b3)
y1 = np.random.normal(mu1, std1, NSAMPLES)
y2 = np.random.normal(mu2, std2, NSAMPLES)
y3 = np.random.normal(mu3, std3, NSAMPLES)
Y = np.vstack((y1, y2, y3))
X = (A @ Y) * xstd + xc

fig, axs = plt.subplots(3, 3, figsize=(10, 15), constrained_layout=True)

axs[0,0].hist2d(muv_space, X[0], bins=(muv_side, lya_side),
                            cmap='Blues_r', cmin=1, rasterized=True)
# axs[0,0].scatter(muv_space, X[0], s=1, color='cyan')
axs[0,0].set_ylabel(r'$\log_{10}L_{\rm Ly\alpha}$ [erg/s]', fontsize=font_size)
axs[0,0].set_xticklabels([])

axs[0,1].hist2d(X[1], X[0], bins=(dv_side, lya_side),
                            cmap='Blues_r', cmin=1, rasterized=True)
# axs[0,1].scatter(X[1], X[0], s=1, color='cyan')
axs[0,1].set_yticklabels([])
axs[0,1].set_xticklabels([])

axs[0,2].hist2d(X[2], X[0], bins=(lha_side, lya_side),
                            cmap='Blues_r', cmin=1, rasterized=True)
# axs[0,2].scatter(X[2], X[0], s=1, color='cyan')
axs[0,2].set_yticklabels([])
axs[0,2].set_xticklabels([])

axs[1,0].hist2d(muv_space, X[1], bins=(muv_side, dv_side),
                            cmap='Blues_r', cmin=1, rasterized=True)
# axs[1,0].scatter(muv_space, X[1], s=1, color='cyan')
axs[1,0].set_ylabel(r'$\Delta v$ [km/s]', fontsize=font_size)
axs[1,0].set_xticklabels([])

axs[1,1].hist2d(X[1], X[1], bins=(dv_side, dv_side),
                            cmap='Blues_r', cmin=1, rasterized=True)
# axs[1,1].scatter(X[1], X[1], s=1, color='cyan')
axs[1,1].set_yticklabels([])
axs[1,1].set_xticklabels([])

axs[1,2].hist2d(X[2], X[1], bins=(lha_side, dv_side),
                            cmap='Blues_r', cmin=1, rasterized=True)
# axs[1,2].scatter(X[2], X[1], s=1, color='cyan')
axs[1,2].set_yticklabels([])
axs[1,2].set_xticklabels([])

axs[2,0].hist2d(muv_space, X[2], bins=(muv_side, lha_side),
                            cmap='Blues_r', cmin=1, rasterized=True)
# axs[2,0].scatter(muv_space, X[2], s=1, color='cyan')
axs[2,0].set_ylabel(r'$\log_{10}L_{\rm H\alpha}$ [erg/s]', fontsize=font_size)
axs[2,0].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)

axs[2,1].hist2d(X[1], X[2], bins=(dv_side, lha_side),
                            cmap='Blues_r', cmin=1, rasterized=True)
# axs[2,1].scatter(X[1], X[2], s=1, color='cyan')
axs[2,1].set_yticklabels([])
axs[2,1].set_xlabel(r'$\Delta v$ [km/s]', fontsize=font_size)

axs[2,2].hist2d(X[2], X[2], bins=(lha_side, lha_side),
                            cmap='Blues_r', cmin=1, rasterized=True)
# axs[2,2].scatter(X[2], X[2], s=1, color='cyan')
axs[2,2].set_yticklabels([])
axs[2,2].set_xlabel(r'$\log_{10}L_{\rm H\alpha}$ [erg/s]', fontsize=font_size)

plt.show()