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
# presentation = True
presentation = False
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

# A = np.load('../data/pca/A.npy')
I = np.array([[1,0,0],[0,1,0],[0,0,1]])
A1 = np.array([[0,0,0],[0,0,-1],[0,1,0]])
A2 = np.array([[0,0,1],[0,0,0],[-1,0,0]])
A3 = np.array([[0,-1,0],[1,0,0],[0,0,0]])
c1, c2, c3, c4 = np.load('../data/pca/coefficients.npy')
A = c1 * I + c2 * A1 + c3 * A2 + c4 * A3
xc = np.load('../data/pca/xc.npy')
xstd = np.load('../data/pca/xstd.npy')
m1, m2, m3, b1, b2, b3, std1, std2, std3, w1, w2, f1, f2, fh = np.load('../data/pca/fit_params.npy')
m1, m2, m3, b1, b2, b3, std1, std2, std3 = -0.05042401, -0.52413956, -0.36336859, \
    -0.91255108, -0.73920641, -0.4175795, 0.85332351, 0.45477191, 0.31269265

NSAMPLES = 100000
muv_space = -21.5 * np.ones(NSAMPLES)

mu1 = line(muv_space, m1, b1)
mu2 = line(muv_space, m2, b2)
mu3 = line(muv_space, m3, b3)
y1 = np.random.normal(mu1, std1, NSAMPLES)
y2 = np.random.normal(mu2, std2, NSAMPLES)
y3 = np.random.normal(mu3, std3, NSAMPLES)
Y = np.vstack((y1, y2, y3))
X = (A @ Y) * xstd + xc


condition = (X[0] > 42.72) * (X[2] > 43.2) * (X[0] < 42.83) * (X[2] < 43.3)
X_cond = X[:, condition]

plt.figure(figsize=(8,6), constrained_layout=True)
plt.hist(X[1], density=True, histtype='stepfilled', alpha=0.5, color='orange')
plt.hist(X_cond[1], density=True, histtype='stepfilled', alpha=0.5, color='blue')
plt.ylabel(r'PDF', fontsize=font_size)
plt.title(r'Conditional PDF at $M_{\mathrm{UV}}=-21.5$, $\log_{10}(L_{\mathrm{H}\alpha})=43.0$', fontsize=font_size)
plt.show()