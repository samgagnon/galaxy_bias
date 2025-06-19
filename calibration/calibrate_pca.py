import os

import numpy as np
import py21cmfast as p21c

from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, k_B, m_p, e, m_e

from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from scipy.special import gamma, erf
from scipy.optimize import minimize, curve_fit

from sklearn.decomposition import PCA

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

A = np.load('../data/pca/transformation.npy')
XSTD = np.load('../data/pca/xstd.npy')
XC = np.load('../data/pca/xc.npy')

print(np.linalg.inv(A))
print(XSTD)
print(XC)
# x = (np.linalg.inv(A) @ y) * XSTD + XC