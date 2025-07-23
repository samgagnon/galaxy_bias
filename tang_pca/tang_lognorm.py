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
    color1 = 'black'
    color2 = 'blue'
    color3 = 'red'
    color4 = 'orange'
    textcolor = 'black'

