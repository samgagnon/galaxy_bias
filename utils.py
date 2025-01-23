"""
Useful functions for computing the Lyman-alpha transmission lightcone.

Samuel Gagnon-Hartman 2024
Scuola Normale Superiore, Pisa
"""

import numpy as np

from astropy import units as u
from astropy import constants as c

# lyman alpha parameters
wave_Lya = 1215.67*u.AA
freq_Lya = (c.c.to(u.AA/u.s)/wave_Lya).to('Hz')
omega_Lya = 2*np.pi*freq_Lya
decay_factor = 6.25*1e8*u.s**-1
wavelength_range = np.linspace(1215, 1220, 1000)*u.AA

def delta_nu(T):
    """
    Returns the Doppler width of a Lya line with temperature T
    """
    return ((freq_Lya/c.c) * np.sqrt(2*c.k_B*T/c.m_p.to('kg'))).to('Hz')

def get_a(dnu):
    """
    Returns the damping parameter of a Lya line with Doppler width dnu
    """
    return (decay_factor/(4*np.pi*dnu)).to('')

def voigt_tasitsiomi(x, dnu):
    """
    Returns the Voigt profile of a line with damping parameter a
    """
    dL = 9.936e7*u.Hz
    a = 0.5*(dL/dnu)
    xt = x**2
    z = (xt - 0.855)/(xt + 3.42)
    q = np.zeros(len(x))
    IDX = (z > 0.0)
    q[IDX] = z[IDX]*(1 + 21/xt[IDX])*(a/np.pi/(xt[IDX] + 1.0))*\
        (0.1117 + z[IDX]*(4.421 + z[IDX]*(-9.207 + 5.674*z[IDX])))
    return (q + np.exp(-xt)/1.77245385)*np.sqrt(np.pi)