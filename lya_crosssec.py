"""
Calculates the cross-sectional profile of Lyman alpha as 
a function of frequency. Based on Peebles 1993(?)

Samuel Gagnon-Hartman
Scuola Normale Superiore
October 2024
"""


import os

import numpy as np
from scipy import integrate

from astropy import units as u
from astropy import constants as c
from astropy.cosmology import z_at_value
from astropy.cosmology import Planck18

freq_Lya = 2.47*1e15*u.Hz
omega_Lya = 2*np.pi*freq_Lya
wave_Lya = c.c.to(u.AA/u.s)/freq_Lya
decay_factor = 6.25*1e8*u.s**-1

def sigma_Lya(omega):
    """
    Returns the cross section of Lyman alpha photons with neutral hydrogen
    Calculated, e.g., by Peebles 1993 section 23
    """
    # prefactor comes from the normalization of the cross section
    prefactor = 3*wave_Lya**2*decay_factor**2/(8*np.pi)
    # profile shape comes from quantum mechanics
    sigma_lya = (omega/omega_Lya)**4 / \
        ((decay_factor**2/4)*(omega/omega_Lya)**6 + (omega - omega_Lya)**2)
    return (prefactor*sigma_lya).to('cm2').value

def doppler_width(nu, T):
    """
    Returns the emission spectrum of a Lya line with temperature T
    """
    delta_v = (freq_Lya/c.c) * np.sqrt(2*c.k_B*T*u.K/c.m_p.to('kg'))
    spectrum = np.exp(-1*(nu-freq_Lya)**2/(delta_v**2))/(delta_v*np.pi**0.5)
    return spectrum.to('s').value

def lya_crosssec1(freq, T=None):
    if T is not None:
        sigma = np.zeros(len(freq))
        doppler = np.zeros(len(freq))
        for i, nu in enumerate(freq):
            sigma[i] = sigma_Lya(nu*(2*np.pi))
            doppler[i] = doppler_width(nu, T)
        total_photon = integrate.simps(sigma, freq)
        doppler /= integrate.simps(doppler, freq)
        doppler *= total_photon
        voigt_profile = np.convolve(sigma, doppler, mode='same')
        voigt_profile /= integrate.simps(voigt_profile, freq)
        voigt_profile *= total_photon
        return voigt_profile
    else:
        sigma = np.zeros(len(freq))
        for i, nu in enumerate(freq):
            sigma[i] = sigma_Lya(nu*(2*np.pi))
        return sigma

def voigt(x, a):
    """
    Returns the Voigt profile of a line with damping parameter a
    """
    # wtf is going on in here?
    k = -16
    cvoigt = np.zeros(30)
    for i in range(30):
        k += i
        cvoigt[i] = 0.0897935610625833 * np.exp(-k*k/9.0)
    
    q1 = 9.42477796076938
    q2 = 0.564189583547756

    if a < 0.01:
        return q2 * np.exp(-x*x)
    
    a1 = 3*a
    a2 = a*a
    e = np.exp(-1*q1*a)
    if a < 0.1:
        zr = 0.5 * (e + 1.0/e) * np.cos(q1*x)
        zi = 0.5 * (e - 1.0/e) * np.sin(q1*x)
        vg1 = q2 * np.exp(a2 - x*x) * np.cos(2*a*x)
    else:
        zr = e * np.cos(a*x)
        zi = e * np.sin(a*x)
        vg1 = 0.0
    
    b1 = (1.0 - zr) * a * 1.5
    b2 = -1*zi
    s = -8 -1.5 * x
    t = s*s + 2.25 * a2
    for i in range(30):
        t += s + 0.25
        s += 0.5
        b1 = a1 - b1
        b2 *= -1
        if t > 2.5e-12:
            vg1 += cvoigt[i] * (b1 + b2*s)/t
        else:
            vg1 -= cvoigt[i] * a * 29.608813203268
    return vg1

def lya_crosssec(nu, T):
    """
    Returns the cross section of Lyman alpha photons with neutral hydrogen
    Calculated, e.g., by Peebles 1993 section 23
    """
    dopwidth = 1056924075.0 * np.sqrt(T)
    avoigt = 49854330.99 / dopwidth
    x = (nu - freq_Lya.value) / dopwidth
    return 0.011046 * voigt(x, avoigt)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})

    from scipy.special import voigt_profile

    # min_freq = 2.41
    # max_freq = 2.49
    min_freq = 2.469
    max_freq = 2.471
    freq = np.linspace(min_freq, max_freq, 1000)*1e15*u.Hz
    wavelength = c.c.to('cm/s')/freq
    diff_crosssec = lya_crosssec(freq.value, 1e4)
    diff_crosssec1 = lya_crosssec1(freq, 1e4)
    x = np.linspace(-10, 10, 100)
    voigt1 = voigt(x, 10)
    voigt2 = voigt_profile(x, 10, 10)
    plt.plot(x, voigt1, color='red')
    plt.plot(x, voigt2, color='blue')
    plt.show()
    quit()
    # crosssec = integrate.cumulative_trapezoid(diff_crosssec, wavelength.value)
    plt.plot(wavelength.to('AA'), diff_crosssec, color='black')
    plt.plot(wavelength.to('AA'), diff_crosssec1, color='blue')
    plt.yscale('log')
    plt.show()
    quit()
    # colors = ['black', 'blue', 'green']
    # for i, T in enumerate([1e3, 1e4, 1e5]):
    #     sigma = np.zeros(len(freq))
    #     for j, nu in enumerate(freq):
    #         sigma[j] = lya_crosssec(nu.value, T)
    #     plt.plot(freq, sigma, label=f"T={T} K", color=colors[i])
    
    sigma = np.zeros(len(freq))
    doppler = np.zeros(len(freq))
    for i, nu in enumerate(freq):
        sigma[i] = sigma_Lya(nu*(2*np.pi))
        doppler[i] = doppler_width(nu, 100)
    total_photon = integrate.simps(sigma, freq)
    doppler /= integrate.simps(doppler, freq)
    doppler *= total_photon
    voigt_profile = np.convolve(sigma, doppler, mode='same')
    voigt_profile /= integrate.simps(voigt_profile, freq)
    voigt_profile *= total_photon

    plt.plot(freq, sigma, color='red')
    plt.plot(freq, doppler, color='blue')
    plt.plot(freq, voigt_profile, color='green')
    plt.xlabel(r"$\nu$ [Hz]")
    plt.ylabel(r"$\sigma_{\rm Ly\alpha}$ [cm$^2$]")
    plt.ylim(sigma.min(), sigma.max())
    plt.xlim(min_freq*1e15, max_freq*1e15)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend([r'Peebles 1993$\S$23', r'Doppler: $T=10^4$ K', r'Voigt: $T=10^4$ K'])
    plt.show()