"""
A script to investigate the relationship between MUV and Lya EW

Samuel Gagnon-Hartman
January 2025

Scuola Normale Superiore, Pisa, Italy
"""

import numpy as np
from data import get_tang_data, get_magpi_data, get_vandels_data

def Wc(MUV):
    return 31 + 12 * np.tanh(4 * (MUV + 20.25))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc) 
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')

    MUV, MUV_err, z, ew_lya, ew_lya_err = get_tang_data()
    z_MAGPI, MUV_MAGPI, EW_MAGPI, EW_MAGPI_sigma = get_magpi_data()
    z_vandels, f_lya, f_lya_err, ew_lya_vandels, ew_lya_vandels_err = get_vandels_data()

    from astropy.cosmology import Planck18
    from astropy import units as u

    # convert flux to luminosity
    f_lya *= u.mW/u.m**2
    f_lya_err *= u.mW/u.m**2
    L_lya = 4*np.pi*(Planck18.luminosity_distance(z_vandels).to('cm'))**2*f_lya*10
    L_lya_err = 4*np.pi*(Planck18.luminosity_distance(z_vandels).to('cm'))**2*f_lya_err*10

    L_lya = L_lya.to(u.erg/u.s).value
    L_lya_err = L_lya_err.to(u.erg/u.s).value
    constant = 2.47 * 1e15 / 1216 / (
                        1500 / 1216) ** (-2 - 2)
    # these values are a bit high... let's see how they cluster with the others
    MUV_vandels = -2.5*np.log10(L_lya/constant) + 51.6
    # not sure if this is the right error
    MUV_vandels_err = np.abs(2.5/np.log(10)*L_lya_err/(L_lya*np.log(10)))

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), sharey=True)

    muv_range = np.linspace(-24, -16, 100)
    wc = Wc(muv_range)

    ax.errorbar(MUV_vandels[(z_vandels>5)*(ew_lya_vandels>=25)], ew_lya_vandels[(z_vandels>5)*(ew_lya_vandels>=25)], \
                xerr=MUV_vandels_err[(z_vandels>5)*(ew_lya_vandels>=25)], \
                yerr=np.abs(ew_lya_vandels_err)[(z_vandels>5)*(ew_lya_vandels>=25)], fmt='.', color='lime', label='VANDELS')
    ax.errorbar(MUV, ew_lya, yerr=ew_lya_err, xerr=MUV_err, fmt='o', color='white', label='JADES/FRESCO/MUSE')
    ax.errorbar(MUV_MAGPI[MUV_MAGPI<0], EW_MAGPI[MUV_MAGPI<0], yerr=EW_MAGPI_sigma[MUV_MAGPI<0], fmt='o', color='red', label='MAGPI')

    ax.plot(muv_range, wc, color='cyan', linestyle='--', label=r'$W_c$, Mason+18')

    ax.set_xlabel(r'$M_{\rm UV}$')
    ax.set_xlim(-24, -16)
    ax.set_yscale('log')

    # ax[1].errorbar(z_vandels, ew_lya_vandels, yerr=np.abs(ew_lya_vandels_err), fmt='o', color='lime', label='VANDELS')
    # ax[1].errorbar(z, ew_lya, yerr=ew_lya_err, xerr=MUV_err, fmt='o', color='white', label='JADES/FRESCO/MUSE')
    # ax[1].errorbar(z_MAGPI, EW_MAGPI, yerr=EW_MAGPI_sigma, fmt='o', color='red', label='MAGPI')
    # ax[1].set_xlabel(r'$z$')

    ax.set_ylabel(r'EW$_{\rm Ly\alpha}$')
    ax.legend()
    plt.show()

