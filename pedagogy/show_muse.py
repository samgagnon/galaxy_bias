"""
A script to investigate the relationship between MUV and Lya EW

Samuel Gagnon-Hartman
January 2025

Scuola Normale Superiore, Pisa, Italy
"""

import os

import numpy as np

from scipy.special import erf

from astropy.cosmology import Planck18
from astropy import units as u
from astropy.constants import c, G, k_B, m_p

def delta_nu(T):
    """
    Returns the Doppler width of a Lya line with temperature T
    """
    nu_lya = (c/(1215.67*u.Angstrom)).to('Hz')
    return ((nu_lya/c) * np.sqrt(2*k_B*T/m_p.to('kg'))).to('Hz').value

def get_tang_data():
    # TANG LAE data
    # https://arxiv.org/pdf/2402.06070
    MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, fescA, \
        fescA_err, fescB, fescB_err, ID = np.load('../data/tang24.npy').T
    return MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, \
        fescA, fescA_err, fescB, fescB_err, ID

def get_a_mason(m):
    return 0.65 + 0.1 * np.tanh(3 * (m + 20.75))

def get_wc_mason(m):
    return 31 + 12 * np.tanh(4 * (m + 20.25))

# def get_wc_mason(m):
#     return 70 + 40 * np.tanh(4 * (m + 20.25))

def mh_muv_mason2015(muv, redshift):
    """
    Returns the halo mass from the UV magnitude
    using the fit function from Mason et al. 2015.
    """
    gamma = np.zeros_like(muv)
    gamma[muv >= -20 - redshift*0.26] = -0.3
    gamma[muv < -20 - redshift*0.26] = -0.7
    return 10**(gamma*(muv + 20 + 0.26*redshift) + 11.75)

def get_ewpdf_mason2018(Muv):
    """
    Samples EW and emission probability from the
    fit functions obtained by Mason et al. 2018.
    """
    A = get_a_mason(Muv)
    rv_A = np.random.uniform(0, 1, len(Muv))
    emit_bool = rv_A < A
    Wc = get_wc_mason(Muv[emit_bool])
    rv_W = np.random.uniform(0, 1, len(Wc))
    W = -1*Wc*np.log(rv_W)
    return W, emit_bool

def get_dv_mason2018(muv, redshift):
    halo_masses = mh_muv_mason2015(muv, redshift)
    m = 0.32
    c2 = 2.48
    dv_mean = m*(np.log10(halo_masses) - np.log10(1.55) - 12) + c2
    dv_sigma = 0.24
    dv = np.random.normal(dv_mean, dv_sigma, len(halo_masses))
    return 10**dv

def get_beta_bouwens14(muv):
    # https://arxiv.org/pdf/1306.2950
    return -2.05 + -0.2*(muv+19.5)

def get_w_for_flux(muv_range, flux, redshift):
    beta = get_beta_bouwens14(muv_range)
    lum_dens_uv_bkgd = 10**(-0.4*(muv_range - 51.6))
    l_lya_bkgd = 1215.67
    nu_lya_bkgd = (c/(l_lya_bkgd*u.Angstrom)).to('Hz').value
    intensity_bkgd = flux / nu_lya_bkgd
    lum_dens_alpha_bkgd = intensity_bkgd * 4*np.pi*(Planck18.luminosity_distance(redshift).to('cm').value)**2
    w_range = lum_dens_alpha_bkgd * l_lya_bkgd / lum_dens_uv_bkgd / (1215.67/1500)**(beta + 2)
    return w_range

def get_boundary(x, y, z, level=-19):
    X, Y = [], []
    for i, m in enumerate(x):
        if np.sum(z[i] >= level) == 0:
            X.append(m)
            Y.append(np.max(y))
        else:
            X.append(m)
            Y.append(np.min(y[z[i] >= level]))
    return np.array(X), np.array(Y)

def muv_w_completeness(muv_space, w_space, flux_lim, redshift, muv_lim=None):
    w_lim_draw = get_w_for_flux(muv_space, flux_lim, redshift)
    completeness = np.ones((len(muv_space), len(w_space)))
    for i, wl in enumerate(w_lim_draw):
        completeness[i] *= w_space/wl/5
        completeness[i][w_space > wl] = 1
    if muv_lim is not None:
        for i, ml in enumerate(muv_space):
            completeness[i][ml > muv_lim] *= 10**(-0.4*5*(ml - muv_lim))
    return completeness

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc) 
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')
    textcolor = 'white'
    cmap = 'Greys_r'
    datacolor = 'red'
    datacolor2 = 'yellow'
    hist_cmap = 'hot'
    import matplotlib as mpl
    label_size = 20
    font_size = 30
    mpl.rcParams['xtick.labelsize'] = label_size 
    mpl.rcParams['ytick.labelsize'] = label_size 
    points_or_hist = 'points'
    # points_or_hist = 'hist'
    assert points_or_hist in ['points', 'hist']

    ALPHA = 1.0

    os.makedirs('../data/model_hist', exist_ok=True)

    redshift = 5.0
    nu_lya = (c/(1215.67*u.Angstrom)).to('Hz').value

    # measured lya properties from https://arxiv.org/pdf/2402.06070
    MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, \
        fescA, fescA_err, fescB, fescB_err, ID = get_tang_data()


    # get mean relations from Mason 2018
    muv_range = np.linspace(-23, -16, 1000)

    halo_masses = mh_muv_mason2015(muv_range, redshift)
    m = 0.32
    c2 = 2.48
    dv_mean_exp = m*(np.log10(halo_masses) - np.log10(1.55) - 12) + c2
    dv_sigma = 0.24
    dv_mean = 10**(dv_mean_exp)
    dv_up = 10**(dv_mean_exp + dv_sigma)
    dv_low = 10**(dv_mean_exp - dv_sigma)
    wc = get_wc_mason(muv_range)

    v_circ_mean = ((10*G*mh_muv_mason2015(muv_range, 5.0)*u.solMass*Planck18.H(5.0))**(1/3)).to('km/s').value
    fesc_model = 1 - 0.5*(1 + erf((v_circ_mean - dv_mean)/(np.sqrt(2)*dv_mean)))
    fesc_up = 1 - 0.5*(1 + erf((v_circ_mean - dv_up)/(np.sqrt(2)*dv_up)))
    fesc_low = 1 - 0.5*(1 + erf((v_circ_mean - dv_low)/(np.sqrt(2)*dv_low)))


    fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # Muv vs EW
    axs[0].errorbar(MUV[ID==0], ew_lya[ID==0], xerr=MUV_err[ID==0], yerr=ew_lya_err[ID==0],
        fmt='o', markersize=10, capsize=5, color=datacolor, alpha=ALPHA, label='TANG LAE data')
    axs[0].errorbar(MUV[ID==1], ew_lya[ID==1], xerr=MUV_err[ID==1], yerr=ew_lya_err[ID==1],
        fmt='o', markersize=10, capsize=5, color=datacolor2, alpha=ALPHA, label='TANG LAE data')
    axs[0].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
    axs[0].set_ylabel(r'${\rm W}_{\rm emerg}$ [$\AA$]', fontsize=font_size)

    # axs[0].fill_between(muv_range, wc*0.5, wc*2, color='orange', alpha=0.5, zorder=1e2)
    # axs[0].plot(muv_range, wc, color='orange', linestyle='-', label='classic model', zorder=1e3)


    # W vs dv
    axs[1].errorbar(MUV[ID==0], dv_lya[ID==0], xerr=MUV_err[ID==0], yerr=dv_lya_err[ID==0],
        fmt='o', markersize=10, capsize=5, color=datacolor, alpha=ALPHA)
    axs[1].errorbar(MUV[ID==1], dv_lya[ID==1], xerr=MUV_err[ID==1], yerr=dv_lya_err[ID==1],
        fmt='o', markersize=10, capsize=5, color=datacolor2, alpha=ALPHA)
    axs[1].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
    axs[1].set_ylabel(r'$\Delta v$ [km s$^{-1}$]', fontsize=font_size)

    # axs[1].fill_between(muv_range, dv_low, dv_up, color='orange', alpha=0.5, zorder=1e2)
    # axs[1].plot(muv_range, dv_mean, color='orange', linestyle='-', label='classic model', zorder=1e3)
    # axs[1].plot(muv_range, v_circ_mean, color='orange', linestyle='--', label='classic model', zorder=1e3)

    # dv vs fesc
    axs[2].errorbar(dv_lya[ID==0], fescB[ID==0], xerr=dv_lya_err[ID==0], yerr=fescB_err[ID==0],
        fmt='o', markersize=10, capsize=5, color=datacolor, alpha=ALPHA)
    axs[2].errorbar(dv_lya[ID==1], fescB[ID==1], xerr=dv_lya_err[ID==1], yerr=fescB_err[ID==1],
        fmt='o', markersize=10, capsize=5, color=datacolor2, alpha=ALPHA)
    axs[2].set_xlabel(r'$\Delta v$ [km s$^{-1}$]', fontsize=font_size)
    axs[2].set_ylabel(r'$f_{\rm esc, B}$', fontsize=font_size)

    # axs[2].fill_between(dv_mean, fesc_low, fesc_up, color='orange', alpha=0.5, zorder=1e2)
    # axs[2].plot(dv_mean, fesc_model, color='orange', linestyle='--', label='classic model', zorder=1e3)

    axs[0].set_xlim(-22, -16)
    axs[1].set_xlim(-22, -16)
    axs[2].set_ylim(0, 2.6)
    axs[1].set_ylim(0, 810)

    plt.show()
