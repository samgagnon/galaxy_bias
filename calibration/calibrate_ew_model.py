"""
A script to investigate the relationship between MUV and Lya EW

Samuel Gagnon-Hartman
January 2025

Scuola Normale Superiore, Pisa, Italy
"""

import numpy as np

from scipy.special import erf

from astropy.cosmology import Planck18
from astropy import units as u
from astropy.constants import c, G

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
    w_lim = get_w_for_flux(muv_space, flux_lim, redshift)
    completeness = np.ones((len(muv_space), len(w_space)))
    for i, wl in enumerate(w_lim):
        completeness[i] *= w_space/wl/5
        completeness[i][w_space > wl] = 1
    if muv_lim is not None:
        for i, ml in enumerate(muv_space):
            completeness[i][ml > muv_lim] *= 10**(-0.4*5*(ml - muv_lim))
    return completeness

def get_lya_properties(halo_masses, muv, redshift, mode='mason'):
    if mode == 'mason':
        v_circ = ((10*G*halo_masses*u.solMass*Planck18.H(redshift))**(1/3)).to('km/s').value

        w, emit_bool = get_ewpdf_mason2018(muv)
        muv = muv[emit_bool]
        dv = get_dv_mason2018(muv, redshift)
        sigma_v = dv / 1
        halo_masses = halo_masses[emit_bool]

        fesc = 0.5*(1 - erf((v_circ[emit_bool] - dv)/(np.sqrt(2)*(sigma_v))))
        # w *= cgm_transmission_fraction

        beta = get_beta_bouwens14(muv)
        lum_dens_uv = 10**(-0.4*(muv - 51.6))
        l_lya = 1215.67 #* (1 + redshift)
        lum_dens_alpha = (w / l_lya) * lum_dens_uv * (1215.67/1500)**(beta + 2)
        intensity = lum_dens_alpha/(4*np.pi*(Planck18.luminosity_distance(redshift).to('cm').value)**2)
        mab = -2.5 * np.log10(intensity) - 48.6

    elif mode == 'gaussian':

        # gaussian parameters
        GAUS_PARAMS = (0, 500, 100, 100)
        # correlation coefficients
        RHO = (0.0, 0.0, 0.0, 0.0) # muv-w, w-dv, w-fesc, dv-fesc
        # RHO = (0.5, 0.5, 0.5, 0.5) # muv-w, w-dv, w-fesc, dv-fesc
        # RHO = (0.5, 0.5, 0.5, 0.5) # muv-w, w-dv, w-fesc, dv-fesc

        muv_mean = np.mean(muv)
        muv_std = np.std(muv)

        v_circ = ((10*G*halo_masses*u.solMass*Planck18.H(redshift))**(1/3)).to('km/s').value
        # perturb v_circ
        # v_circ *= np.random.normal(1.0, 0.5, len(halo_masses))
        # sample W from a gaussian
        w = np.random.normal(GAUS_PARAMS[0], GAUS_PARAMS[1], len(halo_masses))
        # introduce correlation with MUV
        muv_w = (muv - muv_mean)*GAUS_PARAMS[1]/muv_std + GAUS_PARAMS[0]
        w = RHO[0]*muv_w + (1 - RHO[0])*w
        # sample dv from a gaussian
        dv = np.random.normal(GAUS_PARAMS[2], GAUS_PARAMS[3], len(halo_masses))
        # introduce correlation with W
        # w_dv = (w - GAUS_PARAMS[0])*GAUS_PARAMS[3]/GAUS_PARAMS[1] + GAUS_PARAMS[2]
        # dv = RHO[1]*w_dv + (1 - RHO[1])*dv
        # sigma_v = dv / 3
        # introduce correlation with v_circ
        v_circ_mean = np.mean(v_circ)
        v_circ_std = np.std(v_circ)
        dv_vcirc = (v_circ - v_circ_mean)*GAUS_PARAMS[3]/v_circ_std + GAUS_PARAMS[2]
        dv = 0.5*dv_vcirc + (1 - 0.5)*dv
        sigma_v = dv / 3

        # blueward attenuation by CGM
        fesc = 0.5*(1 - erf((v_circ - dv)/(np.sqrt(2)*(sigma_v))))
        # introduce correlation with dv
        fesc_mean = np.mean(fesc)
        fesc_std = np.std(fesc)
        dv_fesc = (dv - GAUS_PARAMS[2])*fesc_std/GAUS_PARAMS[1] + fesc_mean
        fesc = RHO[3]*dv_fesc + (1 - RHO[3])*fesc

        # fesc *= np.random.uniform(0, 1, len(halo_masses))
        w *= fesc

        # adjust fesc to account for energy in blue peak
        # blue_coeff = np.random.uniform(0, 0.5, len(w))
        # fesc *= 1 - blue_coeff

        beta = get_beta_bouwens14(muv)
        lum_dens_uv = 10**(-0.4*(muv - 51.6))
        l_lya = 1215.67 #* (1 + redshift)
        lum_dens_alpha = (w / l_lya) * lum_dens_uv * (1215.67/1500)**(beta + 2)
        intensity = lum_dens_alpha/(4*np.pi*(Planck18.luminosity_distance(redshift).to('cm').value)**2)
        mab = -2.5 * np.log10(intensity) - 48.6

    return halo_masses, muv, mab, dv, w, fesc

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc) 
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')

    redshift = 5.0
    nu_lya = (c/(1215.67*u.Angstrom)).to('Hz').value

    _, _, _, halo_masses, _, sfr = np.load(f'../data/halo_field_{redshift}.npy')

    muv = 51.64 - np.log10(sfr*3.1557e7/1.15e-28) / 0.4

    halo_masses = halo_masses[muv<=-16]
    muv = muv[muv<=-16]

    # simulated lya properties
    halo_masses, muv, mab, dv, w, fesc = get_lya_properties(halo_masses, muv, redshift, mode='mason')
    # halo_masses, muv, mab, dv, w, fesc = get_lya_properties(halo_masses, muv, redshift, mode='gaussian')

    # measured lya properties from https://arxiv.org/pdf/2402.06070
    MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, \
        fescA, fescA_err, fescB, fescB_err, ID = get_tang_data()

    muv_space = np.linspace(-24, -16, 100)
    dv_space = np.linspace(0, 1000, 100)
    dv_logspace = 10**np.linspace(0, 3, 100)
    w_space = 10**np.linspace(0, 3, 100)
    fesc_space = np.linspace(0, 1, 100)

    _mab = np.copy(mab)
    _halo_masses = np.copy(halo_masses)
    _dv = np.copy(dv)
    _w = np.copy(w)
    _muv = np.copy(muv)
    _fesc = np.copy(fesc)

    fig, axs = plt.subplots(3, 5, figsize=(18, 10), constrained_layout=True)

    # ROW ONE: MUSE-Wide

    # apply flux completeness limit for each sample
    flux_limit_draw = np.random.normal(loc=2e-17, scale=2e-17/5, size=len(_muv))
    flux_limit_draw[flux_limit_draw > 2e-17] = 2e-17
    mab_lim = -2.5 * np.log10(flux_limit_draw/nu_lya) - 48.6

    # muv_limit_draw = -2.5*np.log10(np.random.normal(loc=10**(-0.4*(-18)), \
    #                             scale=10**(-0.4*(-18))/5, size=len(muv)))
    muv_limit_draw = -15

    halo_masses = _halo_masses[(_mab<mab_lim)*(_muv<muv_limit_draw)]
    dv = _dv[(_mab<mab_lim)*(_muv<muv_limit_draw)]
    fesc = _fesc[(_mab<mab_lim)*(_muv<muv_limit_draw)]
    w = _w[(_mab<mab_lim)*(_muv<muv_limit_draw)]
    muv = _muv[(_mab<mab_lim)*(_muv<muv_limit_draw)]

    # create 2d histogram of muv, dv
    h_mdv, x_mdv, y_mdv = np.histogram2d(muv, dv, bins=(muv_space, dv_space), \
                                       density=True)
    h_mw, x_mw, y_mw = np.histogram2d(muv, w, bins=(muv_space, w_space), \
                                       density=True)
    h_wdv, x_wdv, y_wdv = np.histogram2d(w, dv, bins=(w_space, dv_space), \
                                       density=True)
    h_dvfesc, x_dvfesc, y_dvfesc = np.histogram2d(dv, fesc, bins=(dv_logspace, fesc_space), \
                                       density=True)
    h_wfesc, x_wfesc, y_wfesc = np.histogram2d(w, fesc, bins=(w_space, fesc_space), \
                                        density=True)

    axs[0,0].pcolormesh(x_mdv, y_mdv, np.log10(h_mdv.T), cmap='Greys_r', rasterized=True)
    axs[0,0].errorbar(MUV[ID==0], dv_lya[ID==0], xerr=MUV_err[ID==0], yerr=dv_lya_err[ID==0], fmt='o', color='red')
    # axs[0].plot(muv_space, dv_mason_mean, color='blue', linestyle='--')

    axs[0,1].pcolormesh(x_mw, y_mw, np.log10(h_mw.T), cmap='Greys_r', rasterized=True)
    axs[0,1].errorbar(MUV[ID==0], ew_lya[ID==0], xerr=MUV_err[ID==0], yerr=ew_lya_err[ID==0], fmt='o', color='red')

    # axs[1].plot(muv_space, ew_mason_mean, color='blue', linestyle='--')

    # axs[2].scatter(w, dv, color='white', alpha=0.1, rasterized=True)
    # 0: MUSE Wide, 1: MUSE Deep, 2: DEIMOS
    axs[0,2].pcolormesh(x_wdv, y_wdv, np.log10(h_wdv.T), cmap='Greys_r', rasterized=True)
    axs[0,2].errorbar(ew_lya[ID==0], dv_lya[ID==0], xerr=ew_lya_err[ID==0], yerr=dv_lya_err[ID==0], fmt='o', color='red')

    axs[0,3].pcolormesh(x_dvfesc, y_dvfesc, np.log10(h_dvfesc.T), cmap='Greys_r', rasterized=True)
    # axs[0,3].errorbar(dv_lya[ID==0], fescA[ID==0], xerr=dv_lya_err[ID==0], yerr=fescA_err[ID==0], fmt='o', color='red')
    axs[0,3].errorbar(dv_lya[ID==0], fescB[ID==0], xerr=dv_lya_err[ID==0], yerr=fescB_err[ID==0], fmt='x', color='red')

    axs[0,4].pcolormesh(x_wfesc, y_wfesc, np.log10(h_wfesc.T), cmap='Greys_r', rasterized=True)
    # axs[0,4].errorbar(ew_lya[ID==0], fescA[ID==0], xerr=ew_lya_err[ID==0], yerr=fescA_err[ID==0], fmt='o', color='red')
    axs[0,4].errorbar(ew_lya[ID==0], fescB[ID==0], xerr=ew_lya_err[ID==0], yerr=fescB_err[ID==0], fmt='x', color='red')

    # SECOND ROW: MUSE-Deep

    # apply flux completeness limit for each sample
    flux_limit_draw = np.random.normal(loc=2e-18, scale=2e-18/5, size=len(_muv))
    flux_limit_draw[flux_limit_draw > 2e-18] = 2e-18
    mab_lim = -2.5 * np.log10(flux_limit_draw/nu_lya) - 48.6

    # muv_limit_draw = -2.5*np.log10(np.random.normal(loc=10**(-0.4*(-18)), \
    #                             scale=10**(-0.4*(-18))/5, size=len(muv)))
    muv_limit_draw = -15

    halo_masses = _halo_masses[(_mab<mab_lim)*(_muv<muv_limit_draw)]
    dv = _dv[(_mab<mab_lim)*(_muv<muv_limit_draw)]
    fesc = _fesc[(_mab<mab_lim)*(_muv<muv_limit_draw)]
    w = _w[(_mab<mab_lim)*(_muv<muv_limit_draw)]
    muv = _muv[(_mab<mab_lim)*(_muv<muv_limit_draw)]

    # create 2d histogram of muv, dv
    h_mdv, x_mdv, y_mdv = np.histogram2d(muv, dv, bins=(muv_space, dv_space), \
                                       density=True)
    h_mw, x_mw, y_mw = np.histogram2d(muv, w, bins=(muv_space, w_space), \
                                       density=True)
    h_wdv, x_wdv, y_wdv = np.histogram2d(w, dv, bins=(w_space, dv_space), \
                                       density=True)
    h_dvfesc, x_dvfesc, y_dvfesc = np.histogram2d(dv, fesc, bins=(dv_logspace, fesc_space), \
                                       density=True)
    h_wfesc, x_wfesc, y_wfesc = np.histogram2d(w, fesc, bins=(w_space, fesc_space), \
                                        density=True)
    
    axs[1,0].pcolormesh(x_mdv, y_mdv, np.log10(h_mdv.T), cmap='Greys_r', rasterized=True)
    axs[1,0].errorbar(MUV[ID==1], dv_lya[ID==1], xerr=MUV_err[ID==1], yerr=dv_lya_err[ID==1], fmt='o', color='red')

    axs[1,1].pcolormesh(x_mw, y_mw, np.log10(h_mw.T), cmap='Greys_r', rasterized=True)
    axs[1,1].errorbar(MUV[ID==1], ew_lya[ID==1], xerr=MUV_err[ID==1], yerr=ew_lya_err[ID==1], fmt='o', color='red')
    
    axs[1,2].pcolormesh(x_wdv, y_wdv, np.log10(h_wdv.T), cmap='Greys_r', rasterized=True)
    axs[1,2].errorbar(ew_lya[ID==1], dv_lya[ID==1], xerr=ew_lya_err[ID==1], yerr=dv_lya_err[ID==1], fmt='o', color='red')

    axs[1,3].pcolormesh(x_dvfesc, y_dvfesc, np.log10(h_dvfesc.T), cmap='Greys_r', rasterized=True)
    # axs[1,3].errorbar(dv_lya[ID==1], fescA[ID==1], xerr=dv_lya_err[ID==1], yerr=fescA_err[ID==1], fmt='o', color='red')
    axs[1,3].errorbar(dv_lya[ID==1], fescB[ID==1], xerr=dv_lya_err[ID==1], yerr=fescB_err[ID==1], fmt='x', color='red')

    axs[1,4].pcolormesh(x_wfesc, y_wfesc, np.log10(h_wfesc.T), cmap='Greys_r', rasterized=True)
    # axs[1,4].errorbar(ew_lya[ID==1], fescA[ID==1], xerr=ew_lya_err[ID==1], yerr=fescA_err[ID==1], fmt='o', color='red')
    axs[1,4].errorbar(ew_lya[ID==1], fescB[ID==1], xerr=ew_lya_err[ID==1], yerr=fescB_err[ID==1], fmt='x', color='red')

    # THIRD ROW: DEIMOS

    # apply flux completeness limit for each sample
    flux_limit_draw = np.random.normal(loc=2e-18, scale=2e-18/5, size=len(_muv))
    flux_limit_draw[flux_limit_draw > 2e-18] = 2e-18
    mab_lim = -2.5 * np.log10(flux_limit_draw/nu_lya) - 48.6

    limit_b = -18.2 - Planck18.distmod(5.0).value + Planck18.distmod(3.8).value
    limit_v = -18.8 - Planck18.distmod(5.0).value + Planck18.distmod(5.0).value
    limit_i = -19.1 - Planck18.distmod(5.0).value + Planck18.distmod(5.9).value
    limit_mean = np.mean([limit_b, limit_v, limit_i])

    muv_limit_draw = -2.5*np.log10(np.random.normal(loc=10**(-0.4*limit_mean), \
                                scale=10**(-0.4*limit_mean)/5, size=len(_muv)))

    halo_masses = _halo_masses[(_mab<mab_lim)*(_muv<muv_limit_draw)]
    dv = _dv[(_mab<mab_lim)*(_muv<muv_limit_draw)]
    fesc = _fesc[(_mab<mab_lim)*(_muv<muv_limit_draw)]
    w = _w[(_mab<mab_lim)*(_muv<muv_limit_draw)]
    muv = _muv[(_mab<mab_lim)*(_muv<muv_limit_draw)]

    # create 2d histogram of muv, dv
    h_mdv, x_mdv, y_mdv = np.histogram2d(muv, dv, bins=(muv_space, dv_space), \
                                       density=True)
    h_mw, x_mw, y_mw = np.histogram2d(muv, w, bins=(muv_space, w_space), \
                                       density=True)
    h_wdv, x_wdv, y_wdv = np.histogram2d(w, dv, bins=(w_space, dv_space), \
                                       density=True)
    h_dvfesc, x_dvfesc, y_dvfesc = np.histogram2d(dv, fesc, bins=(dv_logspace, fesc_space), \
                                       density=True)
    h_wfesc, x_wfesc, y_wfesc = np.histogram2d(w, fesc, bins=(w_space, fesc_space), \
                                        density=True)

    axs[2,0].pcolormesh(x_mdv, y_mdv, np.log10(h_mdv.T), cmap='Greys_r', rasterized=True)
    axs[2,0].errorbar(MUV[ID==2], dv_lya[ID==2], xerr=MUV_err[ID==2], yerr=dv_lya_err[ID==2], fmt='o', color='red')
    axs[2,0].errorbar(MUV[ID==3], dv_lya[ID==3], xerr=MUV_err[ID==3], yerr=dv_lya_err[ID==3], fmt='x', color='red')
    axs[2,0].errorbar(MUV[ID==4], dv_lya[ID==4], xerr=MUV_err[ID==4], yerr=dv_lya_err[ID==4], fmt='^', color='red')

    axs[2,1].pcolormesh(x_mw, y_mw, np.log10(h_mw.T), cmap='Greys_r', rasterized=True)
    axs[2,1].errorbar(MUV[ID==2], ew_lya[ID==2], xerr=MUV_err[ID==2], yerr=ew_lya_err[ID==2], fmt='o', color='red', label='DEIMOS-V')
    axs[2,1].errorbar(MUV[ID==3], ew_lya[ID==3], xerr=MUV_err[ID==3], yerr=ew_lya_err[ID==3], fmt='x', color='red', label='DEIMOS-i')
    axs[2,1].errorbar(MUV[ID==4], ew_lya[ID==4], xerr=MUV_err[ID==4], yerr=ew_lya_err[ID==4], fmt='^', color='red', label='DEIMOS-B')

    axs[2,2].pcolormesh(x_wdv, y_wdv, np.log10(h_wdv.T), cmap='Greys_r', rasterized=True)
    axs[2,2].errorbar(ew_lya[ID==2], dv_lya[ID==2], xerr=ew_lya_err[ID==2], yerr=dv_lya_err[ID==2], fmt='o', color='red')
    axs[2,2].errorbar(ew_lya[ID==3], dv_lya[ID==3], xerr=ew_lya_err[ID==3], yerr=dv_lya_err[ID==3], fmt='x', color='red')
    axs[2,2].errorbar(ew_lya[ID==4], dv_lya[ID==4], xerr=ew_lya_err[ID==4], yerr=dv_lya_err[ID==4], fmt='^', color='red')

    axs[2,3].pcolormesh(x_dvfesc, y_dvfesc, np.log10(h_dvfesc.T), cmap='Greys_r', rasterized=True)
    # axs[2,3].errorbar(dv_lya[ID==2], fescA[ID==2], xerr=dv_lya_err[ID==2], yerr=fescA_err[ID==2], fmt='o', color='red')
    axs[2,3].errorbar(dv_lya[ID==2], fescB[ID==2], xerr=dv_lya_err[ID==2], yerr=fescB_err[ID==2], fmt='x', color='red')
    # axs[2,3].errorbar(dv_lya[ID==3], fescA[ID==3], xerr=dv_lya_err[ID==3], yerr=fescA_err[ID==3], fmt='x', color='red')
    axs[2,3].errorbar(dv_lya[ID==3], fescB[ID==3], xerr=dv_lya_err[ID==3], yerr=fescB_err[ID==3], fmt='x', color='red')
    # axs[2,3].errorbar(dv_lya[ID==4], fescA[ID==4], xerr=dv_lya_err[ID==4], yerr=fescA_err[ID==4], fmt='^', color='red')
    axs[2,3].errorbar(dv_lya[ID==4], fescB[ID==4], xerr=dv_lya_err[ID==4], yerr=fescB_err[ID==4], fmt='^', color='red')

    axs[2,4].pcolormesh(x_wfesc, y_wfesc, np.log10(h_wfesc.T), cmap='Greys_r', rasterized=True)
    # axs[2,4].errorbar(ew_lya[ID==2], fescA[ID==2], xerr=ew_lya_err[ID==2], yerr=fescA_err[ID==2], fmt='o', color='red')
    axs[2,4].errorbar(ew_lya[ID==2], fescB[ID==2], xerr=ew_lya_err[ID==2], yerr=fescB_err[ID==2], fmt='x', color='red')
    # axs[2,4].errorbar(ew_lya[ID==3], fescA[ID==3], xerr=ew_lya_err[ID==3], yerr=fescA_err[ID==3], fmt='x', color='red')
    axs[2,4].errorbar(ew_lya[ID==3], fescB[ID==3], xerr=ew_lya_err[ID==3], yerr=fescB_err[ID==3], fmt='x', color='red')
    # axs[2,4].errorbar(ew_lya[ID==4], fescA[ID==4], xerr=ew_lya_err[ID==4], yerr=fescA_err[ID==4], fmt='^', color='red')
    axs[2,4].errorbar(ew_lya[ID==4], fescB[ID==4], xerr=ew_lya_err[ID==4], yerr=fescB_err[ID==4], fmt='x', color='red')

    axs[0,0].text(-23.5, 800, 'MUSE-Wide', color='white')
    axs[1,0].text(-23.5, 800, 'MUSE-Deep', color='white')
    axs[2,0].text(-23.5, 800, 'DEIMOS', color='white')

    axs[0,0].set_xlabel(r'$M_{\rm UV}$')
    axs[0,0].set_ylabel(r'$\Delta v_{\rm Ly\alpha}$')
    axs[1,0].set_xlabel(r'$M_{\rm UV}$')
    axs[1,0].set_ylabel(r'$\Delta v_{\rm Ly\alpha}$')
    axs[2,0].set_xlabel(r'$M_{\rm UV}$')
    axs[2,0].set_ylabel(r'$\Delta v_{\rm Ly\alpha}$')

    axs[0,1].set_xlabel(r'$M_{\rm UV}$')
    axs[0,1].set_ylabel(r'EW$_{\rm Ly\alpha}$')
    axs[1,1].set_xlabel(r'$M_{\rm UV}$')
    axs[1,1].set_ylabel(r'EW$_{\rm Ly\alpha}$')
    axs[2,1].set_xlabel(r'$M_{\rm UV}$')
    axs[2,1].set_ylabel(r'EW$_{\rm Ly\alpha}$')

    axs[0,2].set_xlabel(r'EW$_{\rm Ly\alpha}$')
    axs[0,2].set_ylabel(r'$\Delta v_{\rm Ly\alpha}$')
    axs[1,2].set_xlabel(r'EW$_{\rm Ly\alpha}$')
    axs[1,2].set_ylabel(r'$\Delta v_{\rm Ly\alpha}$')
    axs[2,2].set_xlabel(r'EW$_{\rm Ly\alpha}$')
    axs[2,2].set_ylabel(r'$\Delta v_{\rm Ly\alpha}$')

    axs[0,3].set_xlabel(r'$\Delta v_{\rm Ly\alpha}$')
    axs[0,3].set_ylabel(r'$f_{\rm esc}$')
    axs[1,3].set_xlabel(r'$\Delta v_{\rm Ly\alpha}$')
    axs[1,3].set_ylabel(r'$f_{\rm esc}$')
    axs[2,3].set_xlabel(r'$\Delta v_{\rm Ly\alpha}$')
    axs[2,3].set_ylabel(r'$f_{\rm esc}$')

    axs[0,4].set_xlabel(r'EW$_{\rm Ly\alpha}$')
    axs[0,4].set_ylabel(r'$f_{\rm esc}$')
    axs[1,4].set_xlabel(r'EW$_{\rm Ly\alpha}$')
    axs[1,4].set_ylabel(r'$f_{\rm esc}$')
    axs[2,4].set_xlabel(r'EW$_{\rm Ly\alpha}$')
    axs[2,4].set_ylabel(r'$f_{\rm esc}$')

    axs[0,1].set_yscale('log')
    axs[0,2].set_xscale('log')
    axs[0,4].set_xscale('log')
    axs[1,1].set_yscale('log')
    axs[1,2].set_xscale('log')
    axs[1,4].set_xscale('log')
    axs[2,1].set_yscale('log')
    axs[2,2].set_xscale('log')
    axs[2,4].set_xscale('log')
    
    axs[0,3].set_xscale('log')
    axs[1,3].set_xscale('log')
    axs[2,3].set_xscale('log')

    axs[0,3].set_xlim(1e1, 1e3)
    axs[1,3].set_xlim(1e1, 1e3)
    axs[2,3].set_xlim(1e1, 1e3)

    axs[0,3].set_ylim(0, 1)
    axs[0,4].set_ylim(0, 1)
    axs[1,3].set_ylim(0, 1)
    axs[1,4].set_ylim(0, 1)
    axs[2,3].set_ylim(0, 1)
    axs[2,4].set_ylim(0, 1)

    plt.show()
