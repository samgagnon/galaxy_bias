import os

import numpy as np

from astropy.cosmology import Planck18
from astropy import units as u
from astropy import constants as c
from scipy.special import erf

def get_halo_field(redshift):
    """
    Loads a halo field from disk.
    """
    return np.load(f'./data/lightcone_props/halo_fields/halo_field_{redshift}.npy')

def get_muv(sfr):
    """
    Converts star formation rate to
    absolute UV magnitude.
    """
    kappa = 1.15e-28
    luv = sfr * 3.1557e7 / kappa
    muv = 51.64 - np.log10(luv) / 0.4
    return muv

def get_a_mason(m):
    return 0.65 + 0.1 * np.tanh(3 * (m + 20.75))

def get_wc_mason(m):
    return 31 + 12 * np.tanh(4 * (m + 20.25))

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

def mh_muv_mason2015(muv, redshift):
    """
    Returns the halo mass from the UV magnitude
    using the fit function from Mason et al. 2015.
    """
    gamma = np.zeros_like(muv)
    gamma[muv >= -20 - redshift*0.26] = -0.3
    gamma[muv < -20 - redshift*0.26] = -0.7
    return 10**(gamma*(muv + 20 + 0.26*redshift) + 11.75)

def get_dv_mason2018(muv, redshift):
    halo_masses = mh_muv_mason2015(muv, redshift)
    m = 0.32
    c2 = 2.48
    dv_mean = m*(np.log10(halo_masses) - np.log10(1.55) - 12) + c2
    dv_sigma = 0.24
    dv = np.random.normal(dv_mean, dv_sigma, len(halo_masses))
    return 10**dv

def plot(mode: str, *args):
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc) 
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')

    if mode == 'muv':
        halo_masses, muv = args

        fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

        muv_fit = np.linspace(muv.min(), muv.max(), 1000)
        mh_fit = mh_muv_mason2015(muv_fit, 6.6)

        axs.scatter(halo_masses, muv, s=1, alpha=0.1, rasterized=True)  
        axs.plot(mh_fit, muv_fit, color='black', linestyle='--')
        axs.set_xlabel(r'$M_{\rm halo}$ [$M_{\odot}$]')
        axs.set_ylabel(r'$M_{\rm UV}$')
        axs.set_xscale('log')
        axs.set_xlim(5e9, 1e11)
        
        plt.show()
        quit()

    elif mode == 'ew':

        muv, w = args

        fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

        axs.scatter(muv, w, s=1, alpha=0.1, rasterized=True)
        axs.set_xlabel(r'$M_{\rm UV}$')
        axs.set_ylabel(r'EW $[\AA]$')
        axs.set_xlim(-10, -5)
        axs.set_ylim(0, 300)

        plt.show()
        quit()

    elif mode == 'dv':

        halo_masses, dv, redshift = args

        halo_mass_range = np.linspace(5e9, 1e11, 1000)
        v_circ = (((10*c.G*halo_mass_range*u.solMass*Planck18.H(redshift)).to('km^3/s^3').value)**(1/3))

        fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

        axs.scatter(halo_masses[halo_masses>5e9], dv[halo_masses>5e9], s=1, alpha=0.1, color='cyan', rasterized=True)
        axs.plot(halo_mass_range, v_circ, color='orange', linestyle='-')
        axs.set_xlabel(r'$M_{\rm halo}$ [$M_{\odot}$]')
        axs.set_ylabel(r'$\Delta v$ [km/s]')
        axs.set_xscale('log')
        axs.set_xlim(5e9, 1e11)
        axs.set_ylim(0, 200)

        plt.show()
        quit()

    elif mode == 'tau_cgm':

        from get_relevant_fields import get_stellar_mass, get_sfr

        halo_mass, redshift = args

        halo_mass_range = [1e9, 1e10, 1e11]
        tau_cgm_range = []

        for halo_mass in halo_mass_range:

            halo_masses = np.ones(10000)*halo_mass
            stellar_rng = np.random.uniform(0, 1, len(halo_masses))
            stellar_masses = get_stellar_mass(halo_masses, stellar_rng)
            sfr_rng = np.random.uniform(0, 1, len(halo_masses))
            sfr = get_sfr(stellar_masses, sfr_rng, 6.6)
            muv = get_muv(sfr)

            # sample velocity offset using mason 2018 model
            dv = get_dv_mason2018(muv, redshift)
            # CGM absorbs emission beneath the circular velocity
            v_circ = ((10*c.G*halo_masses*u.solMass*Planck18.H(redshift)).to('km^3/s^3').value)**(1/3)
            # plot('dv', halo_masses, dv)
            # treat velocity offset as equal to FWHM
            sigma_v = dv / (2*np.sqrt(2*np.log(2)))
            # what percent of the curve remains after CGM attenuation?
            tau_cgm  = 1 - 0.5*(1 + erf((v_circ - dv)/(np.sqrt(2)*sigma_v)))

            tau_cgm_range.append(tau_cgm)

        fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True, constrained_layout=True)

        for i, tau_cgm in enumerate(tau_cgm_range):
            axs[i].hist(tau_cgm[tau_cgm>0.0], bins=100, color='cyan', alpha=1.0)
            # axs[i].hist(np.log10(tau_cgm[tau_cgm>0.0]), bins=100, color='cyan', alpha=1.0)
            axs[i].set_xlabel(r'$\mathcal{T}_{\rm CGM}$')
            axs[i].set_xlim(0, 1)
            # axs[i].set_xlim(-10, 0)
            axs[i].set_title(r'$M_{\rm halo}=1$e' + str(int(np.log10(halo_mass_range[i]))) \
                            + r' $M_{\odot}$')

        axs[0].set_ylabel(r'$N$')
        plt.show()
        quit()

    elif mode == 'ew_dv':

        muv, w, redshift = args

        # sample velocity offset using mason 2018 model
        dv = get_dv_mason2018(muv, redshift)

        fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

        axs.scatter(w, dv, s=1, alpha=0.1, color='cyan', rasterized=True)
        axs.set_xlabel(r'EW $[\AA]$')
        axs.set_ylabel(r'$\Delta v$ [km/s]')
        axs.set_xlim(0, 300)
        axs.set_ylim(0, 200)
        plt.show()
        quit()

    elif mode == 'ew_obs':

        muv, halo_masses, w, redshift = args

        # sample velocity offset using mason 2018 model
        dv = get_dv_mason2018(muv, redshift)
        # CGM absorbs emission beneath the circular velocity
        v_circ = ((10*c.G*halo_masses*u.solMass*Planck18.H(redshift)).to('km^3/s^3').value)**(1/3)
        # treat velocity offset as equal to FWHM
        sigma_v = dv / (2*np.sqrt(2*np.log(2)))
        # what percent of the curve remains after CGM attenuation?
        tau_cgm  = 1 - 0.5*(1 + erf((v_circ - dv)/(np.sqrt(2)*sigma_v)))

        fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

        axs.scatter(w/tau_cgm, w, s=1, alpha=0.1, color='cyan', rasterized=True)
        axs.set_xlabel(r'EW$_{\rm intr}$ $[\AA]$')
        axs.set_ylabel(r'EW$_{\rm obs}$ $[\AA]$')
        axs.set_xscale('log')
        axs.set_ylim(0, 500)
        plt.show()
        quit()

    elif mode == 'example':

        vspace = np.linspace(-10, 200, 1000)

        fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

        axs.plot(vspace, np.exp(-1*(vspace-50)**2/50), color='cyan', linestyle='-')
        axs.axvline(60, color='orange', linestyle='--')
        axs.set_xlabel(r'$\Delta v$ [km/s]')
        axs.set_ylabel(r'flux (arbitrary units)')
        axs.set_xlim(0, 200)
        axs.set_ylim(0, 1)
        plt.show()
        quit()

if __name__ == "__main__":

    redshift = 5.7
    # redshift = 6.6

    # plot('example')

    # INPUT PARAMETERS: what are you solving for?

    halo_field = get_halo_field(redshift)

    x, y, z, halo_masses, stellar_masses, sfr = halo_field

    # sfr to muv
    muv = get_muv(sfr)

    plot('muv', halo_masses, muv)

    # sample lae fraction
    w, emit_bool = get_ewpdf_mason2018(muv)

    # plot('ew', muv[emit_bool], w)

    # plot('tau_cgm', 1e10, redshift)

    # plot('ew_dv', muv[emit_bool], w, redshift)

    # plot('ew_obs', muv[emit_bool], halo_masses[emit_bool], w, redshift)

    # sample velocity offset using mason 2018 model
    dv = get_dv_mason2018(muv, redshift)

    # CGM absorbs emission beneath the circular velocity
    v_circ = ((10*c.G*halo_masses*u.solMass*Planck18.H(redshift)).to('km^3/s^3').value)**(1/3)

    # plot('dv', halo_masses, dv, redshift)

    # treat velocity offset as equal to FWHM
    sigma_v = dv / (2*np.sqrt(2*np.log(2)))

    # what percent of the curve remains after CGM attenuation?
    tau_cgm  = 1 - 0.5*(1 + erf((v_circ - dv)/(np.sqrt(2)*sigma_v)))

    # Now what do we do?