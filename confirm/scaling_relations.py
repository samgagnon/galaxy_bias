import numpy as np
from astropy.cosmology import Planck18

def get_mean_stellar_mass(halo_masses):
    sigma_star = 0.5
    mp1 = 1e10
    mp2 = 2.8e11
    M_turn = 10**(8.7)
    a_star = 0.5
    a_star2 = -0.61
    f_star10 = 0.05
    omega_b = Planck18.Ob0
    omega_m = Planck18.Om0
    baryon_frac = omega_b/omega_m

    stellar_rng = 0.5
    
    high_mass_turnover = ((mp2/mp1)**a_star + (mp2/mp1)**a_star2)/\
        ((halo_masses/mp2)**(-1*a_star)+(halo_masses/mp2)**(-1*a_star2))
    stoc_adjustment_term = 0.5*sigma_star**2
    low_mass_turnover = np.exp(-1*M_turn/halo_masses + stellar_rng*sigma_star - stoc_adjustment_term)
    stellar_mass = f_star10 * baryon_frac * halo_masses * (high_mass_turnover * low_mass_turnover)
    return stellar_mass

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')

    x, y, z, halo_masses, stellar_masses, sfr = np.load('../data/lightcone_props/halo_fields/halo_field_6.6.npy')

    halo_mass_range = np.linspace(1e8, 1e11, 1000)
    stellar_mass_range = get_mean_stellar_mass(halo_mass_range)

    fig, axs = plt.subplots(1, 1, figsize=(8, 10), constrained_layout=True)

    # mh-mstar
    axs.scatter(halo_masses[::100], stellar_masses[::100], s=1, alpha=0.1, color='cyan', rasterized=True)

    axs.plot(halo_mass_range, stellar_mass_range, color='red', linestyle='--')

    axs.set_xlabel(r'$M_{\rm halo}$ [$M_{\odot}$]')
    axs.set_ylabel(r'$M_{\rm star}$ [$M_{\odot}$]')
    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.set_xlim(1e8, 1e11)
    axs.grid()

    # mh-sfr
    # axs[1].plot(halo_masses, sfr, alpha=0.1, color='cyan', rasterized=True)
    # axs[1].set_xlabel(r'$M_{\rm halo}$ [$M_{\odot}$]')
    # axs[1].set_ylabel(r'SFR [$M_{\odot}$ yr$^{-1}$]')
    # axs[1].set_xscale('log')
    # axs[1].set_yscale('log')
    # axs[1].set_xlim(1e8, 1e9)

    plt.show()