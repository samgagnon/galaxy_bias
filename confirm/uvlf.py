import numpy as np

from astropy.cosmology import Planck18, z_at_value
from astropy import units as u, constants as c

# BOUWENS 2021
b21_mag = [[-22.52, -22.02, -21.52, -21.02, -20.52, -20.02, -19.52, -18.77, -17.77, -16.77],
          [-22.19, -21.69, -21.19, -20.68, -20.19, -19.69, -19.19, -18.69, -17.94, -16.94],
          [-21.85, -21.35, -20.85, -20.10, -19.35, -18.6, -17.6]]
b21_phi = [[2e-6, 1.4e-5, 5.1e-5, 1.69e-4, 3.17e-4, 7.24e-4, 1.124e-3, 2.82e-3, 8.36e-3, 1.71e-2],
          [1e-6, 4.1e-5, 4.7e-5, 1.98e-4, 2.83e-4, 5.89e-4, 1.172e-3, 1.433e-3, 5.76e-3, 8.32e-3],
          [3e-6, 1.2e-5, 4.1e-5, 1.2e-4, 6.57e-4, 1.1e-3, 3.02e-3]]
b21_phi_err = [[2e-6, 5e-6, 1.1e-5, 2.4e-5, 4.1e-5, 8.7e-5, 1.57e-4, 4.4e-4, 1.66e-3, 5.26e-3],
              [2e-6, 1.1e-5, 1.5e-5, 3.6e-5, 6.6e-5, 1.26e-4, 3.36e-4, 4.19e-4, 1.44e-3, 2.9e-3],
              [2e-6, 4e-6, 1.1e-5, 4e-5, 2.33e-4, 3.4e-4, 1.14e-3]]

b21_6 = np.array(b21_phi[0])
b21_7 = np.array(b21_phi[1])
b21_8 = np.array(b21_phi[2])

b21_6_err = np.array(b21_phi_err[0])
b21_7_err = np.array(b21_phi_err[1])
b21_8_err = np.array(b21_phi_err[2])

logphi_b21_6 = np.log10(b21_6)
logphi_b21_7 = np.log10(b21_7)
logphi_b21_8 = np.log10(b21_8)

logphi_err_b21_6_up = np.log10(b21_6 + b21_6_err) - logphi_b21_6
logphi_err_b21_7_up = np.log10(b21_7 + b21_7_err) - logphi_b21_7
logphi_err_b21_8_up = np.log10(b21_8 + b21_8_err) - logphi_b21_8

logphi_err_b21_6_low = logphi_b21_6 - np.log10(b21_6 - b21_6_err)
logphi_err_b21_7_low = logphi_b21_7 - np.log10(b21_7 - b21_7_err)
logphi_err_b21_8_low = logphi_b21_8 - np.log10(b21_8 - b21_8_err)

logphi_err_b21_6_low[np.isinf(logphi_err_b21_6_low)] = np.abs(logphi_b21_6[np.isinf(logphi_err_b21_6_low)])
logphi_err_b21_7_low[np.isinf(logphi_err_b21_7_low)] = np.abs(logphi_b21_7[np.isinf(logphi_err_b21_7_low)])
logphi_err_b21_8_low[np.isinf(logphi_err_b21_8_low)] = np.abs(logphi_b21_8[np.isinf(logphi_err_b21_8_low)])

logphi_err_b21_6_low[np.isnan(logphi_err_b21_6_low)] = np.abs(logphi_b21_6[np.isnan(logphi_err_b21_6_low)])
logphi_err_b21_7_low[np.isnan(logphi_err_b21_7_low)] = np.abs(logphi_b21_7[np.isnan(logphi_err_b21_7_low)])
logphi_err_b21_8_low[np.isnan(logphi_err_b21_8_low)] = np.abs(logphi_b21_8[np.isnan(logphi_err_b21_8_low)])

def get_halo_field(redshift):
    """
    Loads a halo field from disk.
    """
    return np.load(f'../data/lightcone_props/halo_fields/halo_field_{redshift}.npy')

def get_muv(sfr):
    """
    Converts star formation rate to
    absolute UV magnitude.
    """
    kappa = 1.15e-28
    luv = sfr * 3.1557e7 / kappa
    muv = 51.64 - np.log10(luv) / 0.4
    return muv

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')

    redshift = 5.7
    # redshift = 6.6
    
    halo_field = get_halo_field(redshift)

    x, y, z, halo_masses, stellar_masses, sfr = halo_field

    # convert SFR to absolute UV magnitude
    muv = get_muv(sfr)

    bin_edges = []
    bin_centers = np.asarray(b21_mag[1])
    bin_edge = np.zeros(len(bin_centers)+1)
    bin_widths = (bin_centers[1:] - bin_centers[:-1])*0.5
    bin_edge[0] = bin_centers[0]-bin_widths[0]
    bin_edge[-1] = bin_centers[-1] + bin_widths[-1]
    bin_edge[1:-1] = bin_widths + bin_centers[:-1]
    bin_edges += [bin_edge]

    heights, bins = np.histogram(muv, bins=bin_edges[0])
    # heights, bins = np.histogram(Llya)
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    bin_widths = bins[1:]-bins[:-1]
    uv_phi = heights/bin_widths/(200**3)
    uv_phi_err = np.sqrt(heights)/bin_widths/(200**3)
    log_uv_phi_err_up = np.abs(np.log10(uv_phi_err+uv_phi) - np.log10(uv_phi))
    log_uv_phi_err_low = np.abs(np.log10(np.abs(uv_phi-uv_phi_err)) - np.log10(uv_phi))
    log_uv_phi_err_low[np.isinf(log_uv_phi_err_low)] = np.abs(np.log10(uv_phi[np.isinf(log_uv_phi_err_low)]))
    log_uv_asymmetric_error = np.array(list(zip(log_uv_phi_err_low, log_uv_phi_err_up))).T


    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    ax.errorbar(bin_centers, np.log10(uv_phi), yerr=log_uv_asymmetric_error, fmt='o', color='white', label='This Work')
    ax.errorbar(b21_mag[1], logphi_b21_7, yerr=[logphi_err_b21_7_low, logphi_err_b21_7_up], fmt='o', color='red', label='Bouwens+2021')
    ax.set_xlabel(r'$M_{\rm UV}$')
    ax.set_ylabel(r'$\log_{10}(\Phi)$')
    # ax.set_xlim(-10, -5)
    # ax.set_ylim(-8, -1)
    ax.legend()
    plt.show()