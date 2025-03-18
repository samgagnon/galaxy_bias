import numpy as np
from astropy.cosmology import Planck18

import py21cmfast as p21c

# from p21c_utils.analysis_funcs import get_props_from_halofield
# from p21c_utils.analysis_plots import scaling_plot

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')

    # inputs = p21c.InputParameters.from_template('../sgh24.toml',\
    #     random_seed=1).evolve_input_structs(SAMPLER_MIN_MASS=1e8)

    # halo_field = p21c.HaloField.read('../data/halo_fields/z6.6/sgh24_MIN_MASS_8.0.h5')
    # halo_properties = get_props_from_halofield(halo_field, inputs, sel=None,kinds=['sfr',])
    # halo_masses = halo_properties['halo_masses']

    halo_masses = np.load('../data/lightcone_props/halo_fields/halo_field_6.6.npy')[3]

    fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

    ST_ref_dndm = np.load(f'../data/z6.6/hmf_dndm_ST.npy')*Planck18.h**4
    ST_ref_m = np.load(f'../data/z6.6/hmf_m_ST.npy')/Planck18.h
    axs.plot(np.log10(ST_ref_m), np.log10(ST_ref_dndm), color='red', linestyle='-', label='ST')

    heights, bins = np.histogram(np.log10(halo_masses), bins=100)
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    bin_widths = bins[1:]-bins[:-1]
    hmf = heights/bin_widths/(200**3)
    hmf_err = np.sqrt(heights)/bin_widths/(200**3)

    log_hmf_err_up = np.abs(np.log10(hmf_err+hmf) - np.log10(hmf))
    log_hmf_err_low = np.abs(np.log10(np.abs(hmf-hmf_err)) - np.log10(hmf))
    log_hmf_err_low[np.isinf(log_hmf_err_low)] = np.abs(np.log10(hmf[np.isinf(log_hmf_err_low)]))
    log_hmf_asymmetric_error = np.array(list(zip(log_hmf_err_low, log_hmf_err_up))).T

    axs.errorbar(bin_centers, np.log10(hmf)-bin_centers, yerr=log_hmf_asymmetric_error, fmt='o', \
                color='white', label='This Work')

    axs.set_xlabel(r'$\log_{10}(M_{\rm halo}/M_{\odot})$')
    axs.set_ylabel(r'$\log_{10}(dn/dm)$')
    axs.set_ylim(np.log10(hmf[-1])-bin_centers[-1], np.log10(hmf[0])-bin_centers[0])
    axs.legend()
    plt.show()