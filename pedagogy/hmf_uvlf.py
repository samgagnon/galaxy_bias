import os
import py21cmfast as p21c
import numpy as np

from scipy import integrate
from astropy.cosmology import Planck18, z_at_value
from astropy import units as U, constants as c

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

def get_A(m):
            return 0.65 + 0.1 * np.tanh(3 * (m + 20.75))

def get_Wc(m):
    return 31 + 12 * np.tanh(4 * (m + 20.25))

def probability_EW(W, Wc):
    return (1/Wc) * np.exp(-W/Wc)

# Bouwens 2014
def get_beta(Muv):
    a = -2.05
    b = -0.2
    return a + b*(Muv+19.5)

def sfr2Muv(sfr):
        kappa = 1.15e-28
        Luv = sfr * 3.1557e7 / kappa
        Muv = 51.64 - np.log10(Luv) / 0.4
        return Muv

def mason2018(Muv):
    """
    Samples EW and emission probability from the
    fit functions obtained by Mason et al. 2018.
    """
    A = get_A(Muv)
    rv_A = np.random.uniform(0, 1, len(Muv))
    emit_bool = rv_A < A
    Wc = get_Wc(Muv[emit_bool])
    rv_W = np.random.uniform(0, 1, len(Wc))
    W = -1*Wc*np.log(rv_W)
    return W, emit_bool

def lu2024(Mh, Muv, z):
    """
    Samples EW and emission probability from the
    fit functions obtained by Lu et al. 2018.
    """
    A = get_A(Muv)
    rv_A = np.random.uniform(0, 1, len(Muv))
    emit_bool = rv_A < A

    # this takes the halo mass!
    m = 0.32
    C = 2.48
    dv_mean = m*(np.log10(Mh) - np.log10(1.55) - 12) + C
    dv_sigma = 0.24
    dv = np.random.normal(dv_mean, dv_sigma, len(Mh))
    # sample velocity offsets
    # apply truncation at v_circ
    v_circ = ((100*C.G*Mh*U.solMass*Planck18.H(z)).to('').value)**(1/3)

    Wc = get_Wc(Muv[emit_bool])
    rv_W = np.random.uniform(0, 1, len(Wc))
    W = -1*Wc*np.log(rv_W)
    return W, emit_bool

def get_PTH_fn(rs):
    """
    Given a random seed (rs), return the relevant PTH file.
    This must be modified to instead take redshift as an argument.
    """
    path = '../lya_langevin/auxiliary_fields/'
    field_list = os.listdir(path)
    field_list = [field for field in field_list if len(field.split('_'))==3]
    field_dict = {}
    for field in field_list:
        rs = int(field.split('_')[-1][2:])
        field_dict[rs] = field
    item_list = os.listdir(f'{path}{field_dict[rs]}')

    for item in item_list:
        if item.startswith('PerturbHaloField'):
            return f'{path}{field_dict[rs]}/{item}'

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc) 
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')
    import matplotlib as mpl
    label_size = 20
    font_size = 30
    mpl.rcParams['xtick.labelsize'] = label_size 
    mpl.rcParams['ytick.labelsize'] = label_size 

    import argparse

    parser = argparse.ArgumentParser(description='Plot summaries of the halo catalog.')
    parser.add_argument('-z', type=float, default=6.6, help='Redshift of the halo catalog.')

    args = parser.parse_args()
    z = args.z

    mode_names = ['21cmFASTv4']
    modes = ['sfr']
    colors = ['cyan', 'magenta']

    # location of HMFs
    dat_dir = f'./data/z{z}/'
    load_dir = f'./summaries/z{z}/'

    uv_list = []
    uv_err_list = []
    lya_list = []
    err_list = []
    W_list = []
    W_err_list = []
    uv_bins = []
    lya_bins = []
    W_bins = []

    for mode in modes:

        uv_phi = np.load(f'.{load_dir}/{mode}_uv_phi.npy')
        uv_phi_err = np.load(f'.{load_dir}/{mode}_uv_phi_err.npy')
        Lya_phi = np.load(f'.{load_dir}/{mode}_Lya_phi.npy')
        log_Lya_asymmetric_error = np.load(f'.{load_dir}/{mode}_Lya_phi_err.npy')
        W_density = np.load(f'.{load_dir}/{mode}_W_density.npy')
        W_density_err = np.load(f'.{load_dir}/{mode}_W_density_err.npy')
        lya_bin_centers = np.load(f'.{load_dir}/{mode}_lya_bin_centers.npy')
        W_bin_centers = np.load(f'.{load_dir}/{mode}_W_bin_centers.npy')
        uv_bin_centers = np.load(f'.{load_dir}/{mode}_uv_bin_centers.npy')
        
        uv_list += [uv_phi]
        uv_err_list += [uv_phi_err]
        lya_bins += [lya_bin_centers]
        W_bins += [W_bin_centers]
        lya_list += [Lya_phi]
        err_list += [log_Lya_asymmetric_error]
        W_list += [W_density]
        W_err_list += [W_density_err]
        uv_bins += [uv_bin_centers]

    hmf_dndm = np.load(f'.{load_dir}/hmf_dndm.npy')
    hmf_dndm_err = np.load(f'.{load_dir}/hmf_dndm_err.npy')
    hmf_m = np.load(f'.{load_dir}/hmf_log10m.npy')

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    # axs[0].set_yticks([])
    axs[0].set_xlabel(r'$\log_{10} M_h$ [M$_{\odot}$]', fontsize=font_size)
    # axs[0].set_xticklabels([])
    ax_hmf = axs[0]
    # ax_hmf.set_xticklabels([])

    ax_hmf.errorbar(hmf_m, np.log10(hmf_dndm)-hmf_m, yerr=hmf_dndm_err, fmt='o', \
                    markersize=10, capsize=15, color='cyan', alpha=1.0, label=r'21cmFASTv4')
    
    ST_ref_dndm = np.load(f'.{dat_dir}/hmf_dndm_ST.npy')*Planck18.h**4
    ST_ref_m = np.load(f'.{dat_dir}/hmf_m_ST.npy')/Planck18.h
    ax_hmf.plot(np.log10(ST_ref_m), np.log10(ST_ref_dndm), color='red', linestyle='-', label='Sheth-Tormen')

    ax_hmf.set_xlim(8, 13)
    ax_hmf.set_ylim(-20, -5)
    ax_hmf.set_ylabel(r'$\log_{10} \mathrm{d}n/\mathrm{d}M$ [Mpc$^{-3}$]', fontsize=font_size)
    ax_hmf.grid()
    ax_hmf.legend(fontsize=font_size)

    asymmetric_error = np.array(list(zip(logphi_err_b21_6_low, logphi_err_b21_6_up))).T
    axs[1].errorbar(b21_mag[0], logphi_b21_6, yerr=asymmetric_error, fmt='o', color='red', \
                    markersize=10, capsize=15, label=r'HST ($z=6$)')
    asymmetric_error = np.array(list(zip(logphi_err_b21_7_low, logphi_err_b21_7_up))).T
    axs[1].errorbar(b21_mag[1], logphi_b21_7, yerr=asymmetric_error, fmt='x', color='red', \
                    markersize=10, capsize=15, label=r'HST ($z=7$)')

    for uv_phi, log_uv_asymmetric_error, bin_centers, mode, color in zip(uv_list, uv_err_list, uv_bins, mode_names, colors):
        # this assumes that the UV errors are relative, it's not clear that they are
        axs[1].errorbar(bin_centers, np.log10(uv_phi), yerr=-1*log_uv_asymmetric_error*np.log10(uv_phi), \
                    fmt='o', markersize=10, capsize=15, color=color, alpha=1.0, label=mode)

    axs[1].set_xlabel(r'$M_{\mathrm{UV}}$', fontsize=font_size)
    axs[1].set_ylabel(r'$\log_{10} \phi \, [\mathrm{Mpc}^{-3} \, \mathrm{mag}^{-1}]$', fontsize=font_size)
    axs[1].set_ylim(-7, -1)
    axs[1].grid()
    axs[1].legend(fontsize=font_size)

    plt.show()