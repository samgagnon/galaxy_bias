import os
import py21cmfast as p21c
import numpy as np

from scipy import integrate
from astropy.cosmology import Planck18, z_at_value
from astropy import units as U, constants as c

from rng2sfr import *
from data import *

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

    # modes = ['sfr', 'mh']
    # mode_names = [r'$M_{UV}$-SFR (SGH)', r'$M_{UV}$-$M_{h}$ (M+18)']
    mode_names = ['This Work']
    modes = ['sfr']
    colors = ['cyan', 'magenta']
    z = 6.6

    # location of HMFs
    dat_dir = f'/data/z{z}/'
    # location of halo summaries
    load_dir = f'/data/z{z}/minmass_8/sam/'
    # load_dir = f'/data/z{z}/minmass_9/sam/'
    # load_dir = f'/data/z{z}/minmass_10/sam/'

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

    if z in [4.9, 5.7, 6.6]:
        fig, axs = plt.subplots(1, 4, figsize=(18, 6), constrained_layout=True)
    else:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    axs[0].set_yticks([])
    axs[0].set_xlabel(r'$\log_{10} M_{\mathrm{h}} \, [M_{\odot}]$')
    # axs[0].set_xticklabels([])
    ax_hmf = axs[0].inset_axes([0, 0.2, 1, 0.8])
    ax_hmf.set_xticklabels([])

    ST_ref_dndm = np.load(f'.{dat_dir}/hmf_dndm_ST.npy')*Planck18.h**3
    ST_ref_m = np.load(f'.{dat_dir}/hmf_m_ST.npy')/Planck18.h
    ax_hmf.plot(np.log10(ST_ref_m), np.log10(ST_ref_dndm), color='red', linestyle='-', label='ST')

    STM_ref_dndm = np.load(f'.{dat_dir}/hmf_dndm_Jenkins.npy')*Planck18.h**3
    STM_ref_m = np.load(f'.{dat_dir}/hmf_m_SMT.npy')/Planck18.h
    ax_hmf.plot(np.log10(STM_ref_m), np.log10(STM_ref_dndm), color='lime', linestyle='-', label='Jenkins')

    PS_ref_dndm = np.load(f'.{dat_dir}/hmf_dndm_PS.npy')*Planck18.h**3
    PS_ref_m = np.load(f'.{dat_dir}/hmf_m_PS.npy')/Planck18.h
    ax_hmf.plot(np.log10(PS_ref_m), np.log10(PS_ref_dndm), color='magenta', linestyle='-', label='PS')

    ax_hmf.errorbar(hmf_m, np.log10(hmf_dndm)-hmf_m, yerr=hmf_dndm_err, fmt='o', color='cyan', alpha=0.8, label=r'$21cmFASTv4$')

    ax_hmf.set_xlim(8, 13)
    ax_hmf.set_ylim(-20, -5)
    ax_hmf.set_ylabel(r'$\log_{10} \mathrm{d}n/\mathrm{d}M \, [\mathrm{Mpc}^{-3} \, \mathrm{dex}^{-1}]$')
    ax_hmf.grid()
    ax_hmf.legend()

    ax_hmf_r = axs[0].inset_axes([0, 0, 1, 0.2], sharex=ax_hmf)
    # ax_hmf_r = axs[0].inset_axes([0.3, 0, 0.7, 0.2], sharex=ax_hmf)

    mlim = np.log10(ST_ref_m[0])
    jenkins_interpolated = np.interp(hmf_m[hmf_m>mlim], np.log10(STM_ref_m), np.log10(STM_ref_dndm))
    st_interpolated = np.interp(hmf_m[hmf_m>mlim], np.log10(ST_ref_m), np.log10(ST_ref_dndm))
    ps_interpolated = np.interp(hmf_m[hmf_m>mlim], np.log10(PS_ref_m), np.log10(PS_ref_dndm))

    ax_hmf_r.errorbar(hmf_m[hmf_m>mlim], np.log10(hmf_dndm[hmf_m>mlim])-st_interpolated-hmf_m[hmf_m>mlim], \
                    yerr=hmf_dndm_err[:,hmf_m>mlim], fmt='o', color='red', alpha=0.8)
    ax_hmf_r.errorbar(hmf_m[hmf_m>mlim], np.log10(hmf_dndm[hmf_m>mlim])-ps_interpolated-hmf_m[hmf_m>mlim], \
                    yerr=hmf_dndm_err[:,hmf_m>mlim], fmt='o', color='magenta', alpha=0.8)
    ax_hmf_r.errorbar(hmf_m[hmf_m>mlim], np.log10(hmf_dndm[hmf_m>mlim])-jenkins_interpolated-hmf_m[hmf_m>mlim], \
                    yerr=hmf_dndm_err[:,hmf_m>mlim], fmt='o', color='lime', alpha=0.8)

    ax_hmf_r.set_xlim(8, 13)
    ax_hmf_r.set_ylim(-0.5, 2.0)
    axs[0].set_xticklabels([8, 9, 10, 11, 12, 13])
    ax_hmf_r.set_ylabel('Residuals')
    ax_hmf_r.grid()

    for uv_phi, log_uv_asymmetric_error, bin_centers, mode, color in zip(uv_list, uv_err_list, uv_bins, mode_names, colors):
        axs[1].errorbar(bin_centers, np.log10(uv_phi), yerr=log_uv_asymmetric_error, \
                    fmt='o', color=color, alpha=0.8, label=mode)

    asymmetric_error = np.array(list(zip(logphi_err_b21_6_low, logphi_err_b21_6_up))).T
    axs[1].errorbar(b21_mag[0], logphi_b21_6, yerr=asymmetric_error, fmt='o', color='red', label=r'B+21 ($z=6$)')
    asymmetric_error = np.array(list(zip(logphi_err_b21_7_low, logphi_err_b21_7_up))).T
    axs[1].errorbar(b21_mag[1], logphi_b21_7, yerr=asymmetric_error, fmt='x', color='red', label=r'B+21 ($z=7$)')

    axs[1].set_xlabel(r'$M_{\mathrm{UV}}$')
    axs[1].set_ylabel(r'$\log_{10} \phi \, [\mathrm{Mpc}^{-3} \, \mathrm{mag}^{-1}]$')
    axs[1].set_ylim(-7, -1)
    axs[1].grid()
    axs[1].legend()

    for Lya_phi, log_Lya_asymmetric_error, bin_centers, mode, color in zip(lya_list, err_list, lya_bins, mode_names, colors):
        axs[2].errorbar(bin_centers, np.log10(Lya_phi), yerr=log_Lya_asymmetric_error, \
                    fmt='o', color=color, alpha=0.8, label=mode)
        
    # supply redshift as an argument
    lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver = get_silverrush_laelf(z)
    
    g18_phi_upper = np.asarray(logphi_up_silver)
    g18_phi_lower = np.asarray(logphi_low_silver)
    asymmetric_error = np.array(list(zip(g18_phi_lower, g18_phi_upper))).T
    axs[2].errorbar(lum_silver, logphi_silver, yerr=asymmetric_error, fmt='o', color='lime', label='Umeda+24')
    # g18_phi_upper = np.asarray(logphi_up_konno)
    # g18_phi_lower = np.asarray(logphi_low_konno)
    # asymmetric_error = np.array(list(zip(g18_phi_lower, g18_phi_upper))).T
    # axs[2].errorbar(lum_konno, logphi_konno, yerr=asymmetric_error, fmt='o', color='red', label='Konno+18')
    axs[2].set_xlabel(r'$\log_{10} L_{\mathrm{Ly}\alpha} \, [\mathrm{erg/s}]$')
    axs[2].set_ylabel(r'$\log_{10} \phi \, [\mathrm{Mpc}^{-3} \, \mathrm{dex}^{-1}]$')
    axs[2].set_ylim(-7, -3)
    axs[2].grid()
    axs[2].legend()

    # SILVERRUSH XIV
    if z in [4.9, 5.7, 6.6]:
        ew0_low, ew0_low_upper, ew0_low_lower, \
        ew0_high, ew0_high_upper, ew0_high_lower \
            = get_silverrush_ewpdf(z)
        
        wrange_low = np.linspace(40, 200)
        wrange_high = np.linspace(200, 500)


        # TODO: rename variables for self-consistency
        exp1c_upper = ew0_low + ew0_low_upper
        exp1c_lower = ew0_low - ew0_low_lower
        exp2c_upper = ew0_high + ew0_high_upper
        exp2c_lower = ew0_high - ew0_high_lower
        exp1 = (1/ew0_low)*np.exp(-wrange_low/ew0_low)
        exp1_upper = (1/exp1c_upper)*np.exp(-wrange_low/exp1c_upper)
        exp1_lower = (1/exp1c_lower)*np.exp(-wrange_low/exp1c_lower)

        exp2 = (1/ew0_high)*np.exp(-wrange_high/ew0_high)
        exp2_upper = (1/exp2c_upper)*np.exp(-wrange_high/exp2c_upper)
        exp2_lower = (1/exp2c_lower)*np.exp(-wrange_high/exp2c_lower)

        exp2 *= exp1[-1]/exp2[0]
        exp2_upper *= exp1_upper[-1]/exp2_upper[0]
        exp2_lower *= exp1_lower[-1]/exp2_lower[0]

        sum_under_low = integrate.trapezoid(exp1, wrange_low)
        sum_under_high = integrate.trapezoid(exp2[wrange_high<W_bin_centers.max()], wrange_high[wrange_high<W_bin_centers.max()])
        sum_under_fit = sum_under_low + sum_under_high

        interpolated_density_low = np.interp(wrange_low, W_bin_centers, W_density)
        interpolated_density_high = np.interp(wrange_high[wrange_high<W_bin_centers.max()], W_bin_centers, W_density)
        sum_under_data = integrate.trapezoid(interpolated_density_low, wrange_low) + \
            integrate.trapezoid(interpolated_density_high, wrange_high[wrange_high<W_bin_centers.max()])

        exp1 *= sum_under_data/sum_under_fit
        exp1_upper *= sum_under_data/sum_under_fit
        exp1_lower *= sum_under_data/sum_under_fit

        exp2 *= sum_under_data/sum_under_fit
        exp2_upper *= sum_under_data/sum_under_fit
        exp2_lower *= sum_under_data/sum_under_fit

        W_density_err_up = np.abs(np.log10(W_density_err+W_density) - np.log10(W_density))
        W_density_err_low = np.abs(np.log10(np.abs(W_density-W_density_err)) - np.log10(W_density))
        W_density_asymmetric_err = np.array(list(zip(W_density_err_low, W_density_err_up))).T

        for W_density, W_density_asymmetric_err, bin_centers, mode, color in zip(W_list, W_err_list, W_bins, mode_names, colors):
            axs[3].errorbar(bin_centers, W_density, yerr=W_density_asymmetric_err,\
                        fmt='o', color=color, alpha=0.8, label=mode)
        
        axs[3].fill_between(wrange_low, exp1_lower, exp1_upper, color='lime', alpha=0.8)
        axs[3].plot(wrange_low, exp1, color='lime', linestyle='--', label='Umeda+24')
        axs[3].fill_between(wrange_high, exp2_lower, exp2_upper, color='lime', alpha=0.8)
        axs[3].plot(wrange_high, exp2, color='lime', linestyle='--')

        axs[3].set_yscale('log')
        axs[3].set_xlim(1, 500)

        axs[3].set_xlabel(r'$W_{\mathrm{Ly}\alpha} \, [\mathrm{\AA}]$')
        axs[3].set_ylabel(r'$\mathrm{PDF}$')
        axs[3].grid()
        axs[3].legend()
        axs[3].set_title('EW PDF', fontsize=20)

    ax_hmf.set_title('HMF', fontsize=20)
    axs[1].set_title('UVLF', fontsize=20)
    axs[2].set_title('LAELF', fontsize=20)

    fig.suptitle(r'z=6.6', fontsize=30)

    plt.savefig(f'/mnt/c/Users/sgagn/Downloads/halo_summaries.pdf')
    plt.show()