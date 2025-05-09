import os
import py21cmfast as p21c
import numpy as np

from scipy import integrate
from astropy.cosmology import Planck18, z_at_value
from astropy import units as U, constants as c

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

def get_silverrush_laelf(z):
    if z==4.9:
        # SILVERRUSH XIV z=4.9 LAELF
        lum_silver = np.array([42.75, 42.85, 42.95, 43.05, 43.15, 43.25, 43.35, 43.45, 43.55, 43.65])
        logphi_silver = -1*np.array([2.91, 3.17, 3.42, 3.78, 3.88, 4.00, 4.75, 4.93, 5.23, 4.93])
        logphi_up_silver = 1e-2*np.array([5, 5, 6, 9, 10, 12, 29, 36, 52, 36])
        logphi_low_silver = 1e-2*np.array([5, 5, 6, 9, 10, 12, 34, 45, 77, 45])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    elif z==5.7:
        # SILVERRUSH XIV z=5.7 LAELF
        lum_silver = np.array([42.85, 42.95, 43.05, 43.15, 43.25, 43.35, 43.45, 43.55, 43.65, 43.75, 43.85, 43.95])
        logphi_silver = -1*np.array([3.05, 3.27, 3.56, 3.85, 4.15, 4.41, 4.72, 5.15, 5.43, 6.03, 6.33, 6.33])
        logphi_up_silver = 1e-2*np.array([4, 2, 2, 3, 4, 5, 7, 12, 17, 36, 52, 52])
        logphi_low_silver = 1e-2*np.array([4, 2, 2, 3, 4, 5, 7, 13, 18, 45, 77, 77])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    elif z==6.6:
        # SILVERRUSH XIV z=6.6 LAELF
        lum_silver = np.array([42.95, 43.05, 43.15, 43.25, 43.35, 43.45, 43.55, 43.65, 43.75, 43.95, 44.05])
        logphi_silver = -1*np.array([3.71, 4.11, 4.37, 4.65, 4.83, 5.28, 5.89, 5.9, 5.9, 6.38, 6.38])
        logphi_up_silver = 1e-2*np.array([9, 5, 6, 7, 8, 14, 29, 29, 29, 52, 52])
        logphi_low_silver = 1e-2*np.array([9, 5, 6, 7, 8, 15, 34, 34, 34, 77, 77])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    elif z==7.0:
        # wip
        # SILVERRUSH XIV z=7.0 LAELF
        lum_silver = np.array([43.25, 43.35])
        logphi_silver = -1*np.array([4.4, 4.95])
        logphi_up_silver = 1e-2*np.array([29, 52])
        logphi_low_silver = 1e-2*np.array([34, 77])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    elif z==7.3:
        # wip
        # SILVERRUSH XIV z=7.3 LAELF
        lum_silver = np.array([43.45])
        logphi_silver = -1*np.array([4.81])
        logphi_up_silver = 1e-2*np.array([36])
        logphi_low_silver = 1e-2*np.array([45])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver

def get_silverrush_ewpdf(z):
    assert z in [4.9, 5.7, 6.6]
    if z==4.9:
        ew0_low = 38.5
        ew0_low_upper = 2.2
        ew0_low_lower = 2.0
        ew0_high = 90.3
        ew0_high_upper = 1.9
        ew0_high_lower = 1.7
    elif z==5.7:
        ew0_low = 32.9
        ew0_low_upper = 0.8
        ew0_low_lower = 0.7
        ew0_high = 76.0
        ew0_high_upper = 0.7
        ew0_high_lower = 0.6
    elif z==6.6:
        ew0_low = 52.3
        ew0_low_upper = 6.5
        ew0_low_lower = 5.3
        ew0_high = 114.5
        ew0_high_upper = 5.6
        ew0_high_lower = 4.9
    return ew0_low, ew0_low_upper, ew0_low_lower, \
        ew0_high, ew0_high_upper, ew0_high_lower

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

    mode_names = ['fiducial model']
    modes = ['sfr']
    colors = ['cyan', 'magenta']

    # location of HMFs
    dat_dir = f'./data/z{z}/'
    # location of halo summaries
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

    fig, axs = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)

    for Lya_phi, log_Lya_asymmetric_error, bin_centers, mode, color in zip(lya_list, err_list, lya_bins, mode_names, colors):
        axs[0].errorbar(bin_centers, np.log10(Lya_phi), yerr=log_Lya_asymmetric_error, \
                    markersize=10, capsize=15, fmt='o', color=color, alpha=1.0, label=mode)
        
    # supply redshift as an argument
    lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver = get_silverrush_laelf(z)
    
    g18_phi_upper = np.asarray(logphi_up_silver)
    g18_phi_lower = np.asarray(logphi_low_silver)
    asymmetric_error = np.array(list(zip(g18_phi_lower, g18_phi_upper))).T
    axs[0].errorbar(lum_silver, logphi_silver, yerr=asymmetric_error, fmt='o', \
                    markersize=10, capsize=15, color='white', label='Subaru')
    # g18_phi_upper = np.asarray(logphi_up_konno)
    # g18_phi_lower = np.asarray(logphi_low_konno)
    # asymmetric_error = np.array(list(zip(g18_phi_lower, g18_phi_upper))).T
    # axs[0].errorbar(lum_konno, logphi_konno, yerr=asymmetric_error, fmt='o', color='red', label='Konno+18')
    axs[0].set_xlabel(r'$\log_{10} L_{\mathrm{Ly}\alpha} \, [\mathrm{erg/s}]$', fontsize=font_size)
    axs[0].set_ylabel(r'$\log_{10} \phi \, [\mathrm{Mpc}^{-3} \, \mathrm{dex}^{-1}]$', fontsize=font_size)
    axs[0].set_ylim(-7, -3)
    axs[0].grid()
    # axs[0].legend(fontsize=font_size)

    # SILVERRUSH XIV
    if z in [4.9, 5.7, 6.6]:
        ew0_low, ew0_low_upper, ew0_low_lower, \
        ew0_high, ew0_high_upper, ew0_high_lower \
            = get_silverrush_ewpdf(z)

        w1 = np.random.exponential(ew0_low, 10000)
        w2 = np.random.exponential(ew0_high-ew0_low, 10000)
        w0 = w1 + w2

        # bins, heights = np.hist

        W_density_err_up = np.abs(np.log10(W_density_err+W_density) - np.log10(W_density))
        W_density_err_low = np.abs(np.log10(np.abs(W_density-W_density_err)) - np.log10(W_density))
        W_density_asymmetric_err = np.array(list(zip(W_density_err_low, W_density_err_up))).T

        for W_density, W_density_asymmetric_err, bin_centers, mode, color in zip(W_list, W_err_list, W_bins, mode_names, colors):
            axs[1].errorbar(bin_centers, W_density, yerr=W_density_asymmetric_err,\
                        fmt='o', markersize=10, capsize=15, color=color, alpha=1.0, label=mode)

            bin_edges = np.zeros(len(bin_centers)+1)
            bin_width = (bin_centers[1] - bin_centers[0]) / 2
            bin_edges[:-1] = bin_centers - bin_width
            bin_edges[-1] = bin_centers[-1] + bin_width

            heights, bins = np.histogram(w0, bins=bin_edges)
            factor = np.sum(heights)*bin_width*2
            heights = heights / factor
            height_errs = np.sqrt(heights) / factor

            axs[1].errorbar(bin_centers, heights, yerr=height_errs, color='white', \
                            fmt='o', markersize=10, capsize=15, label='Subaru')
    
        

        axs[1].set_yscale('log')
        axs[1].set_xlim(1, 500)

        axs[1].set_xlabel(r'${\rm W}_{\rm emerg}$ [$\AA$]', fontsize=font_size)
        axs[1].set_ylabel(r'PDF', fontsize=font_size)
        axs[1].grid()
        axs[1].legend(fontsize=font_size)

    plt.show()