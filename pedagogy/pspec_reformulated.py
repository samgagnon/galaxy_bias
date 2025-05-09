import numpy as np

import py21cmfast as p21c

import powerbox

import logging, os, json
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('21cmFAST')

from scipy import integrate
from scipy import special
from scipy.optimize import minimize
from astropy.cosmology import Planck18

def get_slope(gamma):
    return gamma - 1

def get_intercept(r0, gamma):
    return np.real(gamma * np.log10(1j) + 0.5*np.log10(np.pi/2) + \
        -1*gamma*np.log10(r0) - np.log10(special.gamma(gamma)))

def get_line_params(x, y, yerr):
    # Objective function to minimize

    def objective(params):
        m, b = params
        # fit one sigma bounds and mean
        return np.sum(((m*x+b) - y)**2 / yerr**2)

    # Initial guess for slope and intercept
    initial_guess = [2, -5]

    # Minimize the objective function
    result = minimize(objective, initial_guess, bounds=[(0, 5), (-10, 5)])
    m, b  = result.x

    # Calculate residuals
    residuals = y - (m * x + b)

    # Sum of squared residuals
    SSR = np.sum(residuals**2)

    # Estimate variance of residuals
    N = len(x)
    sigma_squared = SSR / (N - 2)

    # Design matrix
    X = np.vstack([np.ones(len(x)), x]).T

    # Compute (X^T X)^-1
    XTX_inv = np.linalg.inv(X.T @ X)

    # Covariance matrix
    cov_matrix = sigma_squared * XTX_inv

    # Standard errors
    m_err, b_err = np.sqrt(np.diag(cov_matrix))

    return m, b, m_err, b_err

def ellipse(x, a, b, h, k):
    return b*np.sqrt(1 - (x-h)**2/a**2) + k

def get_bounds(r0, r0_up_err, r0_low_err, gamma, gamma_up_err, gamma_low_err):
    resolution = 10000
    
    r0_range_right = np.linspace(r0, r0+r0_up_err, resolution)
    r0_range_left = np.linspace(r0-r0_low_err, r0, resolution)

    r0_range_right2 = np.linspace(r0, r0+2*r0_up_err, resolution)
    r0_range_left2 = np.linspace(r0-2*r0_low_err, r0, resolution)

    r0_range_right3 = np.linspace(r0, r0+3*r0_up_err, resolution)
    r0_range_left3 = np.linspace(r0-3*r0_low_err, r0, resolution)

    r0_range_1 = np.concatenate([r0_range_right, r0_range_right[::-1], r0_range_left[::-1], r0_range_left])
    r0_range_2 = np.concatenate([r0_range_right2, r0_range_right2[::-1], r0_range_left2[::-1], r0_range_left2])
    r0_range_3 = np.concatenate([r0_range_right3, r0_range_right3[::-1], r0_range_left3[::-1], r0_range_left3])

    sigma1_upper_right = ellipse(r0_range_right, r0_up_err, gamma_up_err, r0, gamma)
    sigma1_upper_left = ellipse(r0_range_left, r0_low_err, gamma_up_err, r0, gamma)
    sigma1_lower_right = gamma-1*(ellipse(r0_range_right, r0_up_err, gamma_low_err, r0, gamma) - gamma)
    sigma1_lower_left = gamma-1*(ellipse(r0_range_left, r0_low_err, gamma_low_err, r0, gamma) - gamma)

    sigma2_upper_right = ellipse(r0_range_right2, 2*r0_up_err, 2*gamma_up_err, r0, gamma)
    sigma2_upper_left = ellipse(r0_range_left2, 2*r0_low_err, 2*gamma_up_err, r0, gamma)
    sigma2_lower_right = gamma-1*(ellipse(r0_range_right2, 2*r0_up_err, 2*gamma_low_err, r0, gamma) - gamma)
    sigma2_lower_left = gamma-1*(ellipse(r0_range_left2, 2*r0_low_err, 2*gamma_low_err, r0, gamma) - gamma)

    sigma3_upper_right = ellipse(r0_range_right3, 3*r0_up_err, 3*gamma_up_err, r0, gamma)
    sigma3_upper_left = ellipse(r0_range_left3, 3*r0_low_err, 3*gamma_up_err, r0, gamma)
    sigma3_lower_right = gamma-1*(ellipse(r0_range_right3, 3*r0_up_err, 3*gamma_low_err, r0, gamma) - gamma)
    sigma3_lower_left = gamma-1*(ellipse(r0_range_left3, 3*r0_low_err, 3*gamma_low_err, r0, gamma) - gamma)

    sigma1 = np.concatenate([sigma1_upper_right, sigma1_lower_right[::-1], sigma1_lower_left[::-1], sigma1_upper_left])
    sigma2 = np.concatenate([sigma2_upper_right, sigma2_lower_right[::-1], sigma2_lower_left[::-1], sigma2_upper_left])
    sigma3 = np.concatenate([sigma3_upper_right, sigma3_lower_right[::-1], sigma3_lower_left[::-1], sigma3_upper_left])

    r0_range_1[r0_range_1<0] = 0
    r0_range_2[r0_range_2<0] = 0
    r0_range_3[r0_range_3<0] = 0

    sigma1[sigma1<0] = 0
    sigma2[sigma2<0] = 0
    sigma3[sigma3<0] = 0

    return r0_range_1, sigma1, r0_range_2, sigma2, r0_range_3, sigma3

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

    parser = argparse.ArgumentParser(description='Plot the power spectrum of the halo field')
    parser.add_argument('-z', type=float, default=6.6, help='Redshift of the halo field')

    args = parser.parse_args()

    z = args.z

    import h5py

    with h5py.File('../data/halo_fields/HaloField_4726d8c02a89e2cf56ccacd64f593423_r1.h5', 'r') as f:
        halo_coords = f['HaloField']['halo_coords'][()]
        halo_masses = f['HaloField']['halo_masses'][()]

    mass_filter = halo_masses > 1e10
    # random_filter = np.random.choice([True, False], len(mass_filter), p=[0.1, 0.9])
    random_filter = np.ones(len(mass_filter), dtype=bool)

    halo_masses = halo_masses[mass_filter*random_filter]
    x_halo = halo_coords[:, 0][mass_filter*random_filter]
    y_halo = halo_coords[:, 1][mass_filter*random_filter]
    z_halo = halo_coords[:, 2][mass_filter*random_filter]

    SIDE_LENGTH_MPC = 300

    box = np.zeros((int(SIDE_LENGTH_MPC), int(SIDE_LENGTH_MPC), int(SIDE_LENGTH_MPC)))
    for _x, _y, _z in zip(x_halo, y_halo, z_halo):
        box[int(_x), int(_y), int(_z)] += 1

    box = box/np.mean(box) - 1
    ps, k, pvar = powerbox.get_power(box, boxlength=int(SIDE_LENGTH_MPC),\
                log_bins=True, get_variance=True, ignore_zero_mode=True,\
                vol_normalised_power=True)

    ps = ps[~np.isnan(k)]
    pvar = pvar[~np.isnan(k)]
    k = k[~np.isnan(k)]

    if z == 6.6:
        r0 = 8
        r0_upper = 8 + 1.9
        r0_lower = 8 - 5.8
        gamma = 1.4
        gamma_upper = 1.4 + 0.58
        gamma_lower = 1.4 - 0.88
    elif z == 5.7:
        r0 = 4
        r0_upper = r0 + 0.6
        r0_lower = r0 - 0.7
        gamma = 1.4
        gamma_upper = gamma + 0.17
        gamma_lower = gamma - 0.17

    m_dat = gamma
    h = 0.7

    b_dat = h*np.log10((2/np.pi)*r0**gamma*special.gamma(2-gamma)*np.sin(np.pi*gamma/2))

    def data_line(k):
        return 10**(m_dat*np.log10(k) + b_dat)

    k_linear = np.linspace(1.1*k.max(), 0.9*k.min(), 100)

    d2 = ps*k**3/(2*np.pi**2)
    d2_err = pvar*k**3/(2*np.pi**2)
    k_max = 2*np.pi / 10

    m, b, m_err, b_err = get_line_params(np.log10(k[k<k_max]), np.log10(d2[k<k_max]), \
                                        d2_err[k<k_max]/(np.log10(d2[k<k_max])*np.log(10)))

    gamma_fit = m
    r0_fit = ((np.pi*10**b)/(2*special.gamma(2-gamma_fit)*\
                             np.sin(np.pi*gamma_fit/2)))**(1/gamma_fit)/h
    
    print(m, b)
    print(gamma_fit, r0_fit)
    print(gamma, r0)
    # quit()

    fig, axs = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)

    axs.errorbar(k, d2, yerr=d2_err, fmt='o', color='cyan', label='fiducial model')

    axs.plot(k_linear, data_line(k_linear), label='Subaru', color='white')
    axs.plot(k_linear, 10**(m*np.log10(k_linear) + b), label='best fit', color='white', linestyle='--')

    axs.axvline(2*np.pi/10, color='white', linestyle=':', label=r'$10$ Mpc')

    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.set_xlim(k_linear[-1], k_linear[0])
    # axs.set_title(f'Power Spectrum at z={z}')
    axs.set_xlabel(r'$k$ [Mpc$^{-1}$]', fontsize=font_size)
    axs.set_ylabel(r'$\Delta^2(k)$', fontsize=font_size)
    axs.legend(fontsize=font_size)

    # r0_range_1, sigma1, r0_range_2, sigma2, r0_range_3, sigma3 \
    #     = get_bounds(r0, r0_up_err, r0_low_err, gamma, gamma_up_err, gamma_low_err)

    # ax_pdf = axs.inset_axes([0.5, 0, 0.5, 0.4])
    # # ax_pdf.set_xticklabels([])

    # ax_pdf.plot(r0, gamma, 'x', color='cyan', markersize=10)
    # ax_pdf.plot(r0_fit, gamma_fit, 'x', color='white', markersize=10)

    # ax_pdf.plot(r0_range_1, sigma1, color='cyan', linestyle='solid', linewidth=2)
    # ax_pdf.plot(r0_range_2, sigma2, color='cyan', linestyle='dashed', linewidth=2)
    # ax_pdf.plot(r0_range_3, sigma3, color='cyan', linestyle='dotted', linewidth=2)

    # ax_pdf.set_ylabel(r'$\gamma$', fontsize=font_size)
    # ax_pdf.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    # ax_pdf.set_title(r'$r_0$ [h$_{70}^{-1}$ Mpc]', fontsize=font_size)

    plt.show()