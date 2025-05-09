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
    initial_guess = [-1, -2]

    # Minimize the objective function
    result = minimize(objective, initial_guess, bounds=[(-4, 0), (-5, 0)])
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

    gamma_up_err = gamma_upper - gamma
    gamma_low_err = gamma - gamma_lower
    r0_up_err = r0_upper - r0
    r0_low_err = r0 - r0_lower

    gamma_up_rel_err = gamma_up_err / gamma
    gamma_low_rel_err = gamma_low_err / gamma
    r0_up_rel_err = r0_up_err / r0
    r0_low_rel_err = r0_low_err / r0

    gamma_up_special_err = special.gamma(gamma_upper) - special.gamma(gamma)
    gamma_low_special_err = special.gamma(gamma) - special.gamma(gamma_lower)
    r0_up_special_err = special.gamma(r0_upper) - special.gamma(r0)
    r0_low_special_err = special.gamma(r0) - special.gamma(r0_lower)

    slope_up_err = gamma_up_err
    slope_low_err = gamma_low_err

    slope = gamma - 1

    intercept_up_err  = np.sqrt(gamma_up_rel_err**2 + (r0_up_rel_err/np.log(10)/np.log10(r0))**2)\
        *gamma*(np.abs(np.log10(1j) - np.log10(r0))) + gamma_up_special_err/gamma/np.log(10)
    intercept_low_err = np.sqrt(gamma_low_rel_err**2 + (r0_low_rel_err/np.log(10)/np.log10(r0))**2)\
        *gamma*(np.abs(np.log10(1j) - np.log10(r0))) + gamma_low_special_err/gamma/np.log(10)

    intercept = np.log10(np.abs(np.real(1j**gamma))*(np.pi/2)**(1/2)*r0**(-1*gamma) / special.gamma(gamma))

    def best_fit(k):
        return 10**(slope*np.log10(k) + intercept)

    def best_fit_upper(k):
        return 10**((slope)*np.log10(k) + intercept + intercept_up_err)

    def best_fit_lower(k):
        return 10**((slope)*np.log10(k) + intercept - intercept_low_err)

    k_linear = np.linspace(1.1*k.max(), 0.9*k.min(), 100)

    d2 = ps*k**3/(2*np.pi**2)
    d2_err = pvar*k**3/(2*np.pi**2)

    k_max = 2*np.pi / 10
    m, b, m_err, b_err = get_line_params(np.log10(k[k<k_max]), np.log10(ps[k<k_max]), \
                                        pvar[k<k_max]/(np.log10(ps[k<k_max])*np.log(10)))

    gamma_fit = m + 3
    # not positive about whether to multiply h70 here
    r0_fit = 0.7*10**(-1*b/gamma_fit + np.real(np.log10(1j)) + \
                  np.log10(np.pi/2)/2/gamma_fit - np.log10(special.gamma(gamma_fit))/gamma_fit)
    
    gamma_fit_err = m_err
    r0_fit_err = r0_fit * np.sqrt((b_err/b)**2 + (gamma_fit_err/gamma_fit)**2)
    
    if gamma_fit > gamma:
        gamma_err = np.sqrt(gamma_up_err**2 + gamma_fit_err**2)
    elif gamma_fit < gamma:
        gamma_err = np.sqrt(gamma_low_err**2 + gamma_fit_err**2)
    if r0_fit > r0:
        r0_err = np.sqrt(r0_up_err**2 + r0_fit_err**2)
    elif r0_fit < r0:
        r0_err = np.sqrt(r0_low_err**2 + r0_fit_err**2)

    chi_squared = (gamma_fit - gamma)**2 / gamma_err**2 + (r0_fit - r0)**2 / r0_err**2
    std_dev = np.sqrt(chi_squared)
    probability = 0.5*(special.erf(std_dev / 2**(1/2)) - special.erf(-1*std_dev / 2**(1/2)))

    fig, axs = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)

    axs.errorbar(k, d2, yerr=d2_err, fmt='o', color='cyan', label='fiducial model')

    axs.plot(k_linear, best_fit(k_linear)*k_linear/(2*np.pi**2), label='Subaru', color='white')
    axs.plot(k_linear, 10**(m*np.log10(k_linear) + b)*k_linear**3/(2*np.pi**2), label='best fit', color='white', linestyle='--')

    axs.axvline(2*np.pi/10, color='white', linestyle=':', label=r'$10$ Mpc')

    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.set_xlim(k_linear[-1], k_linear[0])
    # axs.set_title(f'Power Spectrum at z={z}')
    axs.set_xlabel(r'$k$ [Mpc$^{-1}$]', fontsize=font_size)
    axs.set_ylabel(r'$\Delta^2(k)$', fontsize=font_size)
    axs.legend(fontsize=font_size)

    r0_range_1, sigma1, r0_range_2, sigma2, r0_range_3, sigma3 \
        = get_bounds(r0, r0_up_err, r0_low_err, gamma, gamma_up_err, gamma_low_err)

    ax_pdf = axs.inset_axes([0.5, 0, 0.5, 0.4])
    # ax_pdf.set_xticklabels([])

    ax_pdf.plot(r0, gamma, 'x', color='cyan', markersize=10)
    ax_pdf.plot(r0_fit, gamma_fit, 'x', color='white', markersize=10)

    ax_pdf.plot(r0_range_1, sigma1, color='cyan', linestyle='solid', linewidth=2)
    ax_pdf.plot(r0_range_2, sigma2, color='cyan', linestyle='dashed', linewidth=2)
    ax_pdf.plot(r0_range_3, sigma3, color='cyan', linestyle='dotted', linewidth=2)

    ax_pdf.set_ylabel(r'$\gamma$', fontsize=font_size)
    ax_pdf.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax_pdf.set_title(r'$r_0$ [h$_{70}^{-1}$ Mpc]', fontsize=font_size)

    # ax_pdf.plot(get_intercept(r0, gamma), get_slope(gamma), 'x', color='cyan', markersize=10)
    # ax_pdf.plot(get_intercept(r0_fit, gamma_fit), get_slope(gamma_fit), 'x', color='white', markersize=10)

    # this turns out to be useless, revert to the original
    # intercept1 = np.concatenate([get_intercept(r0_range_1, sigma1), get_intercept(r0_range_1, sigma1)[0:2]])
    # intercept2 = np.concatenate([get_intercept(r0_range_2, sigma2), get_intercept(r0_range_2, sigma2)[0:2]])
    # intercept3 = np.concatenate([get_intercept(r0_range_3, sigma3), get_intercept(r0_range_3, sigma3)[0:2]])
    # sigma1 = np.concatenate([get_slope(sigma1), get_slope(sigma1)[0:2]])
    # sigma2 = np.concatenate([get_slope(sigma2), get_slope(sigma2)[0:2]])
    # sigma3 = np.concatenate([get_slope(sigma3), get_slope(sigma3)[0:2]])

    # ax_pdf.plot(intercept1, sigma1, color='cyan', linestyle='solid', markersize=10)
    # ax_pdf.plot(intercept2, sigma2, color='cyan', linestyle='dashed', markersize=10)
    # ax_pdf.plot(intercept3, sigma3, color='cyan', linestyle='dotted', markersize=10)

    # ax_pdf.plot(get_intercept(0, sigma1), get_slope(sigma1), '--', color='cyan', markersize=10)

    # ax_pdf.set_ylabel(r'$\gamma-1$', fontsize=font_size)
    # ax_pdf.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    # ax_pdf.set_title(r'$\log_{10}k_0$ [Mpc$^{-1}$]', fontsize=font_size)

    plt.show()