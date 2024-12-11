import numpy as np

import py21cmfast as p21c

import powerbox

import logging, os, json
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('21cmFAST')

from scipy import integrate
from scipy import special
from astropy.cosmology import Planck18

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')

    import argparse

    parser = argparse.ArgumentParser(description='Plot the power spectrum of the halo field')
    parser.add_argument('-z', type=float, default=6.6, help='Redshift of the halo field')

    args = parser.parse_args()

    z = args.z

    halo_dir = 'data/halo_fields/'
    halo_fn_list = os.listdir(halo_dir)
    proceed = False
    for halo_fn in halo_fn_list:
        halo_field = p21c.HaloField.from_file(halo_dir + halo_fn)
        if halo_field.redshift == z:
            proceed = True
            break
    if not proceed:
        logger.error(f"No halo field found at redshift {z}")
        quit()
    SIDE_LENGTH_MPC = halo_field.user_params.BOX_LEN
    comoving_coords = np.linspace(0, SIDE_LENGTH_MPC, int(SIDE_LENGTH_MPC))
    N_PIXELS = halo_field.user_params.HII_DIM
    z = halo_field.redshift
    print(f"Box length: {SIDE_LENGTH_MPC} Mpc")
    print(f"Number of pixels: {N_PIXELS}")

    mass_filter = halo_field.halo_masses > 1e10
    random_filter = np.random.choice([True, False], len(mass_filter), p=[0.1, 0.9])
    random_filter = np.ones(len(mass_filter), dtype=bool)

    halo_masses = halo_field.halo_masses[mass_filter*random_filter]
    x_halo = halo_field.halo_coords[:, 0][mass_filter*random_filter]
    y_halo = halo_field.halo_coords[:, 1][mass_filter*random_filter]
    z_halo = halo_field.halo_coords[:, 2][mass_filter*random_filter]

    # sky plane image
    # image = np.zeros((int(SIDE_LENGTH_MPC), int(SIDE_LENGTH_MPC)))
    # for x, y in zip(x_halo, y_halo):
    #     image[int(x), int(y)] += 1

    box = np.zeros((int(SIDE_LENGTH_MPC), int(SIDE_LENGTH_MPC), int(SIDE_LENGTH_MPC)))
    for _x, _y, _z in zip(x_halo, y_halo, z_halo):
        box[int(_x), int(_y), int(_z)] += 1

    # image = image / image.mean() - 1

    # plt.figure(figsize=(8, 8))
    # plt.pcolormesh(comoving_coords, comoving_coords, \
    #             image, cmap='hot')
    # plt.colorbar()
    # plt.title('Halo distribution')
    # plt.xlabel('x [Mpc]')
    # plt.ylabel('y [Mpc]')
    # plt.show()
    # plt.close()

    # ps, k, pvar = powerbox.get_power(image, boxlength=int(SIDE_LENGTH_MPC),\
    #             log_bins=True, get_variance=True, ignore_zero_mode=True,\
    #             vol_normalised_power=True)

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

    # print(intercept_up_err, intercept_low_err, intercept)
    # quit()

    def best_fit(k):
        return 10**(slope*np.log10(k) + intercept)

    def best_fit_upper(k):
        return 10**((slope)*np.log10(k) + intercept + intercept_up_err)

    def best_fit_lower(k):
        return 10**((slope)*np.log10(k) + intercept - intercept_low_err)

    # print(np.sqrt((r0_up_rel_err/np.log(10)/np.log10(r0))**2 + gamma_up_rel_err**2)*gamma*np.log10(r0))
    # print(gamma_up_err)
    # print(gamma_up_special_err/gamma/np.log(10))
    # quit()

    # print(intercept_up_err, intercept_low_err, intercept)
    # quit()

    k_linear = np.linspace(1.1*k.max(), 0.9*k.min(), 100)

    # best_fit = np.abs(np.real(1j**gamma))*(np.pi/2)**(1/2)*r0**(-1*gamma)*k_linear**(gamma - 1) / special.gamma(gamma)
    # best_fit_upper = np.abs(np.real(1j**gamma))*(np.pi/2)**(1/2)*r0_upper**(-1*gamma_upper)*k_linear**(gamma_upper - 1) / special.gamma(gamma_upper)
    # best_fit_lower = np.abs(np.real(1j**gamma))*(np.pi/2)**(1/2)*r0_lower**(-1*gamma_lower)*k_linear**(gamma_lower - 1) / special.gamma(gamma_lower)

    plt.figure(figsize=(8, 8))
    # plt.errorbar(k, ps, yerr=pvar, fmt='o', color='cyan', label='This Work')
    d2 = ps*k**3/(2*np.pi**2)
    d2_err = pvar*k**3/(2*np.pi**2)
    plt.errorbar(k, d2, yerr=d2_err, fmt='o', color='cyan', label='This Work')

    # best_fit *= k_linear/(2*np.pi**2)
    # best_fit_upper *= k_linear/(2*np.pi**2)
    # best_fit_lower *= k_linear/(2*np.pi**2)
    plt.plot(k_linear, best_fit(k_linear)*k_linear/(2*np.pi**2), label='Umeda+24', color='white')
    plt.fill_between(k_linear, best_fit_upper(k_linear)*k_linear/(2*np.pi**2), best_fit_lower(k_linear)*k_linear/(2*np.pi**2), color='white', alpha=0.5)

    # plt.axvline(2*np.pi / (SIDE_LENGTH_MPC), color='white', linestyle='--', label='Nyquist frequency')
    # plt.axvline(arcsecond_voxel, color='white', linestyle='--', label='Voxel size')
    plt.axvline(2*np.pi/(3*1.5**2)**(1/2), color='white', linestyle='--', label='Voxel scale')
    plt.axvline(2*np.pi/(3*3**2)**(1/2), color='white', linestyle='-.', label='2x Voxel scale')
    plt.axvline(2*np.pi/10, color='white', linestyle=':', label=r'$10$ Mpc')
    # plt.axvline(4*np.pi/3**(3), color='white', linestyle='-.', label='2x Voxel scale')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(k_linear[-1], k_linear[0])
    plt.title(f'Power Spectrum at z={z}')
    # plt.xlabel(r'$\theta$ [arcsec]')
    plt.xlabel(r'$k$ [Mpc$^{-1}$]')
    plt.ylabel(r'$\Delta^2(k)$')
    plt.legend()
    plt.show()