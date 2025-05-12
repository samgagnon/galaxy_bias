import h5py
import numpy as np
from Corrfunc.io import read_catalog

from astropy.cosmology import Planck18
from astropy import units as u

def get_acf_silverrush(z):
    if z == 6.6:
        # z=6.6
        theta      = np.array([18.76, 36.09, 73.01, 144.04, 291.39, 574.86, 1134.08]) # arcmin
        w_p        = np.array([0.27, 0.93, 1.12, 1.22, 0.43, 0.09, 0.10])
        w_p_up_lim = np.array([1.54, 2.32, 2.55, 1.85, 0.85, 0.39, 0.38])
        w_p_lo_lim = np.array([0.0,0.0,0.0,0.57,0.02,0.0,0.0])

    elif z == 5.7:
        # z=5.7
        theta      = np.array([0.35, 0.57, 0.94, 1.58, 2.63, 7.53, 12.55, 21.54, 34.93, 61.68, 102.88, 171.62, 294.54, 491.31, 819.56, 1367.71])
        w_p        = np.array([87.09, 72.44, 63.09, 9.12, 2.19, 1.91, 0.52, 1.58, 0.36, 0.40, 0.32, 0.36, 0.30, 0.17, 0.14, 0.08])
        w_p_up_lim = np.array([114.82, 125.89, 104.71, 15.14, 4.17, 2.63, 0.87, 2.29, 0.58, 0.55, 0.50, 0.48, 0.44, 0.30, 0.28, 0.22])
        w_p_lo_lim = np.array([30.20, 28.84, 22.91, 3.02, 0.33, 1.04, 0.09, 1.05, 0.16, 0.24, 0.17, 0.22, 0.14, 0.02, 0.0, 0.0])

    elif z == 4.9:
        # z=4.9
        theta      = np.array([4.82, 7.58, 12.54, 20.23, 31.82, 48.81, 82.80, 123.85, 204.87, 322.25, 519.81])
        w_p        = np.array([12.56, 2.55, 2.22, 0.189, 0.98, 0.68, 0.71, 0.18, 0.16, 0.13, 0.07, 0.0])
        w_p_up_lim = np.array([18.93, 4.61, 3.84, 0.98, 1.41, 1.17, 0.85, 0.43, 0.23, 0.16, 0.08, 0.03])
        w_p_lo_lim = np.array([4.20, 0.16, 0.89, 0.0, 0.27, 0.18, 0.57, 0.0, 0.10, 0.10, 0.01, 0.0])
    
    return theta, w_p, w_p_up_lim, w_p_lo_lim

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

    use_data = True

    if use_data:

        with h5py.File('../data/halo_fields/HaloField_4726d8c02a89e2cf56ccacd64f593423_r1.h5', 'r') as f:
            halo_coords = f['HaloField']['halo_coords'][()]
            halo_masses = f['HaloField']['halo_masses'][()]

            mass_filter = halo_masses > 1e10
            # random_filter = np.random.choice([True, False], len(mass_filter), p=[0.1, 0.9])
            random_filter = np.ones(len(mass_filter), dtype=bool)

            halo_masses = halo_masses[mass_filter*random_filter]
            X = halo_coords[:, 0][mass_filter*random_filter].astype(np.float32)
            Y = halo_coords[:, 1][mass_filter*random_filter].astype(np.float32)
            Z = halo_coords[:, 2][mass_filter*random_filter].astype(np.float32)

        boxsize = 300.0

    else:
        # Read the default galaxies supplied with
        # Corrfunc. ~ 1 million galaxies on a 420 Mpc/h cube
        X, Y, Z = read_catalog()
        # Specify boxsize for the XYZ arrays
        boxsize = 420.0

    # Number of threads to use
    nthreads = 2

    # load the silverrush data
    theta_silverrush, w_p_silverrush, w_p_up_lim_silverrush, \
        w_p_lo_lim_silverrush = get_acf_silverrush(6.6)
    lower_error = w_p_silverrush - w_p_lo_lim_silverrush
    upper_error = w_p_up_lim_silverrush - w_p_silverrush
    assym_err = np.array(list(zip(lower_error, upper_error))).T
    # Create the bins array
    rbin_centers = (theta_silverrush * (1 + 6.6) * (np.pi/180) / 3600.0) \
        * Planck18.angular_diameter_distance(6.6).to(u.Mpc).value
    rbin_separations = np.diff(rbin_centers)
    rbin_edges = np.concatenate(([rbin_centers[0] - rbin_separations[0] / 2], \
                                rbin_centers[1:] - rbin_separations / 2,\
                                [rbin_centers[-1] + rbin_separations[-1] / 2]))
    rbins = rbin_edges

    # Specify the distance to integrate along line of sight
    pimax = 20.0 # what is this?
    # Specify that an autocorrelation is wanted
    autocorr = 1

    from Corrfunc.theory.wp import wp
    results_wp = wp(boxsize, pimax, nthreads, rbins, X, Y, Z)

    ravg_w = 0.5 * (results_wp['rmin'] + results_wp['rmax'])

    d_a = Planck18.angular_diameter_distance(6.6).to(u.Mpc).value
    print(ravg_w, d_a)
    # quit()
    theta = ravg_w / d_a / (1 + 6.6)
    theta *= 180 / np.pi
    theta *= 3600  # arcmin

    # NOTE no uncertainty in the xi values

    fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    # axs.plot(ravg, results_xi['xi'], 'oc', label='xi')
    axs.plot(theta, results_wp['wp'], 'or', label='wp')
    axs.errorbar(theta_silverrush, w_p_silverrush, yerr=assym_err,
                fmt='o', color='cyan', label='wp silverrush')
    axs.set_xscale('log')
    axs.set_yscale('log')
    # axs.set_xlabel(r'$r$ [Mpc/h]', fontsize=font_size)
    axs.set_xlabel(r'$\theta$ [arcsec]', fontsize=font_size)
    # axs.set_ylabel(r'$\xi(r)$', fontsize=font_size)
    axs.set_ylabel(r'$\omega_p(r)$', fontsize=font_size)
    # axs.legend()
    plt.show()