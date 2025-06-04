import h5py
import numpy as np
from Corrfunc.io import read_catalog
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from Corrfunc.utils import convert_3d_counts_to_cf

from astropy.cosmology import Planck18
from astropy import units as u

def get_muv(sfr):
    """
    Converts star formation rate to
    absolute UV magnitude.
    """
    kappa = 1.15e-28
    luv = sfr * 3.1557e7 / kappa
    muv = 51.64 - np.log10(luv) / 0.4
    return muv

def get_bin_edges(theta):
    """
    Get bin edges for the correlation function
    """
    binfile = np.zeros(len(theta) + 1)
    diff = np.diff(np.log10(theta)) / 2
    binfile[0] = 10**(np.log10(theta[0]) - diff[0])
    binfile[1:-1] = 10**(np.log10(theta[:-1]) + diff)
    binfile[-1] = 10**(np.log10(theta[-1]) + diff[-1])
    return binfile

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
        theta      = np.array([4.82, 7.58, 12.54, 20.23, 31.82, 48.81, 82.80, 123.85, 204.87, 322.25, 519.81, 747.16])
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

        halo_field = np.load('../data/halo_fields/halo_field_6.61.npy')

        X, Y, Z, _, _, sfr = halo_field

        # apply muv cut
        muv = get_muv(sfr)
        muv_lim = -18

        Z -= Z.min()

        ZLIM = 5

        _x, _y, _z = X[(muv < muv_lim)]*300/200, \
            Y[(muv < muv_lim)]*300/200, Z[(muv < muv_lim)]*300/200

        X = X[(muv < muv_lim)*(Z<ZLIM)]*300/200
        Y = Y[(muv < muv_lim)*(Z<ZLIM)]*300/200
        Z = Z[(muv < muv_lim)*(Z<ZLIM)]*300/200

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

    d_a = Planck18.angular_diameter_distance(6.6).to(u.Mpc).value
    RA1 = (X / d_a)  * 180 / np.pi
    DEC1 = (Y / d_a) * 180 / np.pi - 90
    RA1 = RA1.astype(np.float64)
    DEC1 = DEC1.astype(np.float64)

    X2 = np.random.randint(0, 200, size=len(RA1)).astype(RA1.dtype)*300/200
    Y2 = np.random.randint(0, 200, size=len(DEC1)).astype(DEC1.dtype)*300/200
    RA2 = (X2 / d_a)  * 180 / np.pi
    DEC2 = (Y2 / d_a) * 180 / np.pi - 90

    # fig, axs = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)
    # axs[0].scatter(RA1, DEC1, s=0.1, color='cyan', label='21cmFAST')
    # axs[1].scatter(RA2, DEC2, s=0.1, color='cyan', label='Poisson')
    # axs[0].set_xlabel(r'RA [deg]', fontsize=font_size)
    # axs[0].set_ylabel(r'DEC [deg]', fontsize=font_size)
    # axs[0].legend()
    # axs[1].set_xlabel(r'RA [deg]', fontsize=font_size)
    # axs[1].set_ylabel(r'DEC [deg]', fontsize=font_size)
    # axs[1].legend()
    # plt.show()
    # quit()

    # POWER SPECTRUM NOTE not used
    # da_box = (np.array([_x, _y, _z])*200/300).astype(np.int32)
    # da_vox = np.zeros((200, 200, 200))
    # np.add.at(da_vox, (da_box[0], da_box[1], da_box[2]), 1)
    # p3, k3 = get_power(da_vox, boxlength=300, log_bins=True)

    # sus_box = np.random.randint(0, 200, size=(3, len(_x))).astype(da_box.dtype)
    # sus_vox = np.zeros((200, 200, 200))
    # np.add.at(sus_vox, (sus_box[0], sus_box[1], sus_box[2]), 1)
    # p4, k4 = get_power(sus_vox, boxlength=300, log_bins=True)

    # voxel_scale = (1.5 / d_a) * 180 / np.pi * 3600 / 10
    voxel_scale = 0.0

 
    binfile = get_bin_edges(theta_silverrush[theta_silverrush>voxel_scale])/3600

    results_dd = DDtheta_mocks(autocorr=1, nthreads=nthreads, \
                                binfile=binfile, RA1=RA1, DEC1=DEC1)
    
    results_dr = DDtheta_mocks(autocorr=0, nthreads=nthreads, \
                                binfile=binfile, RA1=RA1, DEC1=DEC1, \
                                RA2=RA2, DEC2=DEC2)
    
    results_rr = DDtheta_mocks(autocorr=1, nthreads=nthreads, \
                                binfile=binfile, RA1=RA2, DEC1=DEC2)
    
    dd = np.array(results_dd['npairs'])
    dr = np.array(results_dr['npairs'])
    rr = np.array(results_rr['npairs'])

    wtheta = convert_3d_counts_to_cf(len(DEC1), len(DEC1), \
                                     len(DEC1), len(DEC1),
                                results_dd, results_dr,
                                results_dr, results_rr)

    fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

    axs.errorbar(theta_silverrush, w_p_silverrush, yerr=assym_err, \
                fmt='o', markersize=10, capsize=5, \
                color='cyan', label='Umeda et al. (2024)')
    
    axs.plot(theta_silverrush[theta_silverrush>voxel_scale], wtheta, '*', \
                markersize=20, color='red', label='21cmFAST')

    # axs.axvline(voxel_scale, color='orange', linestyle='--', label='voxel scale')
    # axs.axvline(coeval_scale, color='green', linestyle='--', label='coeval scale')
    axs.set_xscale('log')
    # axs.set_yscale('log')
    # axs.set_xlabel(r'$r$ [Mpc/h]', fontsize=font_size)
    axs.set_xlabel(r'$\theta$ [arcsec]', fontsize=font_size)
    # axs.set_ylabel(r'$\xi(r)$', fontsize=font_size)
    axs.set_ylabel(r'$\omega(\theta)$', fontsize=font_size)
    axs.legend()
    plt.show()

    # compute likelihood
    up_or_lo = (wtheta > w_p_silverrush).astype(np.int32)
    errs = np.ones_like(up_or_lo)
    errs = assym_err[up_or_lo, range(len(up_or_lo))]
    print(assym_err)
    print(errs)
    
    # NOTE
    # there is significant stochasticity in the likelihood between
    # noise realizations -- indication of unquantified errors
    # in the simulation. Jackknife?
    print(np.log(1/(np.sqrt(2*np.pi)*errs) * \
        np.exp(-0.5*((wtheta - w_p_silverrush)/errs)**2)))
    log_likelihood = np.sum(((wtheta - w_p_silverrush)/errs)**2)
    print(f'log-likelihood: {log_likelihood:.2f}')