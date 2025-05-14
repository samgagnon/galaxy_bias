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

def get_halo_pos():
    halo_field = np.load('../data/halo_fields/halo_field_6.61.npy')

    X, Y, Z, _, _, sfr = halo_field

    # apply muv cut
    muv = get_muv(sfr)
    muv_lim = -18

    Z -= Z.min()

    ZLIM = 5

    boxsize = 300.0
    boxlen = 200.0

    X = X[(muv < muv_lim)*(Z<ZLIM)]*boxsize/boxlen
    Y = Y[(muv < muv_lim)*(Z<ZLIM)]*boxsize/boxlen
    Z = Z[(muv < muv_lim)*(Z<ZLIM)]*boxsize/boxlen

    return boxsize, X, Y

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

    # function gives halo positions in units of cMpc
    boxsize, X, Y = get_halo_pos()

    # Number of threads to use
    nthreads = 2

    # convert from Mpc to RA, DEC
    # using Planck18 cosmology
    d_a = Planck18.angular_diameter_distance(6.6).to(u.Mpc).value
    RA1 = (X / d_a)  * 180 / np.pi
    DEC1 = (Y / d_a) * 180 / np.pi - 90
    RA1 = RA1.astype(np.float64)
    DEC1 = DEC1.astype(np.float64)

    # generate Poisson random points
    X2 = np.random.randint(0, 200, size=len(RA1)).astype(RA1.dtype)*300/200
    Y2 = np.random.randint(0, 200, size=len(DEC1)).astype(DEC1.dtype)*300/200
    RA2 = (X2 / d_a)  * 180 / np.pi
    DEC2 = (Y2 / d_a) * 180 / np.pi - 90

    # edges of bins in units of degrees
    binfile = np.logspace(-3, 1, 11)

    # compute number of pairs for data-data, data-random, random-random
    # using the DDtheta_mocks function
    results_dd = DDtheta_mocks(autocorr=1, nthreads=nthreads, \
                                binfile=binfile, RA1=RA1, DEC1=DEC1)
    
    results_dr = DDtheta_mocks(autocorr=0, nthreads=nthreads, \
                                binfile=binfile, RA1=RA1, DEC1=DEC1, \
                                RA2=RA2, DEC2=DEC2)
    
    results_rr = DDtheta_mocks(autocorr=1, nthreads=nthreads, \
                                binfile=binfile, RA1=RA2, DEC1=DEC2)
    
    # convert to correlation function using Landy-Szalay estimator
    # w(theta) = (DD - 2DR + RR) / RR
    wtheta = convert_3d_counts_to_cf(len(DEC1), len(DEC1), \
                                     len(DEC1), len(DEC1),
                                results_dd, results_dr,
                                results_dr, results_rr)
    
    # plot
    bin_centers = 0.5*(binfile[1:] + binfile[:-1])
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.plot(bin_centers*3600, wtheta, '*', markersize=10, color='cyan', label='w(theta)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_ticks([], minor=True)
    ax.xaxis.set_ticks([1, 10, 100, 1000, 10000])
    ax.xaxis.set_ticklabels([r'$1.0$', r'$10.0$', r'$100.0$', r'$1000.0$', r'$10000.0$'])
    ax.set_xlabel(r'$\theta$ [arcsec]', fontsize=font_size)
    ax.set_ylabel(r'$w(\theta)$', fontsize=font_size)
    plt.show()
    