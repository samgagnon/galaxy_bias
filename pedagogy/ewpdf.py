import numpy as np

from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, G, m_p, k_B

from scipy import integrate

def get_silverrush_ewpdf(z):
    assert z in [2.2, 3.3, 4.9, 5.7, 6.6]
    if z == 2.2:
        ew0_low = 58.2
        ew0_low_upper = 1.4
        ew0_low_lower = 1.4
        ew0_high = 110.8
        ew0_high_upper = 1.0
        ew0_high_lower = 0.9
    elif z == 3.3:
        ew0_low = 43.9
        ew0_low_upper = 1.1
        ew0_low_lower = 1.1
        ew0_high = 45.0
        ew0_high_upper = 0.7
        ew0_high_lower = 0.8
    elif z==4.9:
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

    # supply redshift as an argument
    z_list = [6.6, 4.9, 5.7]
    color_list = ['red', 'orange', 'yellow']
    alpha_list = [0.7, 0.7, 0.7]

    # first show the PDFs

    fig, axs = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)

    # SILVERRUSH XIV
    for z, c, a in zip(z_list, color_list, alpha_list):
        ew0_low, ew0_low_upper, ew0_low_lower, \
        ew0_high, ew0_high_upper, ew0_high_lower \
            = get_silverrush_ewpdf(z)

        ew_base = ew0_high - ew0_low

        ew_dist_base = np.random.exponential(ew_base, 10000)
        ew_dist_low = np.random.exponential(ew0_low, 10000)

        ew_dist = ew_dist_base + ew_dist_low

        if z == 6.6:
            bins, edges = np.histogram(ew_dist, bins=10)
            bin_centers = edges[:-1] + np.diff(edges)/2
            bin_width = np.diff(edges)
        axs.hist(ew_dist[ew_dist>40], bins=bin_centers, density=True, \
                 color=c, alpha=a, label=f'z={z}')

        # print('ew mean', ew_dist[ew_dist>40].mean(), ew_dist.mean(), ew0_high, ew0_low)

    axs.set_yscale('log')
    axs.set_xlim(40, bin_centers[-1])

    axs.set_xlabel(r'$W^{\rm emerg}_{\rm Ly\alpha}$ [$\AA$]', fontsize=font_size)
    axs.set_ylabel(r'PDF', fontsize=font_size)
    # axs.grid()
    axs.legend()

    plt.show()

    # second show the e folding scales

    fig, axs = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)

    z_list = [2.2, 3.3, 4.9, 5.7, 6.6]

    ew0_low_list = []
    ew0_low_upper_list = []
    ew0_low_lower_list = []
    ew0_high_list = []
    ew0_high_upper_list = []
    ew0_high_lower_list = []
    # SILVERRUSH XIV
    for z in z_list:
        ew0_low, ew0_low_upper, ew0_low_lower, \
        ew0_high, ew0_high_upper, ew0_high_lower \
            = get_silverrush_ewpdf(z)
        
        ew0_low_list.append(ew0_low)
        ew0_low_upper_list.append(ew0_low_upper)
        ew0_low_lower_list.append(ew0_low_lower)
        ew0_high_list.append(ew0_high)
        ew0_high_upper_list.append(ew0_high_upper)
        ew0_high_lower_list.append(ew0_high_lower)

    ew0_low_list = np.array(ew0_low_list)
    ew0_low_upper_list = np.array(ew0_low_upper_list)
    ew0_low_lower_list = np.array(ew0_low_lower_list)
    ew0_high_list = np.array(ew0_high_list)
    ew0_high_upper_list = np.array(ew0_high_upper_list)
    ew0_high_lower_list = np.array(ew0_high_lower_list)
    z_list = np.array(z_list)

    ew0_low_err = np.array([ew0_low_upper_list, ew0_low_lower_list])
    ew0_high_err = np.array([ew0_high_upper_list, ew0_high_lower_list])

    axs.errorbar(z_list[:-1], ew0_high_list[:-1], yerr=ew0_high_err[:,:-1], fmt='o', \
                marker='v', markersize=10, capsize=20, color='cyan', \
                label=r'$W^{\rm emerg}_{\rm Ly\alpha}\in[40,1000]$ [$\AA$]')
    axs.errorbar(z_list[:-1], ew0_low_list[:-1], yerr=ew0_low_err[:,:-1], fmt='o', \
                marker='^', markersize=10, capsize=20, color='cyan', \
                label=r'$W^{\rm emerg}_{\rm Ly\alpha}\in[40,200]$ [$\AA$]')
    axs.errorbar(z_list[-1], ew0_high_list[-1], yerr=ew0_high_err[:,-1].reshape((2,1)), fmt='o', \
                marker='v', markersize=10, capsize=20, color='orange')
    axs.errorbar(z_list[-1], ew0_low_list[-1], yerr=ew0_low_err[:,-1].reshape((2,1)), fmt='o', \
                marker='^', markersize=10, capsize=20, color='orange')
    axs.set_xlabel(r'$z$', fontsize=font_size)
    axs.set_ylabel(r'$W^{\rm emerg}_{\rm Ly\alpha}$ e-folding scale [$\AA$]', fontsize=font_size)
    axs.set_ylim(0, 200)
    axs.set_xlim(2, 7)
    axs.legend(fontsize=font_size)
    # axs.grid()

    plt.show()