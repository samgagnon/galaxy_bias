import numpy as np

from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, G, m_p, k_B

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

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc) 
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    presentation = False
    if presentation:
        plt.style.use('dark_background')
    import matplotlib as mpl
    label_size = 20
    font_size = 30
    mpl.rcParams['xtick.labelsize'] = label_size 
    mpl.rcParams['ytick.labelsize'] = label_size 

    # supply redshift as an argument
    z_list = [4.9, 5.7, 6.6, 7.0, 7.3]
    if presentation:
        color_list = ['cyan', 'cyan', 'red', 'red', 'red']
        alpha_list = [0.2, 0.3, 0.5, 0.8, 1.0]
    else:
        color_list = ['blue', 'blue', 'red', 'red', 'red']
        alpha_list = [0.4, 0.8, 0.6, 0.8, 1.0]

    fig, axs = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)

    for z, c, a in zip(z_list, color_list, alpha_list):
        lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver = get_silverrush_laelf(z)
        
        g18_phi_upper = np.asarray(logphi_up_silver)
        g18_phi_lower = np.asarray(logphi_low_silver)
        asymmetric_error = np.array(list(zip(g18_phi_lower, g18_phi_upper))).T
        axs.errorbar(lum_silver, logphi_silver, yerr=asymmetric_error, fmt='o', \
                     color=c, alpha=a, label=f'z={z}', markersize=10, capsize=5)

    axs.set_xlabel(r'$\log_{10} L_{\rm Ly\alpha}$ [erg/s]', fontsize=font_size)
    axs.set_ylabel(r'$\log_{10} \phi$ [Mpc$^{-3}$]', fontsize=font_size)
    axs.set_ylim(-7, -3)
    axs.grid()
    axs.legend(fontsize=font_size)
    plt.show()