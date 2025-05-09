import numpy as np

from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, G, m_p, k_B

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


    # second show the e folding scales

    fig, axs = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)

    z_list = [2.2, 3.3, 4.9, 5.7, 6.6]

    r0_list = []
    r0_upper_list = []
    r0_lower_list = []
    gamma_list = []
    gamma_upper_list = []
    gamma_lower_list = []
    # SILVERRUSH XIV
    for z in z_list:
        
        if z == 2.2:
            r0_list.append(3.8)
            r0_upper_list.append(0.5)
            r0_lower_list.append(0.4)
            gamma_list.append(0.87)
            gamma_upper_list.append(0.19)
            gamma_lower_list.append(0.18)
        elif z == 3.3:
            r0_list.append(2.7)
            r0_upper_list.append(0.3)
            r0_lower_list.append(0.4)
            gamma_list.append(1.89)
            gamma_upper_list.append(0.28)
            gamma_lower_list.append(0.2)
        elif z == 4.9:
            r0_list.append(6.4)
            r0_upper_list.append(0.6)
            r0_lower_list.append(0.7)
            gamma_list.append(1.97)
            gamma_upper_list.append(0.16)
            gamma_lower_list.append(0.16)
        elif z == 5.7:
            r0_list.append(4.0)
            r0_upper_list.append(0.6)
            r0_lower_list.append(0.7)
            gamma_list.append(1.02)
            gamma_upper_list.append(0.17)
            gamma_lower_list.append(0.17)
        elif z == 6.6:
            r0_list.append(8.0)
            r0_upper_list.append(1.9)
            r0_lower_list.append(5.8)
            gamma_list.append(1.4)
            gamma_upper_list.append(0.58)
            gamma_lower_list.append(0.88)

    r0_list = np.array(r0_list)
    r0_upper_list = np.array(r0_upper_list)
    r0_lower_list = np.array(r0_lower_list)
    gamma_list = np.array(gamma_list)
    gamma_upper_list = np.array(gamma_upper_list)
    gamma_lower_list = np.array(gamma_lower_list)
    z_list = np.array(z_list)
    r0_err = np.array([r0_upper_list, r0_lower_list])
    gamma_err = np.array([gamma_upper_list, gamma_lower_list])

    axs[0].errorbar(z_list[:-1], r0_list[:-1], yerr=r0_err[:,:-1], fmt='o', \
                marker='o', markersize=10, capsize=20, color='cyan')
    axs[1].errorbar(z_list[:-1], gamma_list[:-1], yerr=gamma_err[:,:-1], fmt='o', \
                marker='o', markersize=10, capsize=20, color='cyan')
    axs[0].errorbar(z_list[-1], r0_list[-1], yerr=r0_err[:,-1].reshape((2,1)), fmt='o', \
                marker='o', markersize=10, capsize=20, color='orange')
    axs[1].errorbar(z_list[-1], gamma_list[-1], yerr=gamma_err[:,-1].reshape((2,1)), fmt='o', \
                marker='o', markersize=10, capsize=20, color='orange')
    
    axs[1].set_xlabel(r'$z$', fontsize=font_size)
    axs[0].set_ylabel(r'$r_0$ [$h_{70}^{-1}$ Mpc$^{-3}$]', fontsize=font_size)
    axs[1].set_ylabel(r'$\gamma$', fontsize=font_size)

    plt.show()