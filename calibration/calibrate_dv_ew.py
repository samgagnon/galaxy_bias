"""
A script to calibrate the intrinsic equivalent width and velocity offset 
of Lyman alpha emitters such that the observed distributions match those 
in Mason (2018).

Samuel Gagnon-Hartman
Scuola Normale Superiore, Pisa, Italy
February 2025
"""

import numpy as np

from scipy.special import erf

def plot(mode: str, *args):
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc) 
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')
    
    if mode == 'transmission_PDF':

        sigma_dv = 100
        delta_v_space = np.linspace(-200, 200, 1000)
        sigma_v_space = delta_v_space*2*np.sqrt(2*np.log(2))
        # vc = 200

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        vc_list = [50]

        for vc in vc_list:
            # transmission PDF
            tau_CGM = 1 - 0.5*(1 + erf((vc - delta_v_space)/np.sqrt(2*sigma_v_space**2)))
            delta_v_PDF = 1/np.sqrt(2*np.pi*sigma_dv**2)*np.exp(-1*delta_v_space**2/(2*sigma_dv**2))
            tau_CGM_grad = np.gradient(tau_CGM)
            tau_CGM_PDF = delta_v_PDF/np.abs(tau_CGM_grad)

            ax.plot(tau_CGM, tau_CGM_PDF, label=r'$v_c=$' + str(vc) + ' km/s')
        
        ax.set_xlabel(r'$\mathcal{T}_{\rm CGM}$')
        ax.set_ylabel(r'$P(\mathcal{T}_{\rm CGM})$')
        # ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        plt.show()

    pass

if __name__ == "__main__":

    # parameters of distribution

    # DV_intr is drawn from a Gaussian distribution with mean 0 and 
    # a standard deviation whose value increases with Mh.

    # EW_intr is drawn from a Gaussian distribution with mean 0 and
    # a standard deviation whose value increases with Mh.

    plot('transmission_PDF')