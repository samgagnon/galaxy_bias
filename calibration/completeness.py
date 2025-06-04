"""
We must make a case that the apparent correlations in our data are not due to selection effects.
"""

import numpy as np

from astropy.cosmology import Planck18
from astropy import units as u
from astropy.constants import c, G, m_p, k_B

def get_tang_data():
    # TANG LAE data
    # https://arxiv.org/pdf/2402.06070
    MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, fescA, \
        fescA_err, fescB, fescB_err, ID = np.load('../data/tang24.npy').T
    return MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, \
        fescA, fescA_err, fescB, fescB_err, ID

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

    muv, muverr, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, \
        fescA, fescA_err, fescB, fescB_err, ID = get_tang_data()
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    ax[0].errorbar(muv[ID==0], ew_lya[ID==0], xerr=muverr[ID==0], yerr=ew_lya_err[ID==0], fmt='o', 
                   color='C0', label='MUSE-Wide')
    ax[0].errorbar(muv[ID==1], ew_lya[ID==1], xerr=muverr[ID==1], yerr=ew_lya_err[ID==1], fmt='o', 
                   color='C1', label='MUSE-Deep')
    # ax[0].errorbar(muv[ID==2], ew_lya[ID==2], xerr=muverr[ID==2], yerr=ew_lya_err[ID==2], fmt='o', 
    #                color='C2', label='DEIMOS')
    ax[0].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=label_size)
    ax[0].set_ylabel(r'$W_{\rm emerg}$ [$\AA$]', fontsize=label_size)
    ax[1].errorbar(muv[ID==0], dv_lya[ID==0], xerr=muverr[ID==0], yerr=dv_lya_err[ID==0], fmt='o', 
                   color='C0', label='dv(Lyα)')
    ax[1].errorbar(muv[ID==1], dv_lya[ID==1], xerr=muverr[ID==1], yerr=dv_lya_err[ID==1], fmt='o', 
                   color='C1', label='dv(Lyα)')
    # ax[1].errorbar(muv[ID==2], dv_lya[ID==2], xerr=muverr[ID==2], yerr=dv_lya_err[ID==2], fmt='o', 
    #                color='C2', label='dv(Lyα)')
    ax[1].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=label_size)
    ax[1].set_ylabel(r'$\Delta v$ [km/s]', fontsize=label_size)
    plt.show()
    