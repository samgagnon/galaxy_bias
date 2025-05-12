import numpy as np

from scipy import odr

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
    import matplotlib as mpl
    label_size = 20
    font_size = 30
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size
    presentation = False
    if presentation == True:
        plt.style.use('dark_background')
        linecolor = 'cyan'
        datacolor = 'cyan'
    else:
        linecolor = 'red'
        datacolor = 'black'

    # measured lya properties from https://arxiv.org/pdf/2402.06070
    MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, \
        fescA, fescA_err, fescB, fescB_err, ID = get_tang_data()
    
    # dataset from https://arxiv.org/pdf/2202.06642
    idx, z, ze, w, we, muv, muve, llya, llyae, peaksep, \
        peakse, fwhm, fwhme, assym, assyme = np.load('../data/muse.npy').T
      
    fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    # axs.set_aspect(0.5)
    axs.errorbar(muv, w, xerr=muve, yerr=we, fmt='.', color='blue', markersize=15, alpha=0.5)
    axs.errorbar(MUV[ID==0], ew_lya[ID==0], xerr=MUV_err[ID==0], \
                 yerr=ew_lya_err[ID==0], fmt='o', color=datacolor, markersize=15)
    axs.errorbar(MUV[ID==1], ew_lya[ID==1], xerr=MUV_err[ID==1], \
                 yerr=ew_lya_err[ID==1], fmt='^', color=datacolor, markersize=15)
    axs.set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
    axs.set_ylabel(r'${\rm W}_{\rm Ly\alpha}$ [Å]', fontsize=font_size)
    axs.set_xlim(-22, -16)
    axs.set_ylim(1, 1000)
    axs.set_yscale('log')
    axs.grid()
    plt.show()