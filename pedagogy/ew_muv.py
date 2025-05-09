import numpy as np

from scipy import integrate
from scipy import special
from scipy.optimize import minimize
from astropy.cosmology import Planck18

def wlim(muv, flim=1e-19):
    return 0.56*muv + 10.2 + np.log10(1.36e19*flim)

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
    textcolor = 'white'
    cmap = 'Greys_r'
    datacolor = 'red'
    datacolor2 = 'yellow'
    hist_cmap = 'hot'
    import matplotlib as mpl
    label_size = 20
    font_size = 30
    mpl.rcParams['xtick.labelsize'] = label_size 
    mpl.rcParams['ytick.labelsize'] = label_size

    hist01 = np.load('../data/muse_hist/hist01.npy')
    hist11 = np.load('../data/muse_hist/hist11.npy')
    model01 = np.load('../data/model_hist/model01.npy')
    model11 = np.load('../data/model_hist/model11.npy')
    nobias = np.load('../data/model_hist/muv_w.npy')

    MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, \
        fescA, fescA_err, fescB, fescB_err, ID = get_tang_data()

    muv_space = np.linspace(-24, -16, 100)
    w_space = 10**np.linspace(0, 3, 100)

    wlim_deep = wlim(muv_space, flim=2e-18)
    wlim_wide = wlim(muv_space, flim=3e-17)

    fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

    # axs.errorbar(MUV, dv_lya, yerr=dv_lya_err, xerr=MUV_err, fmt='o', color=datacolor,
    #     markersize=10, capsize=5, label='TANG LAE data')
    # axs.errorbar(MUV[ID==0], ew_lya[ID==0], yerr=ew_lya_err[ID==0], xerr=MUV_err[ID==0],
    #     fmt='o', markersize=10, capsize=5, color=datacolor2)
    # axs.errorbar(MUV[ID==1], ew_lya[ID==1], yerr=ew_lya_err[ID==1], xerr=MUV_err[ID==1],
        # fmt='o', markersize=10, capsize=5, color=datacolor2)

    # axs.pcolormesh(muv_space, w_space, nobias, cmap=cmap)
    # axs.pcolormesh(muv_space, w_space, np.log10(model01), cmap=cmap)
    axs.pcolormesh(muv_space, w_space, np.log10(model11), cmap=cmap)
    # axs.pcolormesh(muv_space, w_space, hist01, cmap=hist_cmap, alpha=0.5)
    axs.pcolormesh(muv_space, w_space, hist11, cmap=hist_cmap, alpha=0.5)
    # axs.plot(muv_space, 10**(0.36*muv_space + 6.7 + np.log10(2e-18/1e-19)), linestyle='-', color='white')
    # axs.plot(muv_space, 10**(0.36*muv_space + 6.7 + np.log10(2e-17/1e-19)), linestyle='-', color='white')
    # axs.axhline(8, color='white', linestyle='-')
    # axs.axhline(80, color='white', linestyle='-')

    axs.set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
    axs.set_ylabel(r'${\rm W}_{\rm emerg}$ [$\AA$]', fontsize=font_size)
    axs.set_yscale('log')
    axs.set_ylim(1, 1000)
    axs.set_xlim(-22, -16)
    plt.show()