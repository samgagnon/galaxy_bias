import numpy as np

from scipy import integrate
from scipy import special
from scipy.optimize import minimize
from astropy.cosmology import Planck18

def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def mh_muv_mason2015(muv, redshift):
    """
    Returns the halo mass from the UV magnitude
    using the fit function from Mason et al. 2015.
    """
    gamma = np.zeros_like(muv)
    gamma[muv >= -20 - redshift*0.26] = -0.3
    gamma[muv < -20 - redshift*0.26] = -0.7
    return 10**(gamma*(muv + 20 + 0.26*redshift) + 11.75)

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

    vrange = np.linspace(-150, 500, 1000)
    line = gaussian(vrange, 100, 100)
    
    fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    
    axs.plot(vrange, line, color='cyan', linestyle='-')
    axs.fill_between(vrange[vrange>150], line[vrange>150], color='cyan', alpha=0.5)
    axs.set_ylim(0, 1.1)
    axs.set_xlim(-150, 500)
    axs.set_xlabel(r'$\Delta v$ [km/s]', fontsize=font_size)
    axs.set_ylabel(r'flux [arb. units]', fontsize=font_size)
    plt.show()