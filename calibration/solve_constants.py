import numpy as np
from astropy import units as u
from astropy.constants import c

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

    npoints = 10000
    av = -74.73
    bv = np.random.normal(-1206.06, 90.99, npoints)
    aw = -1/641.83
    bw = np.random.normal(2.35, 0.33, npoints)
    bh = np.random.normal(41.96, 0.36, npoints)

    muv = -18
    for muv in np.arange(-22, -16, 1.0):
        logfesc = aw*av*muv + aw*bv + bw - bh + 39.13

        plt.hist(logfesc, bins=100, density=True, alpha=0.7, \
                label=r'${\rm M}_{\rm UV}$='+f'{muv:.2f}', edgecolor='black')
    plt.xlabel(r'$\log_{10} f_{\rm esc}$')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()