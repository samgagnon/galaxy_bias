import numpy as np

from astropy.cosmology import Planck18
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

    u1 = np.random.uniform(0, 1, 1000)
    u2 = np.random.uniform(0, 1, 1000)
    r = np.sqrt(-2*np.log(u1))
    theta = 2*np.pi*u2

    x = r*np.cos(theta)
    y = r*np.sin(theta)

    params = np.array([x, y])
    covar = np.array([[3.0**2, -1],[-1, 1.0**2]])
    params = np.linalg.cholesky(covar)@params
    x, y = params

    plt.scatter(x, y, s=1)
    plt.show()