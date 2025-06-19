import numpy as np
from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, k_B, m_p

from scipy.interpolate import interp1d
from scipy.integrate import trapezoid


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

    dv_space = np.linspace(0, 1000, 1000) * u.km/u.s
    for zs in [1, 6, 7, 8, 9]:
        z = zs - dv_space.to('m/s').value / c.to('m/s').value * (1 + zs)
        dc = Planck18.comoving_distance(zs).to('Mpc').value \
            - Planck18.comoving_distance(z).to('Mpc').value

        plt.plot(dv_space.to('km/s').value, dc, color='white', lw=2)
    plt.xlabel(r'$\Delta v$ [km/s]', fontsize=font_size)
    plt.ylabel(r'$d_c$ [Mpc]', fontsize=font_size)
    # plt.xlim(0, 1000)
    # plt.ylim(0, 500)
    plt.show()