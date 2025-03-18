import numpy as np

from astropy.cosmology import Planck18
from astropy import units as u
from astropy.constants import c, G

def get_beta_bouwens14(muv):
    # https://arxiv.org/pdf/1306.2950
    return -2.05 + -0.2*(muv+19.5)

def get_boundary(x, y, z, level=25):
    X, Y = [], []
    for i, _x in enumerate(x):
        if np.sum(z.T[i] <= level) == 0:
            X.append(_x)
            Y.append(np.max(y))
        else:
            X.append(_x)
            Y.append(np.min(y[z.T[i] <= level]))
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc) 
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')

    redshift = 6.0

    muv_range = np.linspace(-22.5, -16, 1000)
    w_range = np.linspace(0, 800, 300)
    beta = get_beta_bouwens14(muv_range)

    lum_dens_uv = 10**(-0.4*(muv_range - 51.6))
    l_lya = 1215.67# / (1 + redshift)
    nu_lya = (c/(l_lya*u.Angstrom)).to('Hz').value
    lum_dens_alpha = (w_range[:, np.newaxis] / l_lya) * lum_dens_uv * (1215.67/1500)**(beta + 2)
    flux = lum_dens_alpha/(4*np.pi*(Planck18.luminosity_distance(redshift).to('cm').value)**2)
    mab = -2.5 * np.log10(flux) - 48.6

    # lum_dens_uv = 10**(-0.4*(-19.77 - 51.6))
    # l_lya = 1215.67# / (1 + redshift)
    # nu_lya = (c/(l_lya*u.Angstrom)).to('Hz').value
    # lum_dens_alpha = (16 / l_lya) * lum_dens_uv * (1215.67/1500)**(get_beta_bouwens14(-19.77) + 2)
    # flux = lum_dens_alpha/(4*np.pi*(Planck18.luminosity_distance(redshift).to('cm').value)**2)
    # print(flux*nu_lya)
    # print(-2.5*np.log10(flux) - 48.6)
    # quit()

    X25, Y25 = get_boundary(muv_range, w_range, mab, level=-2.5 * np.log10(5e-19/nu_lya) - 48.6)
    X26, Y26 = get_boundary(muv_range, w_range, mab, level=-2.5 * np.log10(1e-18/nu_lya) - 48.6)
    X27, Y27 = get_boundary(muv_range, w_range, mab, level=-2.5 * np.log10(5e-18/nu_lya) - 48.6)

    # plt.pcolormesh(muv_range, w_range, np.log10(flux), vmin=-19, vmax=-16, cmap='inferno')
    plt.pcolormesh(muv_range, w_range, mab, cmap='inferno_r')
    plt.plot(X25, Y25, 'k-', lw=2)
    plt.plot(X26, Y26, 'k-', lw=2)
    plt.plot(X27, Y27, 'k-', lw=2)
    plt.colorbar(label=r'$m_{\rm AB}^{\alpha}$')
    # plt.colorbar(label=r'$\log_{10} f_{\rm Ly\alpha}$ [erg s$^{-1}$ cm$^{-2}$]')
    plt.xlabel(r'$M_{\rm UV}$')
    plt.ylabel('EW')
    plt.show()
