import os
import numpy as np

from astropy.cosmology import Planck18, z_at_value
from astropy import units as U, constants as c

def get_A(m):
    return 0.65 + 0.1 * np.tanh(3 * (m + 20.75))

def get_Wc(m):
    return 31 + 12 * np.tanh(4 * (m + 20.25))

def probability_EW(W, Wc):
    return (1/Wc) * np.exp(-W/Wc)

def mason2018(Muv):
    """
    Samples EW and emission probability from the
    fit functions obtained by Mason et al. 2018.
    """
    A = get_A(Muv)
    rv_A = np.random.uniform(0, 1, len(Muv))
    emit_bool = rv_A < A
    Wc = get_Wc(Muv[emit_bool])
    rv_W = np.random.uniform(0, 1, len(Wc))
    W = -1*Wc*np.log(rv_W)
    return W, emit_bool

def gaussian(x, mu, sigma):
    return 1/(sigma*np.sqrt(2*np.pi))\
        *np.exp(-0.5*((x - mu)/sigma)**2)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')
    # EUCLID has a grism!!!!

    z = 6.6
    Muv = np.linspace(-16, -23, 10)
    Mh = 10**(-0.3*(Muv + 20.0 + 0.26*z) + 11.75)

    A = get_A(Muv)
    rv_A = np.random.uniform(0, 1, len(Muv))
    emit_bool = rv_A < A

    # this takes the halo mass!
    m = 0.32
    C = 2.48
    # this is quite tight, seems unphysical to be honest
    dv_mean = m*(np.log10(Mh) - np.log10(1.55) - 12) + C
    dv_sigma = 0.24
    # we've got to fix this
    dv = []
    for dvm in dv_mean:
        dv.append(10**(np.random.normal(dvm, dv_sigma)))
    dv = np.array(dv)
    # Verhamme et al 2015
    v_dispersion = 20 + dv/2.355
    
    # this is to demonstrate what the profiles look like
    vrange = np.linspace(-50, 200, 1000)

    # for dvm, vd in zip(dv, v_dispersion):
    #     single_gaussian = gaussian(vrange, dvm, vd)
    #     plt.plot(vrange, single_gaussian, linestyle='-', color='lime', alpha=0.5)
    
    # plt.xlabel(r'$v$ [km/s]')
    # plt.ylabel(r'$P(v)$')
    # plt.grid()
    # plt.xlim(-50, 200)
    # plt.show()
    # quit()

    EW = []
    for dvm, vd, mh in zip(dv, v_dispersion, Mh):
        single_gaussian = gaussian(vrange, dvm, vd)
        v_circ = ((100*c.G*mh*U.solMass*Planck18.H(z)).to('km^3/s^3').value)**(1/3)
        EW.append(np.sum(single_gaussian[vrange > v_circ]))
    
    plt.hist(EW, bins=20, histtype='step', color='cyan')
    plt.xlabel(r'$W$ [$\AA$]')
    plt.ylabel(r'$N$')
    plt.grid()
    plt.show()