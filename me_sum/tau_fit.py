import numpy as np
from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, k_B, m_p, e, m_e

from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

from scipy.optimize import minimize

from time import time

def I(x):
    return x**(9/2)/(1 - x) + (9/7)*x**(7/2) + (9/5)*x**(5/2) + \
        3*x**(3/2) + 9*x**(1/2) - np.log(1 + x**(1/2)) + np.log(1 - x**(1/2))

def delta_nu(T):
    """
    Returns the Doppler width of a Lya line with temperature T
    """
    return ((nu_lya*u.Hz/c) * np.sqrt(2*k_B*T/m_p.to('kg'))).to('Hz').value

def sigmoid(x, h, k, r):
        return h/(1 + np.exp(-k*(x - r)))

def tau_me_effective(zs, nfd, zb, ze):
    """
    Computes a damping wing for an effective single voxel of a supplied
    density of neutral hydrogen, systemic redshift, beginning redshift of the 
    neutral region, and end redshift of the neutral region.
    """
    f_osc = 0.4162
    nu_lya = 2.466e15  # Hz
    c_cgs = c.cgs.value  # speed of light in cm/s
    hbar = 1.0545718e-27  # Planck's constant in erg*s
    e2 = (1/137)*hbar*c_cgs
    me = m_e.cgs.value  # electron mass in g
    tau_gp_prefactor = (np.pi * e2 / (me * c_cgs)) * f_osc / nu_lya
    rho_crit = Planck18.critical_density(zs).to('g/cm^3').value  # critical density in g/cm^3
    n_HI = 0.75 * (rho_crit / m_p.cgs.value)  # number density of HI in 1/cm^3

    # velocity_range = np.linspace(0, 1000, 1000) * u.km/u.s
    # wavelength_range = (velocity_range.to('m/s').value / c.to('m/s').value + 1) * 1215.67
    # velocity_range = c.to('km/s').value * (wavelength_range / 1215.67 - 1)  # in km/s
    # z = (wavelength_range / 1215.67 - 1)*(1 + zs) + zs  # redshift-equivalent wavelengths
    rel_z = np.logspace(-3, 1.0, 1000)
    z = rel_z * (1 + zs) + zs  # redshift at the given relative redshift
    # rel_z = (z - zs)/(1 + zs)

    decay_factor = 6.25e8
    ra = (decay_factor / (4 * np.pi * nu_lya))

    tau_gp = tau_gp_prefactor * n_HI * (c_cgs / Planck18.H(zs).to('1/s').value)
    tau_me_prefactor = tau_gp * ra / np.pi

    # redshift at the given relative redshift
    z = rel_z * (1 + zs) + zs

    tau_me = tau_me_prefactor * nfd * ((1 + zb) / (1 + z))**(3/2) * \
        (I((1 + zb) / (1 + z)) - I((1 + ze) / (1 + z)))
    
    return rel_z, tau_me

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

    # neutral_fraction = np.load('../data/xHI_pencil.npy')
    # vz_comoving        = np.load('../data/vz_pencil.npy')
    # density = np.load('../data/density_pencil.npy')
    # d_los = Planck18.comoving_distance(5.0).to('Mpc') + \
    #     np.linspace(0, 1.5*vz_comoving.shape[0], vz_comoving.shape[0]) * u.Mpc
    # z_los = np.array([z_at_value(Planck18.comoving_distance, d).value for d in d_los])
    # vz_proper = (vz_comoving * u.Mpc/u.s).to('km/s').value/(1 + z_los)

    # dh = c.to('km/s').value/Planck18.H0.to('km/s/Mpc').value  # Hubble distance in Mpc
    # omega_m = Planck18.Om0
    # pade approximant in appendix A of
    # https://www.mdpi.com/2075-4434/4/1/4#app2-galaxies-04-00004

    f_osc = 0.4162
    nu_lya = 2.466e15  # Hz
    # zs = 5.97  # systemic redshift
    # rel_depth = 200
    # zs_idx = np.argmin(np.abs(z_los - zs))
    # neutral_fraction = neutral_fraction[zs_idx-rel_depth:zs_idx]
    # vz_proper = vz_proper[zs_idx-rel_depth:zs_idx]
    # vz_proper = vz_proper[-1] - vz_proper
    # z_los = z_los[zs_idx-rel_depth:zs_idx]
    # density = density[zs_idx-rel_depth:zs_idx]

    # get the beginning and end of each voxel in redshift space
    # dz = (z_los[1] - z_los[0]) / 2
    # ze_los = z_los - dz
    # zb_los = np.zeros_like(z_los)
    # zb_los[:-1] = z_los[:-1] + dz
    # zb_los[-1] = z_los[-1]

    # nfd = 0.01
    zs = 7.0
    ds = Planck18.comoving_distance(zs).to('Mpc')
    ze = 5.5

    colors = ['yellow', 'cyan', 'orange']
    linestyles = ['-', '--', '-.', ':']
    for i, nfd in enumerate([0.1]):
        for j, s in enumerate([1, 10, 50]):
            zb = z_at_value(Planck18.comoving_distance, ds - s*u.Mpc).value
            rel_z, tau_me = tau_me_effective(zs, nfd, zb, ze)
            dv = c.to('km/s').value*rel_z/(1 + zs)
            plt.plot(dv, np.exp(-1*tau_me), color=colors[i], linestyle=linestyles[j], \
                     label=f'zs={zs}, nfd={nfd}, zb={zb:.2f}, ze={ze}')
            
            # def residuals(params):
            #     h, k, r = params
            #     model = sigmoid(dv, h, k, r)
            #     return np.sum((np.exp(-1*tau_me) - model)**2)

            # popt_upper = minimize(residuals, [1.0, 1.0, 0.5]).x
            # model = sigmoid(dv, *popt_upper)
            # plt.plot(dv, model, color=colors[1], linestyle=linestyles[j], \
            #          label=f'Fit: zs={zs}, nfd={nfd}, zb={zb:.2f}, ze={ze}')

    plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    plt.show()