import numpy as np
from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, k_B, m_p, e, m_e

from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

from time import time

def I(x):
    return x**(9/2)/(1 - x) + (9/7)*x**(7/2) + (9/5)*x**(5/2) + \
        3*x**(3/2) + 9*x**(1/2) - np.log(1 + x**(1/2)) + np.log(1 - x**(1/2))

def delta_nu(T):
    """
    Returns the Doppler width of a Lya line with temperature T
    """
    return ((nu_lya*u.Hz/c) * np.sqrt(2*k_B*T/m_p.to('kg'))).to('Hz').value

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

    neutral_fraction = np.load('../data/xHI_pencil.npy')
    vz_comoving        = np.load('../data/vz_pencil.npy')
    density = np.load('../data/density_pencil.npy')
    d_los = Planck18.comoving_distance(5.0).to('Mpc') + \
        np.linspace(0, 1.5*vz_comoving.shape[0], vz_comoving.shape[0]) * u.Mpc
    z_los = np.array([z_at_value(Planck18.comoving_distance, d).value for d in d_los])
    vz_proper = (vz_comoving * u.Mpc/u.s).to('km/s').value/(1 + z_los)

    zs = 5.97  # systemic redshift
    rel_depth = 200
    zs_idx = np.argmin(np.abs(z_los - zs))
    neutral_fraction = neutral_fraction[zs_idx-rel_depth:zs_idx]
    vz_proper = vz_proper[zs_idx-rel_depth:zs_idx]
    vz_proper = vz_proper[-1] - vz_proper
    z_los = z_los[zs_idx-rel_depth:zs_idx]
    density = density[zs_idx-rel_depth:zs_idx]

    # get the beginning and end of each voxel in redshift space
    dz = (z_los[1] - z_los[0]) / 2
    ze_los = z_los - dz
    zb_los = np.zeros_like(z_los)
    zb_los[:-1] = z_los[:-1] + dz
    zb_los[-1] = z_los[-1]

    f_osc = 0.4162
    nu_lya = 2.466e15  # Hz
    c_cgs = c.cgs.value  # speed of light in cm/s
    c_kps = c.to('km/s').value  # speed of light in km/s
    hbar = 1.0545718e-27  # Planck's constant in erg*s
    e2 = (1/137)*hbar*c_cgs
    me = m_e.cgs.value  # electron mass in g
    tau_gp_prefactor = (np.pi * e2 / (me * c_cgs)) * f_osc / nu_lya
    print(f"Prefactor for GP optical depth: {tau_gp_prefactor:.2e} cm^2")
    rho_crit = Planck18.critical_density(zs).to('g/cm^3').value  # critical density in g/cm^3
    n_HI = 0.75 * (rho_crit / m_p.cgs.value)  # number density of HI in 1/cm^3
    # multiply this by the neutral fraction, the filling factor, and the overdensity
    dH = c_cgs / Planck18.H(zs).to('1/s').value  # Hubble distance in cm
    print(f"Number density of HI: {n_HI:.2e} 1/cm^3")
    print(f"Hubble distance at z={zs}: {dH:.2e} cm")

    decay_factor = 6.25e8
    ra = (decay_factor / (4 * np.pi * nu_lya))

    tau_gp = tau_gp_prefactor * n_HI * (c_cgs / Planck18.H(zs).to('1/s').value)
    tau_me_prefactor = tau_gp * ra / np.pi

    # velocity_range = np.linspace(0, 3000, 1000) * u.km/u.s
    # wavelength_range = (velocity_range.to('m/s').value / c.to('m/s').value + 1) * 1215.67
    # velocity_range = c.to('km/s').value * (wavelength_range / 1215.67 - 1)  # in km/s
    # z = (wavelength_range / 1215.67 - 1)*(1 + zs) + zs  # redshift-equivalent wavelengths
    # rel_z = (z - zs)/(1 + zs)
    rel_z = np.linspace(0, 0.05, 1000)

    linestyles = [':', '-.', '--', '-']
    # for i, s in enumerate([1, 10, 50, 100]):
    # zb = z_at_value(Planck18.comoving_distance, 
    #                     Planck18.comoving_distance(zs) - s * u.Mpc).value
    #     ze = 5.5  # end of reionization

    # loop through the contribution of each voxel in the lightcone
    tau_me = np.zeros_like(rel_z)
    tau_me_novz = np.zeros_like(rel_z)

    start_integral = time()    

    for zb, ze, delta, nf, vz in zip(zb_los, ze_los, density, neutral_fraction, vz_proper):

        if nf == 0:
            continue

        # redshift at the given relative redshift
        z = (rel_z + vz/c_kps) * (1 + zs) + zs

        tau_me_voxel = tau_me_prefactor * ((1 + zb) / (1 + z))**(3/2) * \
            (I((1 + zb) / (1 + z)) - I((1 + ze) / (1 + z)))
        tau_me_voxel *= (1 + delta) * nf
        
        tau_me += tau_me_voxel

        z = rel_z * (1 + zs) + zs
        tau_me_voxel = tau_me_prefactor * ((1 + zb) / (1 + z))**(3/2) * \
            (I((1 + zb) / (1 + z)) - I((1 + ze) / (1 + z)))
        tau_me_voxel *= (1 + delta) * nf
        tau_me_novz += tau_me_voxel

    print(f"Time to integrate optical depth: {time() - start_integral:.2f} s")

    z = rel_z * (1 + zs) + zs  # redshift at the given relative redshift
    wavelength_range = 1215.67*((z - zs)/(1 + zs) + 1)
    velocity_range = c_kps * (wavelength_range / 1215.67 - 1)  # in km/s
        
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), constrained_layout=True)
    axs[0].plot(z_los, neutral_fraction, color='white', linestyle=linestyles[-1])
    axs[0].set_ylabel(r'$x_{\rm HI}$', fontsize=font_size)
    axs[1].plot(z_los, density, color='white', linestyle=linestyles[-1])
    axs[1].set_ylabel(r'$\delta$', fontsize=font_size)
    axs[2].plot(z_los, vz_proper, color='white', linestyle=linestyles[-1])
    axs[2].set_ylabel(r'$\Delta v_z$ [km/s]', fontsize=font_size)
    axs[2].set_xlabel(r'$z$', fontsize=font_size)
    axs[3].plot(velocity_range, np.exp(-1*tau_me), color='white', linestyle=linestyles[-1])
    axs[3].plot(velocity_range, np.exp(-1*tau_me_novz), color='cyan', linestyle=linestyles[-1])
    axs[3].set_ylabel(r'$e^{-\tau_{\rm IGM}}$', fontsize=font_size)
    axs[3].set_xlabel(r'$\Delta v$ [km/s]', fontsize=font_size)

    plt.show()
    
