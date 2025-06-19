"""
Functions for calculating the rest-frame absorption spectrum
in a voxel of a light cone. 
"""

import numpy as np
from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, k_B, m_p

from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

# lyman alpha parameters
wave_Lya = 1215.67*u.AA
freq_Lya = (c.to(u.AA/u.s)/wave_Lya).to('Hz')
omega_Lya = 2*np.pi*freq_Lya
decay_factor = 6.25e8*u.s**-1

def get_los_properties():
    # neutral_fraction = np.load('../data/xHI_pencil.npy')[500:700]
    # density = np.load('../data/density_pencil.npy')[500:700]
    # vz = np.load('../data/vz_pencil.npy')[500:700]
    dc = Planck18.comoving_distance(5.00) + np.linspace(0, 1.5*2562, 2562)*u.Mpc
    z_los = np.array([z_at_value(Planck18.comoving_distance, d).value for d in dc[500:700]])
    neutral_fraction = np.ones(200)
    density = np.zeros(200)
    # vz = -1*np.random.exponential(scale=100, size=200)
    vz = np.zeros(200)
    return neutral_fraction, density, vz, z_los

def delta_nu(T):
    """
    Returns the Doppler width of a Lya line with temperature T
    """
    return ((freq_Lya/c) * np.sqrt(2*k_B*T/m_p.to('kg'))).to('Hz')

def get_a(dnu):
    """
    Returns the damping parameter of a Lya line with Doppler width dnu
    """
    return (decay_factor/(4*np.pi*dnu)).to('')

def voigt_tasitsiomi(x, dnu):
    """
    Returns the Voigt profile of a line with damping parameter a
    """
    dL = 9.936e7*u.Hz
    a = 0.5*(dL/dnu)
    xt = x**2
    z = (xt - 0.855)/(xt + 3.42)
    q = np.zeros(len(x))
    IDX = (z > 0.0)
    q[IDX] = z[IDX]*(1 + 21/xt[IDX])*(a/np.pi/(xt[IDX] + 1.0))*\
        (0.1117 + z[IDX]*(4.421 + z[IDX]*(-9.207 + 5.674*z[IDX])))
    return (q + np.exp(-xt)/1.77245385)*np.sqrt(np.pi)

def lya_crosssec(wl, v_p, T):
    """
    Returns the cross section of Lyman alpha photons with neutral hydrogen
    Args:
    wl: wavelength of the photon
    v_p: peculiar velocity of the absorbing gas
    T: temperature of the gas
    """
    dnu = delta_nu(T)
    a = get_a(dnu)
    velocity_range = c.to('km/s')*(wl/wave_Lya - 1)
    x_range_stationary = (velocity_range/c).to('')*(freq_Lya/dnu).to('')
    x_peculiar = (v_p/c.to('km/s')*(freq_Lya/dnu)).to('')
    # apply peculiar velocity correction
    x_range = x_range_stationary - x_peculiar
    # get the Voigt profile, approximated by Tasitsiomi's formula
    approx_voigt = voigt_tasitsiomi(x_range, dnu)
    # apply Rayleigh correction
    v_thermal = np.sqrt(2*k_B*T/m_p).to('km/s')
    approx_voigt *= (1 + x_peculiar*v_thermal/c.to('km/s'))**4
    # apply quantum mechanical correction
    approx_voigt *= (1 - 1.792*x_peculiar*v_thermal/c.to('km/s'))
    # multiply by dimensionful prefactor
    sigma_0 = 5.88e-14*(T/(1e4*u.K))**(-0.5)*u.cm**2
    prefactor = sigma_0*a/np.pi
    approx_voigt *= prefactor
    return approx_voigt

def I(x):
    return x**(9/2)/(1 - x) + (9/7)*x**(7/2) + (9/5)*x**(5/2) + \
        3*x**(3/2) + 9*x**(1/2) - np.log(1 + x**(1/2)) + np.log(1 - x**(1/2))

# def tau_igm(zs, bubble_size=5, mode='fast', neutral_fraction=None, \
#             density=None, z_los=None, vz=None):
#     """
#     Computes the absorption spectrum in a voxel.
#     """
#     wave_Lya = 1215.67*u.AA
#     freq_Lya = (c.to(u.AA/u.s)/wave_Lya).to('Hz')
#     omega_Lya = 2*np.pi*freq_Lya
#     decay_factor = 6.25e8*u.s**-1
#     wavelength_range = np.linspace(1200, 1250, 1000)*u.AA
#     # define cross section constants
#     # get width of the Lyman alpha line for the assumed 
#     # temperature of all neutral gas (~10^4 K)
#     dnu = delta_nu(1e4*u.K)
#     a = get_a(dnu)
#     # v_thermal = np.sqrt(2*k_B*1e4*u.K/m_p).to('km/s').value
#     sigma_0 = 5.88e-14*u.cm**2
#     prefactor = (sigma_0*a/np.pi).value
#     av = 4.7e-4
#     c_kps = c.to('km/s').value
#     voigt_constant = av/(np.sqrt(np.pi))
#     # compute the redshift jacobian for the lowres sightline
#     jacobian = (c/Planck18.H(z_los)).to('cm').value

#     # get hires redshift line of sight
#     z_los_hires = np.linspace(z_los[0], z_los[-1], 1000)
#     vz_hires = np.interp(z_los_hires, z_los, vz)

#     # get density of neutral hydrogen along the LoS
#     rho_crit = Planck18.critical_density(z_los_hires)
#     n_hi = 0.75*(rho_crit/m_p).to('1/cm^3') # precompute this!

#     # precompute voigt tables in rest frame of each systemic redshift in the LoS
#     jacobian_rel = np.interp(z_los_hires, z_los, jacobian)
#     dz = np.diff(z_los_hires)

#     # produce x range for template voigt profile
#     velocity_range = c.to('km/s')*(wavelength_range/wave_Lya - 1)
#     x0 = (velocity_range/c).to('')*(freq_Lya/dnu).to('')

#     # voigt profile template for interpolation
#     voigt_template = voigt_tasitsiomi(x0, dnu.value)
#     voigt_interpolator = interp1d(x0, voigt_template, bounds_error=False,\
#                         fill_value=(voigt_template[0], voigt_template[-1]))

#     voigt_table = np.zeros((len(z_los_hires), len(wavelength_range)))
#     for l in range(len(z_los_hires)):
#         # interpolate to frame of reference
#         # TODO add velocity perturbation from peculiar motion?
#         velocity_range = c_kps*(wavelength_range*(1+zs)/(1+z_los_hires[l])/wave_Lya - 1)
#         velocity_range += vz_hires[l]
#         x_range_stationary = (velocity_range/c_kps)*(freq_Lya/dnu)
#         voigt_table[l] = voigt_interpolator(x_range_stationary)

#     neutral_fraction = np.interp(z_los_hires, z_los, neutral_fraction)
#     zb = z_at_value(Planck18.comoving_distance, \
#             Planck18.comoving_distance(zs) - bubble_size*u.Mpc).value
#     neutral_fraction[z_los_hires > zb] = 0.0
#     # neutral_fraction *= 0.5

#     density = np.interp(z_los_hires, z_los, density)
#     n_hi *= (1 + density)*neutral_fraction

#     # perform integral for each velocity bin
#     crosssec_table = voigt_table*prefactor
#     n_hi_table = np.array([n_hi]*voigt_table.shape[1]).T
#     dz_table = np.array([dz]*voigt_table.shape[1]).T
#     z_table = np.array([jacobian_rel]*voigt_table.shape[1]).T

#     integrand = crosssec_table * n_hi_table * z_table
#     # tau = np.sum((integrand[:-1] + integrand[1:]) * dz_table / 2, axis=0)
#     tau = trapezoid(integrand, z_los_hires, axis=0)

#     velocity_range = c_kps*(wavelength_range/wave_Lya - 1)
#     return velocity_range.value, tau

def tau_igm(zs, bubble_size=5):
    wave_Lya = 1215.67*u.AA
    freq_Lya = (c.to(u.AA/u.s)/wave_Lya).to('Hz')
    omega_Lya = 2*np.pi*freq_Lya
    c_kps = c.to('km/s').value
    decay_factor = 6.25e8*u.s**-1
    wavelength_range = np.linspace(1200, 1220, 1000)*u.AA
    ds = Planck18.comoving_distance(zs).to('Mpc').value
    d_space = ds - \
        np.linspace(0, 50, 1000)[::-1]
    z_los = np.array([z_at_value(Planck18.comoving_distance, d*u.Mpc).value for d in d_space])
    zb = z_los[np.argmin(np.abs(d_space - (ds - bubble_size)))]
    ze = 5.5
    tau_gp = 7.16e5*((1+zs)/10)**(3/2)
    ra = (decay_factor/(4*np.pi*delta_nu(1e4*u.K))).to('').value
    prefactor = ra/np.pi
    z = wavelength_range/wave_Lya - 1 + zs

    tau = prefactor * ((1+zb)/(1+z))**(3/2) * (I((1+zb)/(1+z)) - I((1+ze)/(1+z)))

    velocity_range = c_kps*(wavelength_range/wave_Lya - 1)
    return velocity_range, tau

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

    styles = ['-', '--', '-.', ':']
    bubble_sizes = [0.0, 0.1, 1.0, 5.0]
    for i, b in enumerate(bubble_sizes):
        # wavelength_range, z, tau_IGM, crosssec_table, integrand = tau_igm(zs, bubble_size=b)
        neutral_fraction, density, vz, z_los = get_los_properties()
        # neutral_fraction[len(neutral_fraction)-int(b/1.5):] = 0.0
        zs = z_los[-1]
        velocity_range, tau_IGM = tau_igm(zs, bubble_size=b)
        color = 'white'
        # plt.plot(velocity_range, tau_IGM, color='orange', \
        #         linestyle=styles[i])
        
        plt.plot(velocity_range, np.exp(-1*tau_IGM), color=color, \
                linestyle=styles[i])
        # plt.plot(velocity_range, np.exp(-1*tau_IGM_me), color='orange', \
        #         linestyle=styles[i])
    plt.xlabel(r'$\Delta v$ [$km/s$]', fontsize=16)
    plt.ylabel(r'exp$(-\tau)$', fontsize=16)
    # plt.yscale('log')
    teXstr1 = r'$\tau_{\rm IGM}$'
    teXstr2 = rf'$z={np.around(zs, 2)}$'
    plt.title(f'{teXstr1} to an LAE at {teXstr2} in an ionized bubble of radius s Mpc')
    plt.show()