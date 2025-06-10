"""
Functions for calculating the rest-frame absorption spectrum
in a voxel of a light cone. 
"""

import numpy as np
from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, k_B, m_p

from scipy.interpolate import interp1d

# lyman alpha parameters
wave_Lya = 1215.67*u.AA
freq_Lya = (c.to(u.AA/u.s)/wave_Lya).to('Hz')
omega_Lya = 2*np.pi*freq_Lya
decay_factor = 6.25*1e8*u.s**-1
wavelength_range = np.linspace(1215, 1220, 1000)*u.AA

def get_los_properties():
    neutral_fraction = np.loadtxt('./data/xHI_pencil.txt')
    density = np.loadtxt('./data/density_pencil.txt')
    vz = np.loadtxt('./data/vz_pencil.txt')
    dc = Planck18.comoving_distance(5.00) + np.linspace(0, 1.5, len(neutral_fraction))*u.Mpc
    z_los = np.array([z_at_value(Planck18.comoving_distance, d).value for d in dc])
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

def get_tau_igm(neutral_fraction, density, n_hi, z_los):
    """
    Computes the absorption spectrum in a voxel.
    """
    # define cross section constants
    # get width of the Lyman alpha line for the assumed 
    # temperature of all neutral gas (~10^4 K)
    dnu = delta_nu(1e4*u.K)
    a = get_a(dnu)
    # v_thermal = np.sqrt(2*k_B*1e4*u.K/m_p).to('km/s').value
    sigma_0 = 5.88e-14*u.cm**2
    prefactor = (sigma_0*a/np.pi).value
    av = 4.7e-4
    c_kps = c.to('km/s').value
    # voigt_constant = av/(np.sqrt(np.pi))
    zs = z_los[-1]
    # compute the redshift jacobian for the lowres sightline
    jacobian = (c/Planck18.H(z_los)).to('cm').value

    # get hires redshift line of sight
    z_los_hires = np.linspace(z_los[0], z_los[-1], 1000)

    # precompute voigt tables in rest frame of each systemic redshift in the LoS
    jacobian_rel = np.interp(z_los_hires, z_los, jacobian)
    dz = np.diff(z_los_hires)

    # produce x range for template voigt profile
    wavelength_range_extd = np.linspace(1210, 1250, 1000)*u.AA
    velocity_range = c.to('km/s')*(wavelength_range_extd/wave_Lya - 1)
    x0 = (velocity_range/c).to('')*(freq_Lya/dnu).to('')

    # voigt profile template for interpolation
    voigt_template = voigt_tasitsiomi(x0, dnu.value)
    voigt_interpolator = interp1d(x0, voigt_template, bounds_error=False,\
                        fill_value=(voigt_template[0], voigt_template[-1]))

    voigt_table = np.zeros((len(z_los_hires), len(wavelength_range)))
    for l in range(len(z_los_hires)):
        # interpolate to frame of reference
        # TODO add velocity perturbation from peculiar motion?
        velocity_range = c_kps*(wavelength_range*(1+zs)/(1+z_los_hires[l])/wave_Lya - 1)
        x_range_stationary = (velocity_range/c_kps)*(freq_Lya/dnu)
        voigt_table[l] = voigt_interpolator(x_range_stationary)

    neutral_fraction = np.interp(z_los_hires, z_los, neutral_fraction)
    density = np.interp(z_los_hires, z_los, density)
    n_hi = np.interp(z_los_hires, z_los, n_hi)
    n_hi *= (1 + density)*neutral_fraction

    # perform integral for each velocity bin
    crosssec_table = voigt_table*prefactor
    n_hi_table = np.array([n_hi]*voigt_table.shape[1]).T
    dz_table = np.array([dz]*voigt_table.shape[1]).T
    z_table = np.array([jacobian_rel]*voigt_table.shape[1]).T

    integrand = crosssec_table * n_hi_table * z_table
    tau = np.sum((integrand[:-1] + integrand[1:]) * dz_table / 2, axis=0)
    return tau

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

    neutral_fraction, density, vz, z_los = get_los_properties()
    n_hi = 0.75 * (Planck18.critical_density(z_los) / m_p).to('1/cm^3').value
    tau_igm = get_tau_igm(neutral_fraction, density, n_hi, z_los)

    fig, axs = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    axs.plot(z_los, tau_igm, color='white', lw=2)
    axs.set_xlabel(r'$z$', fontsize=font_size)
    axs.set_ylabel(r'$\tau_{\mathrm{IGM}}$', fontsize=font_size)
    plt.show()