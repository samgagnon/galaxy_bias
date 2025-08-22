import h5py
import os

import numpy as np
from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, k_B, m_p

from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

import matplotlib.pyplot as plt
rc = {"font.family" : "serif", 
    "mathtext.fontset" : "stix"}
plt.rcParams.update(rc) 
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({'font.size': 14})
import matplotlib as mpl
label_size = 20
font_size = 30
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size

presentation = False  # Set to True for presentation style
# presentation = True
if presentation == True:
    plt.style.use('dark_background')
    color1 = 'cyan'
    color2 = 'lime'
    color3 = 'orange'
    textcolor = 'white'
else:
    color1 = 'black'
    color2 = 'blue'
    color3 = 'red'
    color4 = 'orange'
    textcolor = 'black'

def get_muv(sfr):
    """
    Converts star formation rate to
    absolute UV magnitude.
    """
    kappa = 1.15e-28
    luv = sfr * 3.1557e7 / kappa
    muv = 51.64 - np.log10(luv) / 0.4
    return muv
    
def mh(muv):
    """
    Returns log10 Mh in solar masses as a function of MUV.
    """
    redshift = 5.0
    muv_inflection = -20.0 - 0.26*redshift
    gamma = 0.4*(muv >= muv_inflection) - 0.7
    return gamma * (muv - muv_inflection) + 11.75

def vcirc(muv):
    """
    Returns circular velocity in km/s as a function of MUV 
    at redshift 5.0.
    """
    log10_mh = mh(muv)
    return (log10_mh - 5.62)/3

def IME(x):
    return x**(9/2)/(1 - x) + (9/7)*x**(7/2) + (9/5)*x**(5/2) + \
        3*x**(3/2) + 9*x**(1/2) - np.log(1 + x**(1/2)) + np.log(1 - x**(1/2))

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

def get_Ab(filter_arr):

    """
    Computes Ab (for computing magnitude) for a given filter.

    Parameters:
        filter_arr: (2, M) array;
                     M wavelengths and value of passband at each WL
        
    Returns:
        Ab: float;

    """

    filter_WL = filter_arr[0]
    filter_vals = filter_arr[1]

    c_As = 2.9979246e18 # speed of light in [Angstrom / s]
    Ab = c_As * np.sum(filter_vals[0:-1]*np.diff(filter_WL)/filter_WL[0:-1])

    return Ab

def get_Ib(filter_arr, beta):

    """
    Computes Ab (for computing magnitude) for a given filter.

    Parameters:
        filter_arr: (2, M) array;
                     M wavelengths and value of passband at each WL
        beta: (N) array, value of UV slope of each halo (function of SFR)
        

    Returns:
        Ib: (N) array;

    """

    filter_WL = filter_arr[0]
    filter_vals = filter_arr[1]

    delta_lambda = np.diff(filter_WL)
    weights = filter_vals[:-1] * delta_lambda
    WL_base = filter_WL[:-1]            

    WL_powers = WL_base[:, np.newaxis] ** (beta[np.newaxis, :] - 1)
    Ib = np.sum(weights[:, np.newaxis] * WL_powers, axis=0)

    # Ib = np.sum(filter_vals[0:-1]*np.diff(filter_WL)*filter_WL[0:-1]**(beta-1))

    return Ib

def r_at_z_cm(z):
    return Planck18.luminosity_distance(z).to('cm').value / (1 + z)

def Bouwens_beta(MUV):
    return get_beta_bouwens14(MUV)

def get_LUV(SFR):
    """
    Converts star formation rate to
    absolute UV magnitude.
    """
    kappa = 1.15e-28
    luv = sfr * 3.1557e7 / kappa
    return luv

def L_to_Mab_unitless(L_1500):
    return 51.64 - 2.5 * np.log10(L_1500)

def read_passband_data_WL(filename):
    """
    Reads numerical wavelength-QE data from a filter passband file,
    ignoring any header lines that start with '#'.

    As a function of wavelegth.

    Parameters:
        filename (str): Path to the filter file.

    Returns:
        data (np.ndarray): Nx2 array of numerical data [wavelength, QE].
    """
    data = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # skip headers and empty lines
            parts = line.split()
            if len(parts) >= 2:
                data.append([float(parts[0]), float(parts[1])])

    return np.array(data).T[0], np.array(data).T[1]

def get_bandpass_mab(z,SFR,La_att,Ab,filter_arr):

    """
    Computes the apparent magnitude observed through a given filter.

    Parameters:
        z: (N) array; redshifts for each halo
        SFR: (N) array; star formation rates for each halo
                        in units of [Msun / yr]
        La_att: (N) array; IGM-attenuated Lya luminosity for each halo
                           must have units of [ergs / s]
        Ab: float; defined as int(Tb * lambda * d_lambda),
                   a constant for a given filter,
                   must have units of [Angstroms / s]
        Tb_az: float; value of the transmission at redshifted lya wavelength
        filter_arr: (2, M) array;
                     M wavelengths and value of passband at each WL
        

    Returns:
        bp_mab: (N) array; apparent ab magnitudes through filter for each halo

    """

    c_As = 2.9979246e18 # speed of light in [Angstrom / s]

    rL = (1+z) * r_at_z_cm(z) # luminosity distance [cm]
    lambda_az = 1215.67 * (1+z) # wavelength of Lya at z [Angstroms]
    lambda_1500 = 1500 # [Angstroms]
    L_1500 = get_LUV(SFR) # in [ergs / s / Hz]
    MUV = L_to_Mab_unitless(L_1500)
    beta = Bouwens_beta(MUV)

    # Ib: float; defined as int(Tb * lambda^{beta+1} * d_lambda),
    #                is a function of the filter and beta(sfr),
    #                must have units of [Angstroms]^{beta}
    Ib = get_Ib(filter_arr, beta)

    # Tb_az; value of bandpass at redshift Lya wavelength
    Tb_az = np.interp(lambda_az, filter_arr[0],filter_arr[1])

    # compute the effective flux density:
    feff = La_att * Tb_az * lambda_az 
    # print(c_As * L_1500 * Ib / ((1+z)**beta * lambda_1500**beta))
    feff += c_As * L_1500 * Ib / ((1+z)**beta * lambda_1500**beta)
    feff *= (4 * np.pi * rL**2 * Ab)**(-1) # [ergs / cm^2]
    # check units:
    #( ergs s^-1 A + A s^-1 ergs A^beta A^-beta ) * (cm^-2 A^-1 s) = ergs cm^-2
    
    return -2.5 * np.log10(feff) - 48.60

# physical constants
wave_Lya = 1215.67*u.AA
freq_Lya = (c.to(u.AA/u.s)/wave_Lya).to('Hz')
omega_Lya = 2*np.pi*freq_Lya
c_kps = c.to('km/s').value
decay_factor = 6.25e8*u.s**-1
wavelength_range = np.linspace(1215.67, 1220, 1000)*u.AA
velocity_range = c_kps*(wavelength_range/wave_Lya - 1)

def I(x):
    return x**(9/2)/(1 - x) + (9/7)*x**(7/2) + (9/5)*x**(5/2) + \
        3*x**(3/2) + 9*x**(1/2) - np.log(1 + x**(1/2)) + np.log(1 - x**(1/2))

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

def get_a(m):
            return 0.65 + 0.1 * np.tanh(3 * (m + 20.75))

def get_wc(m):
    return 31 + 12 * np.tanh(4 * (m + 20.25))

# def mason2018(Muv):
#     """
#     Samples EW and emission probability from the
#     fit functions obtained by Mason et al. 2018.
#     """
#     A = get_a(Muv)
#     rv_A = np.random.uniform(0, 1, len(Muv))
#     emit_bool = rv_A < A
#     Wc = get_wc(Muv)
#     rv_W = np.random.uniform(0, 1, len(Wc))
#     W = -1*Wc*np.log(rv_W)
#     W[~emit_bool] = 0  # Set W to 0 for non-emitting galaxies
#     return W, emit_bool

# muv_t24, muv_emu, muv_dex = np.array([[-19.5, -18.5, -17.5], \
#                                       [10, 16, 27], [1.75, 1.51, 0.99]])
# def mean_t24(muv):
#     mean = np.zeros_like(muv)
#     mean[muv<=-19.5] = muv_emu[0]
#     mean[muv>-17.5] = muv_emu[2]
#     mean[mean==0] = muv_emu[1]
#     return mean

# def sigma_t24(muv):
#     sigma = np.zeros_like(muv)
#     sigma[muv<=-19.5] = muv_dex[0]
#     sigma[muv>-17.5] = muv_dex[2]
#     sigma[sigma==0] = muv_dex[1]
#     return sigma/np.log(10)

# from Bouwens 2021 https://arxiv.org/pdf/2102.07775
phi_5 = 0.79
muv_star_5 = -21.1
alpha_5 = -1.74

def get_beta_bouwens14(muv):
    # https://arxiv.org/pdf/1306.2950
    return -2.05 + -0.2*(muv+19.5)

def schechter(muv, phi, muv_star, alpha):
    return (0.4*np.log(10))*phi*(10**(0.4*(muv_star - muv)))**(alpha+1)*\
        np.exp(-10**(0.4*(muv_star - muv)))
    
print('loading from database')

hdf = h5py.File('../data/id_2001.h5', 'r')
vp_dict = dict(hdf['simulation_parameters']['varying_params'].attrs)
p_dict = dict(hdf['simulation_parameters']['astro_params'].attrs)

aesc = vp_dict['ALPHA_ESC']
fesc10 = 10**vp_dict['F_ESC10']
variable = vp_dict['F_STAR10']
L_X = vp_dict['L_X']
M_turn = vp_dict['M_TURN']
nu_X = vp_dict['NU_X_THRESH']
s8 = vp_dict['SIGMA_8']
sig_sfr_lim = vp_dict['SIGMA_SFR_LIM']
seed = vp_dict['random_seed']
tstar = vp_dict['t_STAR']
Ng = p_dict['POP2_ION']
sig_sfr_idx = p_dict['SIGMA_SFR_INDEX'] 
sig_star = p_dict['SIGMA_STAR']
M_pivot = 10**p_dict['UPPER_STELLAR_TURNOVER_MASS'] # [Msun] 
astar = p_dict['ALPHA_STAR']
astar2 = p_dict['UPPER_STELLAR_TURNOVER_INDEX'] 
fstar10 = 10**p_dict['F_STAR10'] 

def get_stellar_mass(halo_masses, stellar_rng):
    sigma_star = sig_star
    mp1 = 1e10
    mp2 = M_pivot
    a_star = astar
    a_star2 = astar2
    f_star10 = fstar10
    omega_b = Planck18.Ob0
    omega_m = Planck18.Om0
    baryon_frac = omega_b/omega_m
    
    high_mass_turnover = ((mp2/mp1)**a_star + (mp2/mp1)**a_star2)/((halo_masses/mp2)**(-1*a_star)+(halo_masses/mp2)**(-1*a_star2))
    stoc_adjustment_term = 0.5*sigma_star**2
    low_mass_turnover = np.exp(-1*M_turn/halo_masses + stellar_rng*sigma_star - stoc_adjustment_term)
    stellar_mass = f_star10 * baryon_frac * halo_masses * (high_mass_turnover * low_mass_turnover)
    return stellar_mass

def get_sfr(stellar_mass, sfr_rng, z):
    sigma_sfr_lim = sig_sfr_lim
    sigma_sfr_idx = -0.12
    t_h = 1/Planck18.H(z).to('s**-1').value
    t_star = tstar
    sfr_mean = stellar_mass / (t_star * t_h)
    sigma_sfr = sigma_sfr_idx * np.log10(stellar_mass/1e10) + sigma_sfr_lim
    sigma_sfr[sigma_sfr < sigma_sfr_lim] = sigma_sfr_lim
    stoc_adjustment_term = sigma_sfr * sigma_sfr / 2. # adjustment to the mean for lognormal scatter
    sfr_sample = sfr_mean * np.exp(sfr_rng*sigma_sfr - stoc_adjustment_term)
    return sfr_sample

z = 5.7
mh_min = 5e8 # Msol

density = hdf['lightcones']['density'][:]
neutral_fraction = hdf['lightcones']['x_HI'][:]
lightcone_redshifts = dict(hdf['lightcones'].attrs)['lightcone_redshifts']
halo_data_redshifts = np.array(list(hdf['halo_data'].keys()), dtype=float)

z_halo_list = halo_data_redshifts[np.argmin(halo_data_redshifts - z)]

masses = hdf['halo_data'][str(z_halo_list)]['halo_masses'][:]
coords = hdf['halo_data'][str(z_halo_list)]['halo_coords'][:]
sfr_rng_indv = hdf['halo_data'][str(z_halo_list)]['sfr_rng'][:]
stellar_rng_indv = hdf['halo_data'][str(z_halo_list)]['star_rng'][:]

coords = coords[masses > mh_min]
sfr_rng_indv = sfr_rng_indv[masses > mh_min]
stellar_rng_indv = stellar_rng_indv[masses > mh_min]
masses = masses[masses > mh_min]

los_dim = density.shape[-1]
lc_side_voxels = density.shape[0]

# pad lightcones to extend sightlines
pad = np.zeros((lc_side_voxels, lc_side_voxels, 100))
neutral_fraction = np.concatenate([pad, neutral_fraction], axis=-1)
density = np.concatenate([pad, density], axis=-1)

dc_los = Planck18.comoving_distance(5.0).to('Mpc') + \
    np.linspace(0, 1.5*los_dim, los_dim) * u.Mpc
los_z = lightcone_redshifts
los_z = np.concatenate([[z_at_value(Planck18.comoving_distance, d) for d in \
        Planck18.comoving_distance(5.0)-np.linspace(0, 100, 100)[::-1]*u.Mpc], los_z])
dL = Planck18.luminosity_distance(los_z).to('cm').value

n_coeval_centers = int(np.rint(los_dim / lc_side_voxels))
coeval_start_idcs = lc_side_voxels * (np.array(list(range(los_dim))) // lc_side_voxels)
coeval_start_z = los_z[coeval_start_idcs]

z_select = z>coeval_start_z
z_adjust = coeval_start_idcs[np.argmax(coeval_start_z[z_select])] + 100

_x = np.concatenate([coords.T[0], coords.T[0]], axis=0)
_y = np.concatenate([coords.T[1], coords.T[1]], axis=0)
_z = np.concatenate([coords.T[2], coords.T[2] + z_adjust], axis=0)

# masses = np.concatenate([masses, masses], axis=0)
# stellar_rng_indv = np.concatenate([stellar_rng_indv, stellar_rng_indv], axis=0)
# sfr_rng_indv = np.concatenate([sfr_rng_indv, sfr_rng_indv], axis=0)

# purge galaxies in neutral regions
select = neutral_fraction[_x, _y, _z] == 0.0
if np.sum(select) == 0:
    quit()

halo_mass = masses[select]
halo_str_rng = stellar_rng_indv[select]
halo_sfr_rng = sfr_rng_indv[select]
x = _x[select]
y = _y[select]
z = _z[select]

# TODO the above does not work! compare with ngal_from_cache.py

redshifts = los_z[_z]

sfr = get_sfr(get_stellar_mass(halo_mass, halo_str_rng), halo_sfr_rng, redshifts)
muv = get_muv(sfr)

NSAMPLES = len(muv)

print('sampling from SGH model')

# Gagnon-Hartman et al. (2025) model
I = np.array([[1,0,0],[0,1,0],[0,0,1]])
A1 = np.array([[0,0,0],[0,0,-1],[0,1,0]])
A2 = np.array([[0,0,1],[0,0,0],[-1,0,0]])
A3 = np.array([[0,-1,0],[1,0,0],[0,0,0]])
c1, c2, c3, c4 = 1, 1, 1/3, -1
A = c1 * I + c2 * A1 + c3 * A2 + c4 * A3
# xc = np.load('../data/pca/xc.npy')
# xstd = np.load('../data/pca/xstd.npy')
xc = np.array([[42.47, 199.61, 42.03]]).T
xstd = np.array([[0.4222, 99.4, 0.394]]).T
m1, m2, m3, b1, b2, b3, std1, std2, std3 = 6e-3, -0.54, -0.35, -0.34, -0.95, -0.24, 0.56, 0.5, 0.19
# m1, m2, m3, b1, b2, b3, std1, std2, std3, w1, w2, f1, f2, fh = np.load('../data/pca/fit_params.npy')
u1, u2, u3 = np.random.normal(m1*(muv + 18.5) + b1, std1, NSAMPLES), \
            np.random.normal(m2*(muv + 18.5) + b2, std2, NSAMPLES), \
            np.random.normal(m3*(muv + 18.5) + b3, std3, NSAMPLES)
log10lya, dv, log10ha = (A @ np.array([u1, u2, u3]))* xstd + xc

# prefactor for damping wing
ra = (decay_factor/(4*np.pi*delta_nu(1e4*u.K))).to('').value
prefactor = ra/np.pi

# cut out galaxies with dv < vcirc and galaxies which cannot be seen
vc = 10**vcirc(muv)
lum_to_flux = np.log10(4*np.pi*dL[_z]**2)
log10lya_flux = log10lya - lum_to_flux
# apply a pre-selection muv cut
select = (dv>=vc)*(muv <= -16)

muv = muv[select]
lum_to_flux = lum_to_flux[select]
log10lya = log10lya[select]
dv = dv[select]
log10ha = log10ha[select]
_x = _x[select]
_y = _y[select]
_z = _z[select]


# apply IGM attenuation
# 1. gather sightlines
zstart = _z - 100

N_SIGHTLINES = _x.shape[0]

neutral_sightlines = np.array(list(map(lambda xi, yi, zi, ze: neutral_fraction[xi, yi, zi:ze], \
                                        _x, _y, zstart, _z)))
density_sightlines = np.array(list(map(lambda xi, yi, zi, ze: density[xi, yi, zi:ze], \
                                        _x, _y, zstart, _z)))
redshift_sightlines = np.array(list(map(lambda zi, ze: los_z[zi:ze], \
                                        zstart, _z)))
# NOTE we approximate bins as linear in redshift, not strictly true
# beware of this in case of errors where z_rel lies within the nearest bin
redshift_edge_sightlines = np.zeros((redshift_sightlines.shape[0], \
                                        redshift_sightlines.shape[1]+1))
rdiff = 0.5*(redshift_sightlines[:,1] - redshift_sightlines[:,0])
redshift_edge_sightlines[:,0] = redshift_sightlines[:,0] - rdiff
redshift_edge_sightlines[:,1:] = redshift_sightlines + rdiff[:, np.newaxis]
# 2. integrate over sightlines, w/ z_rel using the velocity offset from line center
z_rel = dv / c_kps + los_z[z]
ze = redshift_edge_sightlines[:,:-1]
zb = redshift_edge_sightlines[:,1:]
miralda_escude = ((1+zb)/(1+z_rel[:,np.newaxis]))**(3/2) * \
                        (IME((1+zb)/(1+z_rel[:,np.newaxis])) - IME((1+ze)/(1+z_rel[:,np.newaxis])))
integrand = (1 + density_sightlines)*neutral_sightlines*miralda_escude
integral = np.sum(integrand, axis=1)
tau = integral * prefactor
# transmission_factor = np.exp(-1*integral * prefactor) # maybe we should save this?
# 3. apply transmission fraction in logspace
plt.hist(log10lya, bins=50, color='blue', alpha=0.8)
log10lya -= integral * prefactor * np.log10(np.e)

plt.hist(log10lya, bins=50, color='orange', alpha=0.8)
plt.show()
quit()



z_LAE_z5_SAM = z
SFR_z5_SAM = sfr
La_att_z5_SAM = 10**log10lya
NB816_WL = read_passband_data_WL('../data/filters/NB816SuprimeCam.txt')
Ab_NB816 = get_Ab(NB816_WL)

print('getting bandpass')

# compute magnitudes through narrowband and broadband filters
mab_NB816_Sam = get_bandpass_mab(z_LAE_z5_SAM, SFR_z5_SAM, La_att_z5_SAM,
                                       Ab_NB816, NB816_WL)
NB816_mask_SAM_uniform = mab_NB816_Sam < 26.
observed_mab_NB816_Sam_small = mab_NB816_Sam[NB816_mask_SAM_uniform]
num = len(observed_mab_NB816_Sam_small)

print(num)

