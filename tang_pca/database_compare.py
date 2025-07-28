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

def get_a(m):
            return 0.65 + 0.1 * np.tanh(3 * (m + 20.75))

def get_wc(m):
    return 31 + 12 * np.tanh(4 * (m + 20.25))

def mason2018(Muv):
    """
    Samples EW and emission probability from the
    fit functions obtained by Mason et al. 2018.
    """
    A = get_a(Muv)
    rv_A = np.random.uniform(0, 1, len(Muv))
    emit_bool = rv_A < A
    Wc = get_wc(Muv)
    rv_W = np.random.uniform(0, 1, len(Wc))
    W = -1*Wc*np.log(rv_W)
    W[~emit_bool] = 0  # Set W to 0 for non-emitting galaxies
    return W, emit_bool

muv_t24, muv_emu, muv_dex = np.array([[-19.5, -18.5, -17.5], \
                                      [10, 16, 27], [1.75, 1.51, 0.99]])
def mean_t24(muv):
    mean = np.zeros_like(muv)
    mean[muv<=-19.5] = muv_emu[0]
    mean[muv>-17.5] = muv_emu[2]
    mean[mean==0] = muv_emu[1]
    return mean

def sigma_t24(muv):
    sigma = np.zeros_like(muv)
    sigma[muv<=-19.5] = muv_dex[0]
    sigma[muv>-17.5] = muv_dex[2]
    sigma[sigma==0] = muv_dex[1]
    return sigma/np.log(10)

def get_silverrush_laelf(z):
    if z==4.9:
        # SILVERRUSH XIV z=4.9 LAELF
        lum_silver = np.array([42.75, 42.85, 42.95, 43.05, 43.15, 43.25, 43.35, 43.45, 43.55, 43.65])
        logphi_silver = -1*np.array([2.91, 3.17, 3.42, 3.78, 3.88, 4.00, 4.75, 4.93, 5.23, 4.93])
        logphi_up_silver = 1e-2*np.array([5, 5, 6, 9, 10, 12, 29, 36, 52, 36])
        logphi_low_silver = 1e-2*np.array([5, 5, 6, 9, 10, 12, 34, 45, 77, 45])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    elif z==5.7:
        # SILVERRUSH XIV z=5.7 LAELF
        lum_silver = np.array([42.85, 42.95, 43.05, 43.15, 43.25, 43.35, 43.45, 43.55, 43.65, 43.75, 43.85, 43.95])
        logphi_silver = -1*np.array([3.05, 3.27, 3.56, 3.85, 4.15, 4.41, 4.72, 5.15, 5.43, 6.03, 6.33, 6.33])
        logphi_up_silver = 1e-2*np.array([4, 2, 2, 3, 4, 5, 7, 12, 17, 36, 52, 52])
        logphi_low_silver = 1e-2*np.array([4, 2, 2, 3, 4, 5, 7, 13, 18, 45, 77, 77])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    elif z==6.6:
        # SILVERRUSH XIV z=6.6 LAELF
        lum_silver = np.array([42.95, 43.05, 43.15, 43.25, 43.35, 43.45, 43.55, 43.65, 43.75, 43.95, 44.05])
        logphi_silver = -1*np.array([3.71, 4.11, 4.37, 4.65, 4.83, 5.28, 5.89, 5.9, 5.9, 6.38, 6.38])
        logphi_up_silver = 1e-2*np.array([9, 5, 6, 7, 8, 14, 29, 29, 29, 52, 52])
        logphi_low_silver = 1e-2*np.array([9, 5, 6, 7, 8, 15, 34, 34, 34, 77, 77])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    elif z==7.0:
        # wip
        # SILVERRUSH XIV z=7.0 LAELF
        lum_silver = np.array([43.25, 43.35])
        logphi_silver = -1*np.array([4.4, 4.95])
        logphi_up_silver = 1e-2*np.array([29, 52])
        logphi_low_silver = 1e-2*np.array([34, 77])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    elif z==7.3:
        # wip
        # SILVERRUSH XIV z=7.3 LAELF
        lum_silver = np.array([43.45])
        logphi_silver = -1*np.array([4.81])
        logphi_up_silver = 1e-2*np.array([36])
        logphi_low_silver = 1e-2*np.array([45])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    
lum, logphi, logphi_up, logphi_low = get_silverrush_laelf(5.7)
bin_edges = np.zeros(len(lum) + 1)
bin_edges[0] = lum[0] - 0.5*(lum[1] - lum[0])
bin_edges[1:-1] = 0.5*(lum[1:] + lum[:-1])
bin_edges[-1] = lum[-1] + 0.5*(lum[-1] - lum[-2])

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

density_lightcone = hdf['lightcones']['density'][:]
x_HI_lightcone = hdf['lightcones']['x_HI'][:]
lightcone_redshifts = dict(hdf['lightcones'].attrs)['lightcone_redshifts']
halo_data_redshifts = np.array(list(hdf['halo_data'].keys()), dtype=float)
los_z = lightcone_redshifts

z_halo_list = halo_data_redshifts[np.argmin(halo_data_redshifts - z)]

masses = hdf['halo_data'][str(z_halo_list)]['halo_masses'][:]
coords = hdf['halo_data'][str(z_halo_list)]['halo_coords'][:]
sfr_rng_indv = hdf['halo_data'][str(z_halo_list)]['sfr_rng'][:]
stellar_rng_indv = hdf['halo_data'][str(z_halo_list)]['star_rng'][:]

coords = coords[masses > mh_min]
sfr_rng_indv = sfr_rng_indv[masses > mh_min]
stellar_rng_indv = stellar_rng_indv[masses > mh_min]
masses = masses[masses > mh_min]

los_dim = density_lightcone.shape[-1]

dc_los = Planck18.comoving_distance(5.0).to('Mpc') + \
    np.linspace(0, 1.5*los_dim, los_dim) * u.Mpc
# los_z = np.array([z_at_value(Planck18.comoving_distance, d).value for d in dc_los])
n_coeval_centers = int(np.rint(2000 / 200))
coeval_start_idcs = np.array(list(range(n_coeval_centers))) * 200
coeval_start_z = los_z[coeval_start_idcs]
z_adjust = coeval_start_idcs[np.argmax(coeval_start_z[coeval_start_z<z])]

_x = np.concatenate([coords.T[0], coords.T[0]], axis=0)
_y = np.concatenate([coords.T[1], coords.T[1]], axis=0)
_z = np.concatenate([coords.T[2], coords.T[2] + z_adjust], axis=0)

masses = np.concatenate([masses, masses], axis=0)
stellar_rng_indv = np.concatenate([stellar_rng_indv, stellar_rng_indv], axis=0)
sfr_rng_indv = np.concatenate([sfr_rng_indv, sfr_rng_indv], axis=0)
z = los_z[_z]

sfr = get_sfr(get_stellar_mass(masses, stellar_rng_indv), sfr_rng_indv, z)
muv = get_muv(sfr)

NSAMPLES = len(muv)
EFFECTIVE_VOLUME = 2*300**3

# Mason et al. (2018) model
w_m18, emit_bool_m18 = mason2018(muv)
log10lya_m18 = np.log10(w_m18*(2.47e15/1215.67)*(1500/1215.67)**(get_beta_bouwens14(muv)+2)*\
    10**(0.4*(51.6-muv)))
heights_m18, bins_m18 = np.histogram(log10lya_m18, bins=bin_edges, density=False)
bin_widths = bins_m18[1:]-bins_m18[:-1]
height_err_m18 = np.sqrt(heights_m18) / bin_widths / EFFECTIVE_VOLUME
heights_m18 = heights_m18 / bin_widths / EFFECTIVE_VOLUME
logphi_m18 = np.log10(heights_m18)
logphi_up_m18 = np.abs(np.log10(height_err_m18 + heights_m18) - logphi_m18)
logphi_low_m18 = np.abs(logphi_m18 - np.log10(heights_m18 - height_err_m18))

# Tang et al. (2024) model
w_t24 = np.random.lognormal(mean=np.log(mean_t24(muv)),
                            sigma=sigma_t24(muv),
                            size=NSAMPLES)
log10lya_t24 = np.log10(w_t24*(2.47e15/1215.67)*(1215.67/1500)**(get_beta_bouwens14(muv)+2)*\
    10**(0.4*(51.6-muv)))
heights_t24, bins_t24 = np.histogram(log10lya_t24, bins=bin_edges, density=False)
bin_widths = bins_t24[1:]-bins_t24[:-1]
height_err_t24 = np.sqrt(heights_t24) / bin_widths / EFFECTIVE_VOLUME
heights_t24 = heights_t24 / bin_widths / EFFECTIVE_VOLUME
logphi_t24 = np.log10(heights_t24)
logphi_up_t24 = np.abs(np.log10(height_err_t24 + heights_t24) - logphi_t24)
logphi_low_t24 = np.abs(logphi_t24 - np.log10(heights_t24 - height_err_t24))

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
w_sgh = (1215.67/2.47e15)*(10**log10lya)*10**(-0.4*(51.6-muv))*\
    (1215.67/1500)**(-1*get_beta_bouwens14(muv)-2)

# d_side = np.linspace(0, 300, 200)
# plt.pcolormesh(d_side, d_side, 1-x_HI_lightcone[:,:,0].T, cmap='inferno')
# # plt.imshow(1 - x_HI_lightcone[:, :, 0].T, origin='lower', cmap='inferno')
# plt.scatter(1.5*_x[(z<5.001)*(log10lya>41)], \
#             1.5*_y[(z<5.001)*(log10lya>41)], c='lime', marker='x', s=40)
# plt.colorbar(label='Neutral Hydrogen Fraction')
# plt.show()
# quit()

heights_sgh, bins_sgh = np.histogram(log10lya, bins=bin_edges, density=False)
bin_widths = bins_sgh[1:]-bins_sgh[:-1]
height_err_sgh = np.sqrt(heights_sgh) / bin_widths / EFFECTIVE_VOLUME
heights_sgh = heights_sgh / bin_widths / EFFECTIVE_VOLUME
logphi_sgh = np.log10(heights_sgh)
logphi_up_sgh = np.abs(np.log10(heights_sgh + height_err_sgh) - logphi_sgh)
logphi_low_sgh = np.abs(logphi_sgh - np.log10(heights_sgh - height_err_sgh))



# z, sfr, log10lya_sgh, log10lya_m18, log10lya_t24
print(len(z), len(sfr), len(log10lya), len(log10lya_m18), len(log10lya_t24))
# out = np.array([_x, _y, z, sfr, log10lya, log10lya_m18, log10lya_t24, masses])
out = np.array([_x, _y, z, sfr, log10lya, u1, u2, u3, log10lya_m18, log10lya_t24, masses])
np.save('/mnt/c/Users/sgagn/Downloads/output.npy', out)
# inp = np.load('/mnt/c/Users/sgagn/Downloads/output.npy')
quit()

fig, ax = plt.subplots(figsize=(6, 6.5), constrained_layout=True)
ax.errorbar(lum, logphi, yerr=[logphi_low, logphi_up], 
            fmt='o', markeredgewidth=2, markersize=20, fillstyle='none', color=color4, label='Umeda+25')
ax.errorbar(lum, logphi_m18, yerr=[logphi_low_m18, logphi_up_m18],
            fmt='*', markeredgewidth=2, markersize=20, fillstyle='none', color=color3, label='Mason+18')
ax.errorbar(lum, logphi_t24, yerr=[logphi_low_t24, logphi_up_t24],
            fmt='*', markeredgewidth=2, markersize=20, fillstyle='none', color=color2, label='Tang+24')
ax.errorbar(lum, logphi_sgh, yerr=[logphi_low_sgh, logphi_up_sgh],
            fmt='*', markeredgewidth=2, markersize=20, fillstyle='none', color=color1, label='This Work')
ax.set_xlabel(r'$\log_{10} L_{\rm Ly\alpha}$ [erg s$^{-1}$]', fontsize=font_size)
ax.set_ylabel(r'$\log_{10} \phi$ [Mpc$^{-3}$]', fontsize=font_size)
ax.set_ylim(-8, -2)
ax.legend(fontsize=int(font_size/1.5), loc='lower left')
plt.show()