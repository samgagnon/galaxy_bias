import os

import numpy as np
import py21cmfast as p21c

from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, k_B, m_p, e, m_e

from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from scipy.special import gamma, erf
from scipy.optimize import differential_evolution

from sklearn.decomposition import PCA

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

# from Bouwens 2021 https://arxiv.org/pdf/2102.07775
phi_5 = 0.79
muv_star_5 = -21.1
alpha_5 = -1.74

def schechter(muv, phi, muv_star, alpha):
    return (0.4*np.log(10))*phi*(10**(0.4*(muv_star - muv)))**(alpha+1)*\
        np.exp(-10**(0.4*(muv_star - muv)))

def p_muv(muv, phi, muv_star, alpha):
    return schechter(muv, phi, muv_star, alpha)/gamma(alpha_5+2)/(0.4*np.log(10))/phi_5

# measured lya properties from https://arxiv.org/pdf/2402.06070
MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, fescA, \
    fescA_err, fescB, fescB_err, ID = np.load('./data/tang24.npy').T

lum_lya = (ew_lya/1215.67) * 2.47e15 * 10**(0.4*(51.6 - MUV))
lum_lya_err = lum_lya*np.sqrt((ew_lya_err/ew_lya)**2 + (np.exp(-0.4*np.log(10)*MUV_err)/MUV)**2)

lum_ha = lum_lya / 11.4 / fescA
lum_ha_err = lum_ha*np.sqrt((lum_lya_err/lum_lya)**2 + (fescA_err/fescA)**2)

lum_flux_factor = 4*np.pi*(Planck18.luminosity_distance(5.0).to('cm').value)**2

wide = ID==0
_muv_wide = MUV[wide]
_muv_err_wide = MUV_err[wide]
_dv_lya_wide = dv_lya[wide]
_dv_lya_err_wide = dv_lya_err[wide]
_lum_lya_wide = lum_lya[wide]
_lum_lya_err_wide = lum_lya_err[wide]
_lum_ha_wide = lum_ha[wide]
_lum_ha_err_wide = lum_ha_err[wide]

deep = ID==1
_muv_deep = MUV[deep]
_muv_err_deep = MUV_err[deep]
_dv_lya_deep = dv_lya[deep]
_dv_lya_err_deep = dv_lya_err[deep]
_lum_lya_deep = lum_lya[deep]
_lum_lya_err_deep = lum_lya_err[deep]
_lum_ha_deep = lum_ha[deep]
_lum_ha_err_deep = lum_ha_err[deep]

muv_wide = np.zeros(len(_muv_wide)*1000)
lum_lya_wide = np.zeros_like(muv_wide)
dv_wide = np.zeros_like(muv_wide)
lum_ha_wide = np.zeros_like(muv_wide)

muv_deep = np.zeros(len(_muv_deep)*1000)
lum_lya_deep = np.zeros_like(muv_deep)
dv_deep = np.zeros_like(muv_deep)
lum_ha_deep = np.zeros_like(muv_deep)

for i, (muv, muve, lly, llye, dv, dve, lha, lhae) in enumerate(zip(_muv_wide, _muv_err_wide, _lum_lya_wide, _lum_lya_err_wide, \
                                                 _dv_lya_wide, _dv_lya_err_wide, _lum_ha_wide, _lum_ha_err_wide)):
    muv_wide[1000*i:1000*(i+1)] = np.random.normal(muv, muve, 1000)
    lum_lya_wide[1000*i:1000*(i+1)] = np.random.normal(lly, llye, 1000)
    dv_wide[1000*i:1000*(i+1)] = np.random.normal(dv, dve, 1000)
    lum_ha_wide[1000*i:1000*(i+1)] = np.random.normal(lha, lhae, 1000)
    lum_ha_wide[1000*i:1000*(i+1)][lum_ha_wide[1000*i:1000*(i+1)]<10] = np.mean(lum_ha_wide[1000*i:1000*(i+1)])  # replace with mean
    lum_lya_wide[1000*i:1000*(i+1)][lum_lya_wide[1000*i:1000*(i+1)]<10] = np.mean(lum_lya_wide[1000*i:1000*(i+1)])  # replace with mean

    
for i, (muv, muve, lly, llye, dv, dve, lha, lhae) in enumerate(zip(_muv_deep, _muv_err_deep, _lum_lya_deep, _lum_lya_err_deep, \
                                                 _dv_lya_deep, _dv_lya_err_deep, _lum_ha_deep, _lum_ha_err_deep)):
    muv_deep[1000*i:1000*(i+1)] = np.random.normal(muv, muve, 1000)
    lum_lya_deep[1000*i:1000*(i+1)] = np.random.normal(lly, llye, 1000)
    dv_deep[1000*i:1000*(i+1)] = np.random.normal(dv, dve, 1000)
    lum_ha_deep[1000*i:1000*(i+1)] = np.random.normal(lha, lhae, 1000)
    lum_ha_deep[1000*i:1000*(i+1)][lum_ha_deep[1000*i:1000*(i+1)]<10] = np.mean(lum_ha_deep[1000*i:1000*(i+1)])  # replace with mean
    lum_lya_deep[1000*i:1000*(i+1)][lum_lya_deep[1000*i:1000*(i+1)]<10] = np.mean(lum_lya_deep[1000*i:1000*(i+1)])  # replace with mean

def normal_cdf(x, mu=0):
    """
    Cumulative distribution function for a normal distribution.
    """
    return 0.5 * (1 + erf((x - mu + mu/5) / (mu/5 * np.sqrt(2))))

def p_obs(lly, lha, muv, mode='wide'):
    """
    Probability of observing a galaxy with given Lya luminosity, H-alpha luminosity, and UV magnitude.
    """
    # Convert luminosities to fluxes
    f_lya = lly / lum_flux_factor
    f_ha = lha / lum_flux_factor
    luv = 10**(0.4*(51.64 - muv))
    w_emerg = (1215.67/2.47e15)*(lly/luv)
    f_ha_lim = 2e-18  # H-alpha flux limit in erg/s/cm^2
    if mode == 'wide':
        w_lim = 80
        f_lya_lim = 2e-17
    elif mode == 'deep':
        w_lim = 8
        f_lya_lim = 2e-18
    p_lya = normal_cdf(f_lya, f_lya_lim)
    p_ha = normal_cdf(f_ha, f_ha_lim)
    p_w = normal_cdf(w_emerg, w_lim)
    
    return p_lya * p_ha * p_w

def p_f(_p_obs, f, fe):
    _f = np.sum(_p_obs) / len(_p_obs)
    _f_err = _f * np.sqrt((np.sqrt(len(_p_obs)) / len(_p_obs))**2 + (fe/f)**2)
    dist = np.exp(-0.5 * ((f - _f) / _f_err) ** 2)
    if np.isinf(dist):
        dist = 100
    # get Gaussian distance from expected observed fraction
    return dist


def multivar_normal_pdf(x, mu1, mu2, mu3, std1, std2, std3):
    """
    Multivariate normal probability density function.
    """
    coeff = 1 / (np.sqrt(2 * np.pi) ** 3 * std1 * std2 * std3)
    # coeff = 1
    exponent = -0.5 * ((x[0] - mu1) ** 2 / std1 ** 2 + 
                       (x[1] - mu2) ** 2 / std2 ** 2 + 
                       (x[2] - mu3) ** 2 / std3 ** 2)
    return coeff * np.exp(exponent)

XWIDE = np.array([np.log10(lum_lya_wide), dv_wide, np.log10(lum_ha_wide)])
XDEEP = np.array([np.log10(lum_lya_deep), dv_deep, np.log10(lum_ha_deep)])

# Y = PCA T of X
os.makedirs('./data/pca', exist_ok=True)

pca = PCA(n_components=3)  # Keeping all 3 dimensions

# transform the wide sample to the PCA basis
XALL = np.concatenate((XWIDE, XDEEP), axis=1)

XC = XALL.mean(axis=1, keepdims=True)
np.save('./data/pca/xc.npy', XC)
XSTD = XALL.std(axis=1, keepdims=True)
np.save('./data/pca/xstd.npy', XSTD)
XALL0 = (XALL - XC) / XSTD
pca.fit(XALL0.T)

T = pca.components_
np.save('./data/pca/T.npy', T)

YALL = T @ XALL0

XWIDE0 = XWIDE - XC
XDEEP0 = XDEEP - XC

from tqdm import tqdm

def fit_sample():
    """
    Fits the sample data to a multivariate normal distribution.
    Fits parameters for each MUV bin using a maximum likelihood approach.
    """

    all_params = np.zeros((20, 6))
    all_params_var = np.zeros((20, 6))
    all_n_points = np.zeros((20, 2))
    muv_centers = np.linspace(-20, -16, 20)
    f = np.zeros((20, 2))
    f_err = np.zeros((20, 2))
    for i, muv in tqdm(enumerate(muv_centers)):

        tqdm.write(f'Fitting MUV bin {i+1}/{len(muv_centers)}: {muv:.2f}')

        muv_range = np.linspace(muv-0.5, muv+0.5, 1000)
        GALDENS = trapezoid(schechter(muv_range, phi_5, muv_star_5, alpha_5), muv_range)

        VOL_WIDE = 146.082  # 10^3 Mpc^3
        n_gal = GALDENS * VOL_WIDE
        n_gal_rel_err = np.sqrt(n_gal)/n_gal
        n_muv_wide = np.sum(np.abs(muv_wide - muv) < 0.5)/1000
        n_muv_wide_rel_err = np.sqrt(n_muv_wide)/n_muv_wide

        f[i,0] = n_muv_wide / n_gal
        f_err[i,0] = f[i,0] * np.sqrt(n_gal_rel_err**2 + n_muv_wide_rel_err**2)
        
        VOL_DEEP = 7.877 # 10^3 Mpc^3
        n_gal = GALDENS * VOL_DEEP
        n_gal_rel_err = np.sqrt(n_gal)/n_gal
        n_muv_deep = np.sum(np.abs(muv_deep - muv) < 0.5)/1000
        n_muv_deep_rel_err = np.sqrt(n_muv_deep)/n_muv_deep

        f[i,1] = n_muv_deep / n_gal
        f_err[i,1] = f[i,1] * np.sqrt(n_gal_rel_err**2 + n_muv_deep_rel_err**2)

        select_wide = np.abs(muv_wide - muv) < 0.5
        select_deep = np.abs(muv_deep - muv) < 0.5
        _muv_wide = muv_wide[select_wide]
        _muv_deep = muv_deep[select_deep]
        
        X_MUV_WIDE = XWIDE[:, select_wide]
        Y_MUV_WIDE = T @ ((X_MUV_WIDE - XC)/XSTD)

        X_MUV_DEEP = XDEEP[:, select_deep]
        Y_MUV_DEEP = T @ ((X_MUV_DEEP - XC)/XSTD)

        n_points = X_MUV_WIDE.shape[1]
        all_n_points[i] = n_points

        def p_x(x, y, f, fe, _muv, params, mode):
            mu1, mu2, mu3, std1, std2, std3 = params

            y1 = np.random.normal(mu1, std1, 1000)
            y2 = np.random.normal(mu2, std2, 1000)
            y3 = np.random.normal(mu3, std3, 1000)
            Y = np.stack([y1, y2, y3])
            X = (np.linalg.inv(T) @ Y) * XSTD + XC
            X = (np.linalg.inv(T) @ Y) * XSTD + XC
            lly, dv, lha = X
            _p_obs = p_obs(10**lly, 10**lha, muv, mode=mode)
            mean_lya = np.sum(lly*_p_obs) / np.sum(_p_obs)
            mean_dv = np.sum(dv*_p_obs) / np.sum(_p_obs)
            mean_ha = np.sum(lha*_p_obs) / np.sum(_p_obs)
            std_lya = np.sqrt(np.sum(_p_obs * (lly - mean_lya)**2) / np.sum(_p_obs))
            std_dv = np.sqrt(np.sum(_p_obs * (dv - mean_dv)**2) / np.sum(_p_obs))
            std_ha = np.sqrt(np.sum(_p_obs * (lha - mean_ha)**2) / np.sum(_p_obs))

            smoothing = 1e-6
            # TODO add term to check observed fraction
            _p_y = multivar_normal_pdf(x, mean_lya, mean_dv, mean_ha, \
                                       std_lya, std_dv, std_ha) + smoothing
            _p_f = p_f(_p_obs, f, fe) + smoothing
            return np.log10(_p_y) + np.log10(_p_f)
        
        def loss(params):
            _p_x_wide = p_x(X_MUV_WIDE, Y_MUV_WIDE, f[i,0], f_err[i,0], _muv_wide, params, mode='wide')
            _p_x_deep = p_x(X_MUV_DEEP, Y_MUV_DEEP, f[i,1], f_err[i,1], _muv_deep, params, mode='deep')
            return -1*np.sum(_p_x_wide) -1*np.sum(_p_x_deep)

        bounds = [(-50, 50), (-50, 50), (-50, 50), (0.01, 10), (0.01, 10), (0.01, 10)]
        res = differential_evolution(loss, bounds, maxiter=1000, disp=True)
        all_params[i, :] = res.x

        mu1, mu2, mu3, std1, std2, std3 = res.x

        y1 = np.random.normal(mu1, std1, 1000)
        y2 = np.random.normal(mu2, std2, 1000)
        y3 = np.random.normal(mu3, std3, 1000)
        Y = np.stack([y1, y2, y3])
        X = (np.linalg.inv(T) @ Y) * XSTD + XC
        X = (np.linalg.inv(T) @ Y) * XSTD + XC
        lly, dv, lha = X
        _p_obs = p_obs(10**lly, 10**lha, muv, mode='wide')
        mean_lya = np.sum(lly*_p_obs) / np.sum(_p_obs)
        mean_dv = np.sum(dv*_p_obs) / np.sum(_p_obs)
        mean_ha = np.sum(lha*_p_obs) / np.sum(_p_obs)
        std_lya = np.sqrt(np.sum(_p_obs * (lly - mean_lya)**2) / np.sum(_p_obs))
        std_dv = np.sqrt(np.sum(_p_obs * (dv - mean_dv)**2) / np.sum(_p_obs))
        std_ha = np.sqrt(np.sum(_p_obs * (lha - mean_ha)**2) / np.sum(_p_obs))

        plt.scatter(lly, lha, color='yellow', alpha=0.3, s=2)
        plt.scatter(X_MUV_WIDE[0], X_MUV_WIDE[2], color='red', alpha=0.3, s=2)
        plt.scatter(lly[_p_obs>0.5], lha[_p_obs>0.5], color='orange', alpha=0.3, s=2)
        plt.errorbar(mean_lya, mean_ha, xerr=std_lya, yerr=std_ha, \
                        color='white', marker='x', markersize=25)
        plt.text(0.8, 0.8, f'{np.sum(_p_obs)/len(_p_obs):.2f}', color='white', fontsize=20)
        plt.show()

    return muv_centers, all_params, all_n_points, f, f_err

muv_centers, all_params, all_n_points, f, f_err = fit_sample()

np.save('./data/pca/muv.npy', muv_centers)
np.save('./data/pca/params.npy', all_params)
np.save('./data/pca/n_points.npy', all_n_points)
np.save('./data/pca/f.npy', f)
np.save('./data/pca/f_err.npy', f_err)

quit()

def plot_fit(muv_centers, all_params, all_n_points, mode='wide'):
    """
    Plots the fitted parameters for each MUV bin.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig.suptitle(f'Fitted Parameters for {mode.capitalize()} Sample', fontsize=20)

    # Plot mu1
    axs[0].errorbar(muv_centers, all_params[:, 0], yerr=all_params[:, 3], fmt='o', label='mu1')
    axs[0].set_ylabel('mu1 (log10 Lya Luminosity)')
    axs[0].legend()
    
    # Plot mu2
    axs[1].errorbar(muv_centers, all_params[:, 1], yerr=all_params[:, 4], fmt='o', label='mu2')
    axs[1].set_ylabel('mu2 (log10 Lya Velocity Dispersion)')
    axs[1].legend()
    
    # Plot mu3
    axs[2].errorbar(muv_centers, all_params[:, 2], yerr=all_params[:, 5], fmt='o', label='mu3')
    axs[2].set_ylabel('mu3 (log10 H-alpha Luminosity)')
    axs[2].set_xlabel('MUV')
    axs[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

plot_fit(muv_centers_wide, all_params_wide, all_n_points_wide, mode='wide')
plot_fit(muv_centers_deep, all_params_deep, all_n_points_deep, mode='deep')