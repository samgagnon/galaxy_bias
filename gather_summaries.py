import os
import powerbox

import numpy as np
import py21cmfast as p21c

from scipy import integrate, special
from scipy.optimize import minimize
from astropy.cosmology import Planck18, z_at_value
from astropy import units as U, constants as c

from rng2sfr import *
from data import *
from tau_igm import get_absorption_properties, get_tau_igm

def get_A(m):
            return 0.65 + 0.1 * np.tanh(3 * (m + 20.75))

def get_Wc(m):
    return 31 + 12 * np.tanh(4 * (m + 20.25))

def probability_EW(W, Wc):
    return (1/Wc) * np.exp(-W/Wc)

# Bouwens 2014
def get_beta(Muv):
    a = -2.05
    b = -0.2
    return a + b*(Muv+19.5)

def sfr2Muv(sfr):
    kappa = 1.15e-28
    Luv = sfr * 3.1557e7 / kappa
    Muv = 51.64 - np.log10(Luv) / 0.4
    return Muv

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

def lu2024(Mh, Muv, z):
    """
    Samples EW and emission probability from the
    fit functions obtained by Lu et al. 2024.
    """
    A = get_A(Muv)
    rv_A = np.random.uniform(0, 1, len(Muv))
    emit_bool = rv_A < A

    # this takes the halo mass!
    m = 0.32
    C = 2.48
    dv_mean = m*(np.log10(Mh) - np.log10(1.55) - 12) + C
    dv_sigma = 0.24
    dv = np.random.normal(dv_mean, dv_sigma, len(Mh))
    # sample velocity offsets
    # apply truncation at v_circ
    v_circ = ((100*C.G*Mh*U.solMass*Planck18.H(z)).to('').value)**(1/3)

    Wc = get_Wc(Muv[emit_bool])
    rv_W = np.random.uniform(0, 1, len(Wc))
    W = -1*Wc*np.log(rv_W)
    return W, emit_bool

def get_PTH_fn(rs):
    """
    Given a random seed (rs), return the relevant PTH file.
    This must be modified to instead take redshift as an argument.
    """
    path = '../lya_langevin/auxiliary_fields/'
    field_list = os.listdir(path)
    field_list = [field for field in field_list if len(field.split('_'))==3]
    field_dict = {}
    for field in field_list:
        rs = int(field.split('_')[-1][2:])
        field_dict[rs] = field
    item_list = os.listdir(f'{path}{field_dict[rs]}')

    for item in item_list:
        if item.startswith('PerturbHaloField'):
            return f'{path}{field_dict[rs]}/{item}'
        
def gather_hmf(halo_masses, SIDE_LENGTH_MPC, summary_dir):
    heights, bins = np.histogram(np.log10(halo_masses), bins=100)
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    bin_widths = bins[1:]-bins[:-1]
    hmf = heights/bin_widths/(SIDE_LENGTH_MPC**3)
    hmf_err = np.sqrt(heights)/bin_widths/(SIDE_LENGTH_MPC**3)

    log_hmf_err_up = np.abs(np.log10(hmf_err+hmf) - np.log10(hmf))
    log_hmf_err_low = np.abs(np.log10(np.abs(hmf-hmf_err)) - np.log10(hmf))
    log_hmf_err_low[np.isinf(log_hmf_err_low)] = np.abs(np.log10(hmf[np.isinf(log_hmf_err_low)]))
    log_hmf_asymmetric_error = np.array(list(zip(log_hmf_err_low, log_hmf_err_up))).T

    # maybe I should add a flag for redshift? subdirectories would be better
    # this looks like it could cause dimension errors
    np.save(f'{summary_dir}/hmf_log10m.npy', bin_centers)
    np.save(f'{summary_dir}/hmf_dndm.npy', hmf)
    np.save(f'{summary_dir}/hmf_dndm_err.npy', log_hmf_asymmetric_error)

def get_muv(halo_masses, sfr, mason_muv):
    if mason_muv:
        muv = (-1/0.3)*(np.log10(halo_masses) \
                - 11.75) - 20.0 - 0.26*redshift
    else:
        muv = sfr2Muv(sfr)
    return muv
        
def gather_uvlf(muv, SIDE_LENGTH_MPC, summary_dir):
    bin_edges = []
    bin_centers = np.asarray(b21_mag[1])
    bin_edge = np.zeros(len(bin_centers)+1)
    bin_widths = (bin_centers[1:] - bin_centers[:-1])*0.5
    bin_edge[0] = bin_centers[0]-bin_widths[0]
    bin_edge[-1] = bin_centers[-1] + bin_widths[-1]
    bin_edge[1:-1] = bin_widths + bin_centers[:-1]
    bin_edges += [bin_edge]

    heights, bins = np.histogram(muv, bins=bin_edges[0])
    # heights, bins = np.histogram(Llya)
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    bin_widths = bins[1:]-bins[:-1]
    uv_phi = heights/bin_widths/(SIDE_LENGTH_MPC**3)
    uv_phi_err = np.sqrt(heights)/bin_widths/(SIDE_LENGTH_MPC**3)
    log_uv_phi_err_up = np.abs(np.log10(uv_phi_err+uv_phi) - np.log10(uv_phi))
    log_uv_phi_err_low = np.abs(np.log10(np.abs(uv_phi-uv_phi_err)) - np.log10(uv_phi))
    log_uv_phi_err_low[np.isinf(log_uv_phi_err_low)] = np.abs(np.log10(uv_phi[np.isinf(log_uv_phi_err_low)]))
    log_uv_asymmetric_error = np.array(list(zip(log_uv_phi_err_low, log_uv_phi_err_up))).T

    np.save(f'{summary_dir}/uv_phi.npy', uv_phi)
    np.save(f'{summary_dir}/uv_phi_err.npy', log_uv_asymmetric_error)
    np.save(f'{summary_dir}/uv_bin_centers.npy', bin_centers)

def get_lya_luminosity(muv):
    # obtain equivalent widths via Mason 2018 fit
    w, emit_bool = mason2018(muv)
    # convert equivalent width to lyman alpha luminosity
    beta = get_beta(muv[emit_bool])
    const = 2.47 * 1e15 * U.Hz / 1216 / U.Angstrom * (1500 / 1216) \
        ** (-(beta) - 2)
    luv_mean = 10 ** (-0.4 * (muv[emit_bool] - 51.6))
    lya_lum = w * const.value * luv_mean
    return lya_lum, w, emit_bool

def get_min_lum(redshift):
    lum_silver, _, _, _ = get_silverrush_laelf(redshift)
    bin_centers = np.asarray(lum_silver)
    bin_edge = np.zeros(len(bin_centers)+1)
    bin_widths = (bin_centers[1:] - bin_centers[:-1])*0.5
    bin_edge[0] = bin_centers[0]-bin_widths[0]
    min_lum = bin_edge[0]
    return min_lum

def gather_laelf(lya_lum, redshift, SIDE_LENGTH_MPC, summary_dir):
    lum_silver, _, _, _ = get_silverrush_laelf(redshift)

    bin_edges = []
    bin_centers = np.asarray(lum_silver)
    bin_edge = np.zeros(len(bin_centers)+1)
    bin_widths = (bin_centers[1:] - bin_centers[:-1])*0.5
    bin_edge[0] = bin_centers[0]-bin_widths[0]
    min_lum = bin_edge[0]
    bin_edge[-1] = bin_centers[-1] + bin_widths[-1]
    bin_edge[1:-1] = bin_widths + bin_centers[:-1]
    bin_edges += [bin_edge]

    heights, bins = np.histogram(np.log10(lya_lum), bins=bin_edges[0])
    # heights, bins = np.histogram(Llya)
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    bin_widths = bins[1:]-bins[:-1]
    Lya_phi = heights/bin_widths/(SIDE_LENGTH_MPC**3)
    Lya_phi_err = np.sqrt(heights)/bin_widths/(SIDE_LENGTH_MPC**3)
    log_Lya_phi_err_up = np.abs(np.log10(Lya_phi_err+Lya_phi) - np.log10(Lya_phi))
    log_Lya_phi_err_low = np.abs(np.log10(np.abs(Lya_phi-Lya_phi_err)) - np.log10(Lya_phi))
    log_Lya_phi_err_low[np.isinf(log_Lya_phi_err_low)] = np.abs(np.log10(Lya_phi[np.isinf(log_Lya_phi_err_low)]))
    log_Lya_asymmetric_error = np.array(list(zip(log_Lya_phi_err_low, log_Lya_phi_err_up))).T

    np.save(f'{summary_dir}/lya_bin_centers.npy', bin_centers)
    np.save(f'{summary_dir}/Lya_phi.npy', Lya_phi)
    np.save(f'{summary_dir}/Lya_phi_err.npy', log_Lya_asymmetric_error)
    return min_lum

def gather_ewpdf(W, lya_lum, min_lum, SIDE_LENGTH_MPC, summary_dir):
    # get the EW PDF
    # our EW samples are conditioned on a minimum luminosity for detection
    # *(W>40) should be multiplied into the condition if we want to only 
    # compute the histogram for equivalent widths above 40
    heights, bins = np.histogram(W[(np.log10(lya_lum)>=min_lum)], 10)
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    bin_widths = bins[1:]-bins[:-1]
    W_density = heights/(bin_widths*SIDE_LENGTH_MPC**3)
    area = integrate.trapezoid(W_density, x=bin_centers)
    W_density = W_density/area
    W_density_err = np.sqrt(heights)/bin_widths/(SIDE_LENGTH_MPC**3)/area

    np.save(f'{summary_dir}/W_bin_centers.npy', bin_centers)
    np.save(f'{summary_dir}/W_density.npy', W_density)
    np.save(f'{summary_dir}/W_density_err.npy', W_density_err)

def get_line_params(x, y, yerr):
    # Objective function to minimize

    def objective(params):
        m, b = params
        # fit one sigma bounds and mean
        return np.sum(((m*x+b) - y)**2 / yerr**2)

    # Initial guess for slope and intercept
    initial_guess = [-1, -2]

    # Minimize the objective function
    result = minimize(objective, initial_guess, bounds=[(-4, 0), (-5, 0)])
    m, b  = result.x

    # Calculate residuals
    residuals = y - (m * x + b)

    # Sum of squared residuals
    SSR = np.sum(residuals**2)

    # Estimate variance of residuals
    N = len(x)
    sigma_squared = SSR / (N - 2)

    # Design matrix
    X = np.vstack([np.ones(len(x)), x]).T

    # Compute (X^T X)^-1
    XTX_inv = np.linalg.inv(X.T @ X)

    # Covariance matrix
    cov_matrix = sigma_squared * XTX_inv

    # Standard errors
    m_err, b_err = np.sqrt(np.diag(cov_matrix))

    return m, b, m_err, b_err

def gather_power_spectrum(x, y, z, SIDE_LENGTH_MPC, summary_dir):

    # produce galaxy density field
    box = np.zeros((SIDE_LENGTH_MPC, SIDE_LENGTH_MPC, SIDE_LENGTH_MPC))
    for _x, _y, _z in zip(x, y, z):
        box[_x, _y, _z] += 1
    
    ps, k, pvar = powerbox.get_power(box, boxlength=int(SIDE_LENGTH_MPC),\
                log_bins=True, get_variance=True, ignore_zero_mode=True,\
                vol_normalised_power=True)

    ps = ps[~np.isnan(k)]
    pvar = pvar[~np.isnan(k)]
    k = k[~np.isnan(k)]
    
    # NOTE why is this here?
    k_max = 2*np.pi / 10
    m, b, m_err, b_err = get_line_params(np.log10(k[k<k_max]), np.log10(ps[k<k_max]), \
                                        pvar[k<k_max]/(np.log10(ps[k<k_max])*np.log(10)))

    gamma_fit = m + 3
    # not positive about whether to multiply h70 here
    r0_fit = 0.7*10**(-1*b/gamma_fit + np.real(np.log10(1j)) + \
                  np.log10(np.pi/2)/2/gamma_fit - np.log10(special.gamma(gamma_fit))/gamma_fit)
    
    gamma_fit_err = m_err
    r0_fit_err = r0_fit * np.sqrt((b_err/b)**2 + (gamma_fit_err/gamma_fit)**2)

    power_spectrum_params = np.array([r0_fit, r0_fit_err, gamma_fit, gamma_fit_err])
    np.save(f'{summary_dir}/power_spectrum_params.npy', power_spectrum_params)

def gather_summaries(halo_field_fn, summary_dir, redshift, mason_muv=False, igm=True):
    """
    Given a halo field filename, produce summaries of the halo field.
    - mason_muv: if True, use the Muv-Mh relation from Mason 2018. Else, SFR-Muv.
    - igm: if True, compute the IGM transmission factor.
    """
    
    if not igm:
        summary_dir += '_noIGM/'
    else:
        summary_dir += '/'
    
    os.makedirs(summary_dir, exist_ok=True)

    halo_field = np.load(halo_field_fn)

    # positional parameters relevant to computation of tau_IGM
    x = halo_field[0]
    y = halo_field[1]
    z = halo_field[2]
    halo_masses = halo_field[3]
    # stellar_masses = halo_field[4]
    sfr = halo_field[5]

    SIDE_LENGTH_MPC = 300

    gather_hmf(halo_masses, SIDE_LENGTH_MPC, summary_dir)
    muv = get_muv(halo_masses, sfr, mason_muv)
    gather_uvlf(muv, SIDE_LENGTH_MPC, summary_dir)
    lya_lum, w, emit_bool = get_lya_luminosity(muv)
    # we need to adjust lya luminosities to account for IGM attenuation
    # this is done by multiplying by the IGM transmission factor
    min_lum = get_min_lum(redshift)
    lum_bool = np.log10(lya_lum) > min_lum
    # excise irrelevant galaxy coordinates
    x = x[emit_bool][lum_bool]
    y = y[emit_bool][lum_bool]
    z = z[emit_bool][lum_bool]
    lya_lum = lya_lum[lum_bool]

    # get IGM absorption properties
    if igm:

        xHI, density, vz, z_LoS, voigt_tables, rel_idcs_list, \
            z_LoS_highres_list, z_table_list, dz_list, prefactor, \
            n_HI = get_absorption_properties()
        dvz = np.load('./data/absorption_properties/dvz.npy')
        
        for i, j, k, l in zip(x, y, z, lya_lum):
            # check if ionized and nearest neutral region is out of the lightcone
            if (xHI[i, j, k] == 0) and (dvz[i, j, k]==vz[i, j, k]):
                continue
            else:
                tau_IGM = get_tau_igm(i, j, k, xHI, density, z_LoS, voigt_tables, rel_idcs_list, \
                    z_LoS_highres_list, z_table_list, dz_list, prefactor, n_HI)
                # shift and cut the lya line profile, then multiply by exp(-tau_IGM)
                wavelength_range = np.linspace(1215, 1220, 1000)
                wave_lya = 1216
                velocity_range = c.c.to('km/s')*(wavelength_range/wave_lya - 1)
                # NOTE before sampling dv, etc, let's just take transmission at the line center plus dvz
                # TODO sample dv_gal and add to dvz
                tau_IGM = np.interp(velocity_range, dvz[i, j, k], tau_IGM)
                transmission = np.exp(-tau_IGM)
                # apply the transmission factor to the lya luminosity
                l *= transmission
    
    # apply another selection filter which accounts for IGM transmission
    lum_bool = np.log10(lya_lum) > min_lum
    x = x[lum_bool]
    y = y[lum_bool]
    z = z[lum_bool]
    lya_lum = lya_lum[lum_bool]
    w = w[lum_bool]

    gather_laelf(lya_lum, redshift, SIDE_LENGTH_MPC, summary_dir)
    gather_ewpdf(w, lya_lum, min_lum, SIDE_LENGTH_MPC, summary_dir)
    gather_power_spectrum(x, y, z, SIDE_LENGTH_MPC, summary_dir)

    # convert to apparent AB magnitude if relevant    
    # flux_alpha = la_lum /(4*np.pi*Planck18.luminosity_distance(redshift).to(u.cm).value**2)
    # intensity_alpha = flux_alpha*1216/2.47e15
    # mab = -2.5 * np.log10(intensity_alpha) - 48.6

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')

    halo_field_dir = './data/halo_fields/'
    halo_field_fns = os.listdir(halo_field_dir)

    for halo_field_fn in halo_field_fns:
        redshift = float(halo_field_fn.split('.')[0].split('_')[-1][1:])
        summary_dir = f'./summaries/z{np.around(redshift, 2)}'

        gather_summaries(halo_field_fn, summary_dir, redshift)
        gather_summaries(halo_field_fn, summary_dir, redshift, igm=False)
