import os
import py21cmfast as p21c
import numpy as np

from scipy import integrate
from astropy.cosmology import Planck18, z_at_value
from astropy import units as U, constants as c

from rng2sfr import *
from data import *

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
    fit functions obtained by Lu et al. 2018.
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

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')

    import argparse

    parser = argparse.ArgumentParser(description='Plot the LAELF')
    parser.add_argument('-z', type=float, default=6.6, help='Redshift')
    parser.add_argument('-m', type=float, default=1e8, help='Minimum halo mass')
    args = parser.parse_args()

    z = args.z
    m = args.m

    fn = f'./data/halo_fields/z{np.around(z,1)}/sgh24_MIN_MASS_{np.around(np.log10(m), 1)}.h5'


    summary_dir = f'./summaries/z{np.around(z, 1)}/'
    os.makedirs(summary_dir, exist_ok=True)

    pth = p21c.HaloField.from_file(fn)
    SIDE_LENGTH_MPC = pth.user_params.BOX_LEN
    # extract fields
    # could this excision of zero halo masses be the source of the error?
    halo_masses = pth.halo_masses[pth.halo_masses>0]

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

    stellar_rng = pth.star_rng[pth.halo_masses>0]
    sfr_rng = pth.sfr_rng[pth.halo_masses>0]
    # compute galaxy properties from rng fields
    stellar_masses = get_stellar_mass(halo_masses, stellar_rng)
    sfr = get_sfr(stellar_masses, sfr_rng, pth.redshift)

    modes = ['sfr', 'mh']

    uv_list = []
    uv_bins = []
    uv_err_list = []
    lya_list = []
    lya_bins = []
    err_list = []
    W_list = []
    W_bins = []
    W_err_list = []

    for mode in modes:

        if mode == 'sfr':
            # convert SFR to absolute UV magnitude
            Muv = sfr2Muv(sfr)
        elif mode == 'mh':
            Muv = (-1/0.3)*(np.log10(halo_masses) \
                    - 11.75) - 20.0 - 0.26*pth.redshift

        bin_edges = []
        bin_centers = np.asarray(b21_mag[1])
        bin_edge = np.zeros(len(bin_centers)+1)
        bin_widths = (bin_centers[1:] - bin_centers[:-1])*0.5
        bin_edge[0] = bin_centers[0]-bin_widths[0]
        bin_edge[-1] = bin_centers[-1] + bin_widths[-1]
        bin_edge[1:-1] = bin_widths + bin_centers[:-1]
        bin_edges += [bin_edge]

        heights, bins = np.histogram(Muv, bins=bin_edges[0])
        # heights, bins = np.histogram(Llya)
        bin_centers = 0.5*(bins[1:]+bins[:-1])
        bin_widths = bins[1:]-bins[:-1]
        uv_phi = heights/bin_widths/(SIDE_LENGTH_MPC**3)
        uv_phi_err = np.sqrt(heights)/bin_widths/(SIDE_LENGTH_MPC**3)
        log_uv_phi_err_up = np.abs(np.log10(uv_phi_err+uv_phi) - np.log10(uv_phi))
        log_uv_phi_err_low = np.abs(np.log10(np.abs(uv_phi-uv_phi_err)) - np.log10(uv_phi))
        log_uv_phi_err_low[np.isinf(log_uv_phi_err_low)] = np.abs(np.log10(uv_phi[np.isinf(log_uv_phi_err_low)]))
        log_uv_asymmetric_error = np.array(list(zip(log_uv_phi_err_low, log_uv_phi_err_up))).T

        np.save(f'{summary_dir}/{mode}_uv_phi.npy', uv_phi)
        np.save(f'{summary_dir}/{mode}_uv_phi_err.npy', log_uv_asymmetric_error)
        np.save(f'{summary_dir}/{mode}_uv_bin_centers.npy', bin_centers)

        uv_list += [uv_phi]
        uv_err_list += [log_uv_asymmetric_error]
        uv_bins += [bin_centers]

        # obtain equivalent widths via Mason 2018 fit
        W, emit_bool = mason2018(Muv)    

        # convert equivalent width to lyman alpha luminosity
        beta = get_beta(Muv[emit_bool])
        C_const = 2.47 * 1e15 * U.Hz / 1216 / U.Angstrom * (1500 / 1216) \
            ** (-(beta) - 2)
        L_UV_mean = 10 ** (-0.4 * (Muv[emit_bool] - 51.6))
        la_lum = W * C_const.value * L_UV_mean

        lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver = get_silverrush_laelf(z)

        bin_edges = []
        bin_centers = np.asarray(lum_silver)
        bin_edge = np.zeros(len(bin_centers)+1)
        bin_widths = (bin_centers[1:] - bin_centers[:-1])*0.5
        bin_edge[0] = bin_centers[0]-bin_widths[0]
        min_lum = bin_edge[0]
        bin_edge[-1] = bin_centers[-1] + bin_widths[-1]
        bin_edge[1:-1] = bin_widths + bin_centers[:-1]
        bin_edges += [bin_edge]

        heights, bins = np.histogram(np.log10(la_lum), bins=bin_edges[0])
        # heights, bins = np.histogram(Llya)
        bin_centers = 0.5*(bins[1:]+bins[:-1])
        bin_widths = bins[1:]-bins[:-1]
        Lya_phi = heights/bin_widths/(SIDE_LENGTH_MPC**3)
        Lya_phi_err = np.sqrt(heights)/bin_widths/(SIDE_LENGTH_MPC**3)
        log_Lya_phi_err_up = np.abs(np.log10(Lya_phi_err+Lya_phi) - np.log10(Lya_phi))
        log_Lya_phi_err_low = np.abs(np.log10(np.abs(Lya_phi-Lya_phi_err)) - np.log10(Lya_phi))
        log_Lya_phi_err_low[np.isinf(log_Lya_phi_err_low)] = np.abs(np.log10(Lya_phi[np.isinf(log_Lya_phi_err_low)]))
        log_Lya_asymmetric_error = np.array(list(zip(log_Lya_phi_err_low, log_Lya_phi_err_up))).T

        np.save(f'{summary_dir}/{mode}_lya_bin_centers.npy', bin_centers)
        lya_bins += [bin_centers]

        # get the EW PDF
        # wait lmao
        heights, bins = np.histogram(W[(np.log10(la_lum)>=min_lum)*(W<40)], 5)
        bin_centers = 0.5*(bins[1:]+bins[:-1])
        bin_widths = bins[1:]-bins[:-1]
        W_density = heights/(bin_widths*SIDE_LENGTH_MPC**3)
        area = integrate.trapezoid(W_density, x=bin_centers)
        W_density = W_density/area
        W_density_err = np.sqrt(heights)/bin_widths/(SIDE_LENGTH_MPC**3)/area

        np.save(f'{summary_dir}/{mode}_W_bin_centers.npy', bin_centers)
        W_bins += [bin_centers]

        lya_list += [Lya_phi]
        err_list += [log_Lya_asymmetric_error]
        W_list += [W_density]
        W_err_list += [W_density_err]

        np.save(f'{summary_dir}/{mode}_Lya_phi.npy', Lya_phi)
        np.save(f'{summary_dir}/{mode}_Lya_phi_err.npy', log_Lya_asymmetric_error)
        np.save(f'{summary_dir}/{mode}_W_density.npy', W_density)
        np.save(f'{summary_dir}/{mode}_W_density_err.npy', W_density_err)