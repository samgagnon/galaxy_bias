import numpy as np

from astropy.cosmology import Planck18
from astropy import units as u
from astropy.constants import c, G, m_p, k_B

redshift = 5.0
nu_lya = (c/(1215.67*u.Angstrom)).to('Hz').value
_, _, _, halo_masses, _, sfr = np.load(f'../data/halo_field_{redshift}.npy')
muv = 51.64 - np.log10(sfr*3.1557e7/1.15e-28) / 0.4
halo_masses = halo_masses[muv<=-16]
muv = muv[muv<=-16]
NSAMPLES = len(muv)

beta_factor = (0.959**(muv + 19.5))

lum_flux_factor = 4*np.pi*(Planck18.luminosity_distance(redshift).to('cm').value)**2

muv_space = np.linspace(-24, -16, 100)
dv_space = np.linspace(0, 1000, 100)
dv_logspace = 10**np.linspace(0, 3, 100)
w_space = 10**np.linspace(0, 3, 100)
fesc_space = np.linspace(0, 1, 100)

# compute theoretical maximum LyA emission
lum_dens_uv = 10**(-0.4*(muv - 51.6))
nu_uv = (c/(1500*u.Angstrom)).to('Hz').value
kappa_uv = 1.15e-28
sfr = lum_dens_uv*kappa_uv
l_lya = 1215.67
# assumes case B recombination
# possibly inaccurate c.f. https://arxiv.org/pdf/2503.21896
fesc_prefactor = nu_uv/(11.5*kappa_uv*l_lya)
w_intr_prefactor = 11.5*kappa_uv*l_lya/nu_uv

limit_b = -18.2 - Planck18.distmod(5.0).value + Planck18.distmod(3.8).value
limit_v = -18.8 - Planck18.distmod(5.0).value + Planck18.distmod(5.0).value
limit_i = -19.1 - Planck18.distmod(5.0).value + Planck18.distmod(5.9).value
limit_mean = np.mean([limit_b, limit_v, limit_i])

# a_w = -0.0034
# a_v = -80.5

# theta_min = np.array([-0.0034/1.1, -80.5/1.1, 1e40, 2, -1500.0])
# theta_max = np.array([-0.0034*1.1, -80.5*1.1, 1e42, 3, -1300.0])
FACTOR = 1.2
theta_min = np.array([-0.0034/1.2, -80.5/FACTOR, 1e41, 2.698/FACTOR, -1336.9])
theta_max = np.array([-0.0034*1.2, -80.5*FACTOR, 3.26e41*1.1, 2.698, -1336.9/FACTOR])
# theta_min = np.array([-0.0034/FACTOR, -80.5/FACTOR, 3.26e41/FACTOR, 2.698/FACTOR, -1336.9/FACTOR])
# theta_max = np.array([-0.0034*FACTOR, -80.5*FACTOR, 3.26e41*FACTOR, 2.698*FACTOR, -1336.9*FACTOR])

# theta_mle = np.array([3.26e41, 2.698, -1336.9])

def get_tang_data():
    # TANG LAE data
    # https://arxiv.org/pdf/2402.06070
    MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, fescA, \
        fescA_err, fescB, fescB_err, ID = np.load('../data/tang24.npy').T
    return MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, \
        fescA, fescA_err, fescB, fescB_err, ID

def get_sigma(theta):

    a_w, a_v, a_ha, b_w, b_v = theta

    # compute residuals
    res_ha = np.abs(lum_ha_data/(sfr_data*a_ha))
    sigma_ha = res_ha.std()*a_ha

    res_w = np.abs(np.log10(ew_lya) - a_w*dv_lya + b_w)
    sigma_w = res_w.mean()

    res_v = np.abs(dv_lya - (a_v*MUV + b_v))
    sigma_v = res_v.mean()

    return np.array([sigma_ha, sigma_w, sigma_v])

def get_theta(theta):
    """
    Inputs sampled from a uniform distribution
    """

    _theta = theta_min + (theta_max - theta_min)*theta

    # change range of b_w depending on slope of a_w
    b_v_max = 300 + 19 * _theta[1]
    theta_max[4] = b_v_max
    b_w_max = np.log10(2) + 2 + 20*_theta[0]*_theta[1] - _theta[0]*theta_min[4]
    theta_max[3] = b_w_max

    # print(theta_min[4], theta_max[4])
    # print(theta_min[3], theta_max[3])
    # quit()

    theta = theta_min + (theta_max - theta_min)*theta

    sigma = get_sigma(theta)

    return np.concatenate([theta, sigma])

def get_beta_bouwens14(muv):
    # https://arxiv.org/pdf/1306.2950
    return -2.05 + -0.2*(muv+19.5)

def get_lya_properties(theta):

    # theta = np.concatenate([theta_mle, get_sigma(theta_mle)])

    a_w, a_v, a_ha_mean, b_w_mean, b_v_mean,\
        sigma_ha, sigma_w, sigma_v = theta
    
    # sample random variables
    a_ha = np.random.normal(a_ha_mean, sigma_ha, NSAMPLES)
    b_w = np.random.normal(b_w_mean, sigma_w, NSAMPLES)
    b_v = np.random.normal(b_v_mean, sigma_v, NSAMPLES)

    dv = np.abs(a_v*muv + b_v)
    fesc = fesc_prefactor*((1.04*10**(a_w*a_v))**(muv + 19.5))*\
            (10**(a_w*(b_v - 19.5*a_v) + b_w))/a_ha
    fesc[fesc>1] = 1
    w_intr = w_intr_prefactor * a_ha * beta_factor
    w = w_intr*fesc

    lum_dens_alpha = 0.989 * (w / l_lya) * lum_dens_uv * beta_factor
    intensity = lum_dens_alpha/lum_flux_factor
    lum_alpha = lum_dens_alpha * nu_lya
    mab = -2.5 * np.log10(intensity) - 48.6

    fha = a_ha*sfr/lum_flux_factor

    return halo_masses, muv, mab, fha, dv, w, fesc, lum_alpha

def model(theta):
    # produce complete sample from model
    # start = time.time()

    halo_masses, muv, mab, fha, dv, w, fesc, lum_alpha = get_lya_properties(theta)
    # print('Time to generate model:', time.time() - start)

    # start = time.time()

    # number density comparison with Umeda+24 ~2.58 10^-3 cMpc^-3 La>10^42.7 erg/s
    n_LAE = len(lum_alpha[lum_alpha>5e42])/(3e2**3)
    if n_LAE == 0:
        return np.zeros((15, 10000)), 1e6
    
    print(n_LAE)
    n_likelihood = np.abs(np.log10(n_LAE) - np.log10(2.58e-3))
        
    # appy selection effects
    _mab = np.copy(mab)
    _halo_masses = np.copy(halo_masses)
    _dv = np.copy(dv)
    _w = np.copy(w)
    _muv = np.copy(muv)
    _fesc = np.copy(fesc)
    _fha = np.copy(fha)

    # ROW ONE: MUSE-Wide
    flux_limit_draw = np.random.normal(loc=2e-17, scale=2e-17/5, size=len(_muv))
    flux_limit_draw[flux_limit_draw > 2e-17] = 2e-17
    flux_limit_draw[flux_limit_draw <= 0.0] = 1e-100
    mab_lim = -2.5 * np.log10(flux_limit_draw/nu_lya) - 48.6
    muv_limit_draw = -15
    fha_draw = np.random.normal(loc=2e-18, scale=2e-18/5, size=len(_muv))
    fha_draw[fha_draw > 2e-18] = 2e-18
    w_draw = np.random.normal(loc=80, scale=80/5, size=len(_muv))
    w_draw[w_draw > 80] = 80
    SELECT = (_mab<mab_lim)*(_muv<muv_limit_draw)*(_fha>fha_draw)*(_w>w_draw)
    halo_masses = _halo_masses[SELECT]
    dv = _dv[SELECT]
    fesc = _fesc[SELECT]
    w = _w[SELECT]
    muv = _muv[SELECT]

    if len(w) == 0:
        return np.zeros((15, 10000)), 3
    
    # print('Time to apply selection effects:', time.time() - start)

    # start = time.time()

    h_mdv, _, _ = np.histogram2d(muv, dv, bins=(muv_space, dv_space), density=True)
    model00 = h_mdv.T
    h_mw, _, _ = np.histogram2d(muv, w, bins=(muv_space, w_space), density=True)
    model01 = h_mw.T
    h_wdv, _, _ = np.histogram2d(w, dv, bins=(w_space, dv_space), density=True)
    model02 = h_wdv.T
    h_dvfesc, _, _ = np.histogram2d(dv, fesc, bins=(dv_logspace, fesc_space), density=True)
    model03 = h_dvfesc.T
    h_wfesc, _, _ = np.histogram2d(w, fesc, bins=(w_space, fesc_space), density=True)
    model04 = h_wfesc.T

    # print('Time to create histograms:', time.time() - start)

    # SECOND ROW: MUSE-Deep
    flux_limit_draw = np.random.normal(loc=2e-18, scale=2e-18/5, size=NSAMPLES)
    flux_limit_draw[flux_limit_draw > 2e-18] = 2e-18
    flux_limit_draw[flux_limit_draw <= 0.0] = 1e-100
    mab_lim = -2.5 * np.log10(flux_limit_draw/nu_lya) - 48.6
    w_draw = np.random.normal(loc=8, scale=8/5, size=len(_muv))
    w_draw[w_draw > 8] = 8
    SELECT = (_mab<mab_lim)*(_muv<-15)*(_fha>fha_draw)*(_w>w_draw)
    halo_masses = _halo_masses[SELECT]
    dv = _dv[SELECT]
    fesc = _fesc[SELECT]
    w = _w[SELECT]
    muv = _muv[SELECT]

    if len(w) == 0:
        return np.zeros((15, 10000)), 3

    # create 2d histogram of muv, dv
    h_mdv, _, _ = np.histogram2d(muv, dv, bins=(muv_space, dv_space), density=True)
    model10 = h_mdv.T
    h_mw, _, _ = np.histogram2d(muv, w, bins=(muv_space, w_space), density=True)
    model11 = h_mw.T
    h_wdv, _, _ = np.histogram2d(w, dv, bins=(w_space, dv_space), density=True)
    model12 = h_wdv.T
    h_dvfesc, _, _ = np.histogram2d(dv, fesc, bins=(dv_logspace, fesc_space), density=True)
    model13 = h_dvfesc.T
    h_wfesc, _, _ = np.histogram2d(w, fesc, bins=(w_space, fesc_space), density=True)
    model14 = h_wfesc.T

    # THIRD ROW: DEIMOS

    # apply flux completeness limit for each sample
    flux_limit_draw = np.random.normal(loc=2e-18, scale=2e-18/5, size=NSAMPLES)
    flux_limit_draw[flux_limit_draw > 2e-18] = 2e-18
    flux_limit_draw[flux_limit_draw <= 0.0] = 1e-100
    mab_lim = -2.5 * np.log10(flux_limit_draw/nu_lya) - 48.6
    fuv_limit_draw = np.random.normal(loc=10**(-0.4*limit_mean), \
                    scale=10**(-0.4*limit_mean)/5, size=NSAMPLES)
    muv_limit_draw = np.zeros(NSAMPLES)
    muv_limit_draw[fuv_limit_draw>0] = -2.5*np.log10(fuv_limit_draw[fuv_limit_draw>0])
    SELECT = (_mab<mab_lim)*(_muv<-15)*(_fha>fha_draw)*(_w>w_draw)
    halo_masses = _halo_masses[SELECT]
    dv = _dv[SELECT]
    fesc = _fesc[SELECT]
    w = _w[SELECT]
    muv = _muv[SELECT]

    if len(w) == 0:
        return np.zeros((15, 10000)), 3

    # create 2d histogram of muv, dv
    h_mdv, _, _ = np.histogram2d(muv, dv, bins=(muv_space, dv_space), density=True)
    model20 = h_mdv.T
    h_mw, _, _ = np.histogram2d(muv, w, bins=(muv_space, w_space), density=True)
    model21 = h_mw.T
    h_wdv, _, _ = np.histogram2d(w, dv, bins=(w_space, dv_space), density=True)
    model22 = h_wdv.T
    h_dvfesc, _, _ = np.histogram2d(dv, fesc, bins=(dv_logspace, fesc_space), density=True)
    model23 = h_dvfesc.T
    h_wfesc, _, _ = np.histogram2d(w, fesc, bins=(w_space, fesc_space), density=True)
    model24 = h_wfesc.T

    # array of all flattened histograms
    out = np.array([model00.flatten(), model10.flatten(), model20.flatten(), \
                    model01.flatten(), model11.flatten(), model21.flatten(), model02.flatten(), \
                    model12.flatten(), model22.flatten(), model03.flatten(), model13.flatten(), \
                    model23.flatten(), model04.flatten(), model14.flatten(), model24.flatten()])

    return out, n_likelihood

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc) 
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')

    hist00 = np.load('../data/muse_hist/hist00.npy').flatten()
    hist10 = np.load('../data/muse_hist/hist10.npy').flatten()
    hist20 = np.load('../data/muse_hist/hist20.npy').flatten()
    hist01 = np.load('../data/muse_hist/hist01.npy').flatten()
    hist11 = np.load('../data/muse_hist/hist11.npy').flatten()
    hist21 = np.load('../data/muse_hist/hist21.npy').flatten()
    hist02 = np.load('../data/muse_hist/hist02.npy').flatten()
    hist12 = np.load('../data/muse_hist/hist12.npy').flatten()
    hist22 = np.load('../data/muse_hist/hist22.npy').flatten()
    hist03 = np.load('../data/muse_hist/hist03.npy').flatten()
    hist13 = np.load('../data/muse_hist/hist13.npy').flatten()
    hist23 = np.load('../data/muse_hist/hist23.npy').flatten()
    hist04 = np.load('../data/muse_hist/hist04.npy').flatten()
    hist14 = np.load('../data/muse_hist/hist14.npy').flatten()
    hist24 = np.load('../data/muse_hist/hist24.npy').flatten()

    data_hist_list = [hist00, hist10, hist20, hist01, hist11, hist21, hist02, \
                    hist12, hist22, hist03, hist13, hist23, hist04, hist14, hist24]
    
    MUV, _, _, ew_lya, _, dv_lya, _, \
        _, _, fescB, _, _ = get_tang_data()
    
    likelihood_weights = np.array([24/79, 36/79, 19/79, 24/79, 36/79, 19/79, 24/79, 36/79, 19/79, \
                                   24/79, 36/79, 19/79, 24/79, 36/79, 19/79])
    
    lum_dens_uv_data = 10**(-0.4*(MUV - 51.6))
    sfr_data = lum_dens_uv_data*1.15e-28
    ew_lya_int_data = ew_lya/fescB
    lum_lya_max_caseB_data = ew_lya_int_data * lum_dens_uv_data * (1215.67/1500)**(-2.3 + 2) / (l_lya/nu_uv)
    lum_ha_data = lum_lya_max_caseB_data / 11.4

    likelihood_list = []
    theta_list = []

    for i in range(10):
        
        theta = np.random.uniform(0, 1, 5)
        theta = get_theta(theta)
        model_hist_list, n_likelihood = model(theta)

        if np.sum(model_hist_list) > 0:
           
            likelihood = 1.0

            for dhist, mhist, lw in zip(data_hist_list, model_hist_list, likelihood_weights):

                dhist /= dhist.sum()
                mhist /= mhist.sum()

                SELECT = (dhist > 0) * (mhist > 0)
                dhist = dhist[SELECT]
                mhist = mhist[SELECT]

                dkl_integrand = lw*dhist*(np.log(dhist) - np.log(mhist))
                dkl = dkl_integrand.sum()
                if dkl <= 0:
                    dkl = 1e-3
                likelihood += np.log(dkl)
                
        else:
            likelihood = np.log(n_likelihood)

        likelihood /= 15

        likelihood += np.log(n_likelihood)
        
        print(theta, np.around(n_likelihood, 2), np.around(likelihood, 2))
        likelihood_list.append(likelihood)
        theta_list.append(theta)

    likelihood_list = np.array(likelihood_list)
    theta_list = np.array(theta_list)

    # theta_list[np.argmin(likelihood_list)] = np.concatenate([theta_mle, get_sigma(theta_mle)])
    print('Best fit:', theta_list[np.argmin(likelihood_list)])
    np.save('../data/theta.npy', theta_list[np.argmin(likelihood_list)])
