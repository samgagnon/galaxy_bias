"""
Fits straight lines with Gaussian scatter to the TANG LAE data.

These should be used to inform the priors of the model.
"""

import numpy as np

from astropy.cosmology import Planck18
from astropy import units as u
from astropy.constants import c, G, m_p, k_B

from scipy.optimize import differential_evolution

def get_beta_bouwens14(muv):
    # https://arxiv.org/pdf/1306.2950
    return -2.05 + -0.2*(muv+19.5)

# Define the model function (straight line: y = m*x + c)
def linear_func(p, x):
    m, b = p
    return m * x + b

def linear_func_b0(p, x):
    m = p[0]
    return m * x

def get_tang_data():
    # TANG LAE data
    # https://arxiv.org/pdf/2402.06070
    MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, fescA, \
        fescA_err, fescB, fescB_err, ID = np.load('../data/tang24.npy').T
    return MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, \
        fescA, fescA_err, fescB, fescB_err, ID

def linear_scatter(x, m, b_mu, b_sigma):
    """
    Generate a scatter plot of a linear function with added Gaussian noise.
    
    Parameters:
    x : array-like
        The x values for the linear function.
    m : float
        The slope of the linear function.
    b_mu : float
        The mean of the Gaussian noise to be added to the y values.
    b_sigma : float
        The standard deviation of the Gaussian noise to be added to the y values.
    
    Returns:
    y : array-like
        The y values of the linear function with added Gaussian noise.
    """
    y = m * x + np.random.normal(b_mu, b_sigma, len(x))
    return y

def kl_divergence(p, q):
    """
    Calculate the Kullback-Leibler divergence between two probability distributions.
    
    Parameters:
    p : array-like
        The first probability distribution (true distribution).
    q : array-like
        The second probability distribution (approximate distribution).
    Returns:
    float
        The Kullback-Leibler divergence between p and q.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # normalize the distributions
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # avoid division by zero and log of zero
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    
    return np.sum(p * np.log(p / q))

def dice_loss(p, q):
    """
    Calculate the DICE loss between two probability distributions.
    
    Parameters:
    p : array-like
        The first probability distribution (true distribution).
    q : array-like
        The second probability distribution (approximate distribution).
        
    Returns:
    float
        The DICE loss between p and q.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # avoid division by zero
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)

    # normalize the distributions
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    return 1 - np.sum(np.sqrt(p * q))

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

    # measured lya properties from https://arxiv.org/pdf/2402.06070
    MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, \
        fescA, fescA_err, fescB, fescB_err, ID = get_tang_data()
    
    _muv_wide, _ew_wide, _dv_wide, _fescA_wide = [], [], [], []
    _muv_deep, _ew_deep, _dv_deep, _fescA_deep = [], [], [], []

    wide_lim = 2e-17
    deep_lim = 2e-18

    n_wide = np.sum(ID==0)
    n_deep = np.sum(ID==1)
    f_wide = n_wide / (n_wide + n_deep)
    f_deep = n_deep / (n_wide + n_deep)

    for i in np.argwhere(ID==0):
        _muv_wide.append(np.random.normal(MUV[i], MUV_err[i], 100))
        _ew_wide.append(np.random.normal(ew_lya[i], ew_lya_err[i], 100))
        _dv_wide.append(np.random.normal(dv_lya[i], dv_lya_err[i], 100))
        _fescA_wide.append(np.random.normal(fescA[i], fescA_err[i], 100))

    for i in np.argwhere(ID==1):
        _muv_deep.append(np.random.normal(MUV[i], MUV_err[i], 100))
        _ew_deep.append(np.random.normal(ew_lya[i], ew_lya_err[i], 100))
        _dv_deep.append(np.random.normal(dv_lya[i], dv_lya_err[i], 100))
        _fescA_deep.append(np.random.normal(fescA[i], fescA_err[i], 100))

    _muv_wide = np.concatenate(_muv_wide)
    _ew_wide = np.concatenate(_ew_wide)
    _dv_wide = np.concatenate(_dv_wide)
    _fescA_wide = np.concatenate(_fescA_wide)

    _muv_deep = np.concatenate(_muv_deep)
    _ew_deep = np.concatenate(_ew_deep)
    _dv_deep = np.concatenate(_dv_deep)
    _fescA_deep = np.concatenate(_fescA_deep)

    _sfr_wide = 1.15e-28*10**(0.4*(51.6 - _muv_wide))
    _b_wide = -0.2*(19.5 - _muv_wide) - 2.05
    _lya_wide = _ew_wide*(2.47e15/1215.67)*10**(0.4*(51.6 - _muv_wide))*(1215.67/1500)**(_b_wide+2)
    _lha_wide = _lya_wide/11.4

    _sfr_deep = 1.15e-28*10**(0.4*(51.6 - _muv_deep))
    _b_deep = -0.2*(19.5 - _muv_deep) - 2.05
    _lya_deep = _ew_deep*(2.47e15/1215.67)*10**(0.4*(51.6 - _muv_deep))*(1215.67/1500)**(_b_deep+2)
    _lha_deep = _lya_deep/11.4

    # wide bounds 
    wdv_wide_bounds = [[0.9*np.log10(8), 1.1*np.log10(_ew_wide.max())], 
                                    [0.9*_dv_wide.min(), 1.1*_dv_wide.max()]]
    # deep bounds
    wdv_deep_bounds = [[0.9*np.log10(8), 1.1*np.log10(_ew_deep.max())], 
                                    [0.9*_dv_deep.min(), 1.1*_dv_deep.max()]]

    # all data at redshift 5
    distance_factor = 4*np.pi*Planck18.luminosity_distance(5.0).to(u.cm).value**2

    wide_lum = _ew_wide*(2.47e15/1215.67)*10**(0.4*(51.6 - _muv_wide))*(1215.67/1500)**(_b_wide+2)
    wide_flux = wide_lum / distance_factor

    deep_lum = _ew_deep*(2.47e15/1215.67)*10**(0.4*(51.6 - _muv_deep))*(1215.67/1500)**(_b_deep+2)
    deep_flux = deep_lum / distance_factor

    # first scatterplot: ew vs dv
    wdv_wide_obs, log10ew_wide_axs, dv_wide_axs = np.histogram2d(np.log10(_ew_wide[wide_flux>wide_lim]), 
                            _dv_wide[wide_flux>wide_lim], bins=50, range=wdv_wide_bounds)
    # first scatterplot: ew vs dv
    wdv_deep_obs, log10ew_deep_axs, dv_deep_axs = np.histogram2d(np.log10(_ew_deep[deep_flux>deep_lim]), 
                            _dv_deep[deep_flux>deep_lim], bins=50, range=wdv_deep_bounds)

    def optimise_wdv(theta):
        invm, b_mu, b_sigma = theta
        m = 1/invm  # Convert m to linear scale
        y_wide = linear_scatter(_dv_wide, m, b_mu, b_sigma)
        ylya_wide = 10**y_wide*(2.47e15/1215.67)*10**(0.4*(51.6 - _muv_wide))*(1215.67/1500)**(_b_wide+2)
        yflux_wide = ylya_wide / distance_factor
        y_deep = linear_scatter(_dv_deep, m, b_mu, b_sigma)
        ylya_deep = 10**y_deep*(2.47e15/1215.67)*10**(0.4*(51.6 - _muv_deep))*(1215.67/1500)**(_b_deep+2)
        yflux_deep = ylya_deep / distance_factor
        
        wdv_wide_sim, _, _ = np.histogram2d(y_wide[yflux_wide>wide_lim], _dv_wide[yflux_wide>wide_lim], bins=50, 
                             range=wdv_wide_bounds)
        wdv_deep_sim, _, _ = np.histogram2d(y_deep[yflux_deep>deep_lim], _dv_deep[yflux_deep>deep_lim], bins=50, 
                             range=wdv_deep_bounds)
    
        return f_wide*dice_loss(wdv_wide_obs, wdv_wide_sim) + \
            f_deep*dice_loss(wdv_deep_obs, wdv_deep_sim)
        
    bounds = [(-1000, -200), (1.5, 4), (1e-2, 1.0)]
    result = differential_evolution(optimise_wdv, bounds, maxiter=1000, disp=True)
    # error estimation requires a fisher information matrix, which is not implemented here

    m, b_mu, b_sigma = result.x
    m_v, b_mu_v, b_sigma_v = result.x
    log10ew_wide_sim = linear_scatter(_dv_wide, 1/m, b_mu, b_sigma)
    sim_wide_lya = 10**log10ew_wide_sim*(2.47e15/1215.67)*10**(0.4*(51.6 - _muv_wide))*(1215.67/1500)**(_b_wide+2)
    sim_wide_flux = sim_wide_lya / distance_factor

    log10ew_deep_sim = linear_scatter(_dv_deep, 1/m, b_mu, b_sigma)
    sim_deep_lya = 10**log10ew_deep_sim*(2.47e15/1215.67)*10**(0.4*(51.6 - _muv_deep))*(1215.67/1500)**(_b_deep+2)
    sim_deep_flux = sim_deep_lya / distance_factor
    
    wdv_wide_sim, _, _ = np.histogram2d(log10ew_wide_sim[sim_wide_flux>wide_lim], 
                            _dv_wide[sim_wide_flux>wide_lim], bins=50, range=wdv_wide_bounds)
    
    wdv_deep_sim, _, _ = np.histogram2d(log10ew_deep_sim[sim_deep_flux>deep_lim], 
                            _dv_deep[sim_deep_flux>deep_lim], bins=50, range=wdv_wide_bounds)
    
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))
    axs[0,0].pcolormesh(dv_wide_axs, log10ew_wide_axs, wdv_wide_obs, shading='auto', cmap='plasma')
    axs[0,0].set_ylabel(r'$\log_{10}W_{\rm emerg}$ [$\AA$]', fontsize=font_size)
    axs[0,0].text(0.05, 0.95, 'Wide', transform=axs[0,0].transAxes, 
            fontsize=font_size, verticalalignment='top', color='white')
    axs[0,0].text(0.05, 0.75, 'observed', transform=axs[0,0].transAxes, 
            fontsize=font_size, verticalalignment='top', color='white')
    im = axs[0,1].pcolormesh(dv_wide_axs, log10ew_wide_axs, wdv_wide_sim, shading='auto', cmap='plasma')
    axs[0,1].text(0.05, 0.85, 'simulated', transform=axs[0,1].transAxes, fontsize=font_size, 
            verticalalignment='top', color='white')
    axs[0,1].text(0.05, 0.10, f'log10W=(dv/{m:.2f}) + b~N({b_mu:.2f},{b_sigma:.2f})', 
            transform=axs[0,1].transAxes, fontsize=15, 
            verticalalignment='top', color='white')
    cb = plt.colorbar(im, ax=axs[0,1])
    cb.set_label('Counts', fontsize=font_size)

    axs[1,0].pcolormesh(dv_deep_axs, log10ew_deep_axs, wdv_deep_obs, shading='auto', cmap='plasma')
    axs[1,0].set_ylabel(r'$\log_{10}W_{\rm emerg}$ [$\AA$]', fontsize=font_size)
    axs[1,0].set_xlabel(r'$\Delta v$ [km s$^{-1}$]', fontsize=font_size)
    axs[1,0].text(0.05, 0.95, 'Deep', transform=axs[1,0].transAxes, 
            fontsize=font_size, verticalalignment='top', color='white')
    axs[1,0].text(0.05, 0.75, 'observed', transform=axs[1,0].transAxes, 
            fontsize=font_size, verticalalignment='top', color='white')
    im = axs[1,1].pcolormesh(dv_deep_axs, log10ew_deep_axs, wdv_deep_sim, shading='auto', cmap='plasma')
    axs[1,1].text(0.05, 0.85, 'simulated', transform=axs[1,1].transAxes, fontsize=font_size, 
            verticalalignment='top', color='white')
    axs[1,1].set_xlabel(r'$\Delta v$ [km s$^{-1}$]', fontsize=font_size)
    cb = plt.colorbar(im, ax=axs[1,1])
    cb.set_label('Counts', fontsize=font_size)

    plt.savefig('/mnt/c/Users/sgagn/Documents/phd/lyman_alpha/plots/wdv_muse_fit.pdf', dpi=300)

    # PART TWO: dv - muv relation
    dvmuv_wide_bounds = [[0.9*_dv_wide.min(), 1.1*_dv_wide.max()],
                        [1.1*_muv_wide.min(), 0.9*_muv_wide.max()]]
    dvmuv_deep_bounds = [[0.9*_dv_deep.min(), 1.1*_dv_deep.max()],
                        [1.1*_muv_deep.min(), 0.9*_muv_deep.max()]]
    
    dvmuv_wide_obs, dv_wide_axs, muv_wide_axs = np.histogram2d(_dv_wide[wide_flux>wide_lim], 
                            _muv_wide[wide_flux>wide_lim], bins=50, range=dvmuv_wide_bounds)
    
    dvmuv_deep_obs, dv_deep_axs, muv_deep_axs = np.histogram2d(_dv_deep[deep_flux>deep_lim], 
                            _muv_deep[deep_flux>deep_lim], bins=50, range=dvmuv_deep_bounds)

    def optimise_dvmuv(theta):

        m, b_mu, b_sigma = theta

        dv_sim_wide = linear_scatter(_muv_wide, m, b_mu, b_sigma)
        dv_sim_deep = linear_scatter(_muv_deep, m, b_mu, b_sigma)
        
        dvmuv_wide_sim, _, _ = np.histogram2d(dv_sim_wide[wide_flux>wide_lim], 
                            _muv_wide[wide_flux>wide_lim], bins=50, range=dvmuv_wide_bounds)
        dvmuv_deep_sim, _, _ = np.histogram2d(dv_sim_deep[deep_flux>deep_lim], 
                                _muv_deep[deep_flux>deep_lim], bins=50, range=dvmuv_deep_bounds)
    
        return f_wide*dice_loss(dvmuv_wide_obs, dvmuv_wide_sim) + \
            f_deep*dice_loss(dvmuv_deep_obs, dvmuv_deep_sim)
    
    bounds = [(-100, -50), (-1500, -1200), (50, 200)]
    result = differential_evolution(optimise_dvmuv, bounds, maxiter=1000, disp=True)
    # error estimation requires a fisher information matrix, which is not implemented here

    m, b_mu, b_sigma = result.x
    m_v, b_mu_v, b_sigma_v = result.x
    dv_sim_wide = linear_scatter(_muv_wide, m, b_mu, b_sigma)
    dv_sim_deep = linear_scatter(_muv_deep, m, b_mu, b_sigma)
    
    dvmuv_wide_sim, _, _ = np.histogram2d(dv_sim_wide[wide_flux>wide_lim], 
                            _muv_wide[wide_flux>wide_lim], bins=50, range=dvmuv_wide_bounds)
    dvmuv_deep_sim, _, _ = np.histogram2d(dv_sim_deep[deep_flux>deep_lim], 
                            _muv_deep[deep_flux>deep_lim], bins=50, range=dvmuv_deep_bounds)
    
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))
    axs[0,0].pcolormesh(muv_wide_axs, dv_wide_axs, dvmuv_wide_obs, shading='auto', cmap='plasma')
    axs[0,0].set_ylabel(r'$\Delta v$ [km s$^{-1}$]', fontsize=font_size)
    axs[0,0].text(0.05, 0.95, 'Wide', transform=axs[0,0].transAxes, 
            fontsize=font_size, verticalalignment='top', color='white')
    axs[0,0].text(0.05, 0.75, 'observed', transform=axs[0,0].transAxes, 
            fontsize=font_size, verticalalignment='top', color='white')
    im = axs[0,1].pcolormesh(muv_wide_axs, dv_wide_axs, dvmuv_wide_sim, shading='auto', cmap='plasma')
    axs[0,1].text(0.05, 0.85, 'simulated', transform=axs[0,1].transAxes, fontsize=font_size, 
            verticalalignment='top', color='white')
    axs[0,1].text(0.05, 0.10, f'Del v={m:.2f}MUV + b~N({b_mu:.2f},{b_sigma:.2f})', 
            transform=axs[0,1].transAxes, fontsize=15, 
            verticalalignment='top', color='white')
    cb = plt.colorbar(im, ax=axs[0,1])
    cb.set_label('Counts', fontsize=font_size)

    axs[1,0].pcolormesh(muv_wide_axs, dv_wide_axs, dvmuv_deep_obs, shading='auto', cmap='plasma')
    axs[1,0].set_ylabel(r'$\Delta v$ [km s$^{-1}$]', fontsize=font_size)
    axs[1,0].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
    axs[1,0].text(0.05, 0.95, 'Deep', transform=axs[1,0].transAxes, 
            fontsize=font_size, verticalalignment='top', color='white')
    axs[1,0].text(0.05, 0.75, 'observed', transform=axs[1,0].transAxes, 
            fontsize=font_size, verticalalignment='top', color='white')
    im = axs[1,1].pcolormesh(muv_wide_axs, dv_wide_axs, dvmuv_deep_sim, shading='auto', cmap='plasma')
    axs[1,1].text(0.05, 0.85, 'simulated', transform=axs[1,1].transAxes, fontsize=font_size, 
            verticalalignment='top', color='white')
    axs[1,1].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
    cb = plt.colorbar(im, ax=axs[1,1])
    cb.set_label('Counts', fontsize=font_size)

    plt.savefig('/mnt/c/Users/sgagn/Documents/phd/lyman_alpha/plots/dvmuv_muse_fit.pdf', dpi=300)

    # make muv ew histograms
    ewmuv_wide_bounds = [[0.9*np.log10(8), 1.1*np.log10(_ew_wide.max())],
                        [1.1*_muv_wide.min(), 0.9*_muv_wide.max()]]
    ewmuv_deep_bounds = [[0.9*np.log10(8), 1.1*np.log10(_ew_deep.max())],
                        [1.1*_muv_deep.min(), 0.9*_muv_deep.max()]]
    ewmuv_wide_obs, log10ew_wide_axs, muv_wide_axs = np.histogram2d(np.log10(_ew_wide[wide_flux>wide_lim]),
                            _muv_wide[wide_flux>wide_lim], bins=50, range=ewmuv_wide_bounds)
    ewmuv_deep_obs, log10ew_deep_axs, muv_deep_axs = np.histogram2d(np.log10(_ew_deep[deep_flux>deep_lim]),
                            _muv_deep[deep_flux>deep_lim], bins=50, range=ewmuv_deep_bounds)
    ewmuv_wide_sim, _, _ = np.histogram2d(log10ew_wide_sim[sim_wide_flux>wide_lim],
                            _muv_wide[sim_wide_flux>wide_lim], bins=50, range=ewmuv_wide_bounds)
    ewmuv_deep_sim, _, _ = np.histogram2d(log10ew_deep_sim[sim_deep_flux>deep_lim],
                            _muv_deep[sim_deep_flux>deep_lim], bins=50, range=ewmuv_deep_bounds)

    fig, axs = plt.subplots(2, 2, figsize=(18, 10))
    axs[0,0].pcolormesh(muv_wide_axs, log10ew_wide_axs, ewmuv_wide_obs, shading='auto', cmap='plasma')
    axs[0,0].set_ylabel(r'$\log_{10}W_{\rm emerg}$ [$\AA$]', fontsize=font_size)
    axs[0,0].text(0.05, 0.95, 'Wide', transform=axs[0,0].transAxes, 
            fontsize=font_size, verticalalignment='top', color='white')
    axs[0,0].text(0.05, 0.75, 'observed', transform=axs[0,0].transAxes, 
            fontsize=font_size, verticalalignment='top', color='white')
    im = axs[0,1].pcolormesh(muv_wide_axs, log10ew_wide_axs, ewmuv_wide_sim, shading='auto', cmap='plasma')
    axs[0,1].text(0.05, 0.85, 'simulated', transform=axs[0,1].transAxes, fontsize=font_size, 
            verticalalignment='top', color='white')
    cb = plt.colorbar(im, ax=axs[0,1])
    cb.set_label('Counts', fontsize=font_size)

    axs[1,0].pcolormesh(muv_deep_axs, log10ew_deep_axs, ewmuv_deep_obs, shading='auto', cmap='plasma')
    axs[1,0].set_ylabel(r'$\log_{10}W_{\rm emerg}$ [$\AA$]', fontsize=font_size)
    axs[1,0].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
    axs[1,0].text(0.05, 0.95, 'Deep', transform=axs[1,0].transAxes, 
            fontsize=font_size, verticalalignment='top', color='white')
    axs[1,0].text(0.05, 0.75, 'observed', transform=axs[1,0].transAxes, 
            fontsize=font_size, verticalalignment='top', color='white')
    im = axs[1,1].pcolormesh(muv_deep_axs, log10ew_deep_axs, ewmuv_deep_sim, shading='auto', cmap='plasma')
    axs[1,1].text(0.05, 0.85, 'simulated', transform=axs[1,1].transAxes, fontsize=font_size, 
            verticalalignment='top', color='white')
    axs[1,1].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
    cb = plt.colorbar(im, ax=axs[1,1])
    cb.set_label('Counts', fontsize=font_size)

    plt.savefig('/mnt/c/Users/sgagn/Documents/phd/lyman_alpha/plots/wmuv_muse_fit.pdf', dpi=300)

    # fit lha vs sfr
    lha_wide_bounds = [[0.9*_lha_wide.min(), 1.1*_lha_wide.max()],
                        [0.9*_sfr_wide.min(), 1.1*_sfr_wide.max()]]
    
    lha_deep_bounds = [[0.9*_lha_deep.min(), 1.1*_lha_deep.max()],
                        [0.9*_sfr_deep.min(), 1.1*_sfr_deep.max()]]
    lha_wide_obs, lha_wide_axs, sfr_wide_axs = np.histogram2d(_lha_wide[wide_flux>wide_lim], 
                            _sfr_wide[wide_flux>wide_lim], bins=50, range=lha_wide_bounds)
    lha_deep_obs, lha_deep_axs, sfr_deep_axs = np.histogram2d(_lha_deep[deep_flux>deep_lim],
                            _sfr_deep[deep_flux>deep_lim], bins=50, range=lha_deep_bounds)
    
    def optimise_lha(theta):
        b_mu, b_sigma = theta

        lha_sim_wide = 10**np.random.normal(b_mu, b_sigma, len(_sfr_wide)) * _sfr_wide
        lha_sim_deep = 10**np.random.normal(b_mu, b_sigma, len(_sfr_deep)) * _sfr_deep

        lha_wide_sim, _, _ = np.histogram2d(lha_sim_wide[wide_flux>wide_lim], 
                            _sfr_wide[wide_flux>wide_lim], bins=50, range=lha_wide_bounds)
        lha_deep_sim, _, _ = np.histogram2d(lha_sim_deep[deep_flux>deep_lim], 
                            _sfr_deep[deep_flux>deep_lim], bins=50, range=lha_deep_bounds)

        return f_wide*dice_loss(lha_wide_obs, lha_wide_sim) + \
            f_deep*dice_loss(lha_deep_obs, lha_deep_sim)
    
    bounds = [(40.5, 42), (0.1, 0.5)]
    result = differential_evolution(optimise_lha, bounds, maxiter=1000, disp=True)
    # error estimation requires a fisher information matrix, which is not implemented here
    b_mu, b_sigma = result.x

    lha_sim_wide = 10**np.random.normal(b_mu, b_sigma, len(_sfr_wide)) * _sfr_wide
    lha_sim_deep = 10**np.random.normal(b_mu, b_sigma, len(_sfr_deep)) * _sfr_deep

    lha_wide_sim, _, _ = np.histogram2d(lha_sim_wide[wide_flux>wide_lim], 
                        _sfr_wide[wide_flux>wide_lim], bins=50, range=lha_wide_bounds)
    lha_deep_sim, _, _ = np.histogram2d(lha_sim_deep[deep_flux>deep_lim], 
                        _sfr_deep[deep_flux>deep_lim], bins=50, range=lha_deep_bounds)
    
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))
    axs[0,0].pcolormesh(sfr_wide_axs, lha_wide_axs, lha_wide_obs, shading='auto', cmap='plasma')
    axs[0,0].set_ylabel(r'${\rm L}_{\rm H\alpha}$ [erg s$^{-1}$]', fontsize=font_size)
    axs[0,0].text(0.05, 0.95, 'Wide', transform=axs[0,0].transAxes, 
            fontsize=font_size, verticalalignment='top', color='white')
    axs[0,0].text(0.05, 0.75, 'observed', transform=axs[0,0].transAxes,
            fontsize=font_size, verticalalignment='top', color='white')
    im = axs[0,1].pcolormesh(sfr_wide_axs, lha_wide_axs, lha_wide_sim, shading='auto', cmap='plasma')
    axs[0,1].text(0.05, 0.85, 'simulated', transform=axs[0,1].transAxes, fontsize=font_size, 
            verticalalignment='top', color='white')
    axs[0,1].text(0.05, 0.10, f'LHA=10^b~N({b_mu:.2f},{b_sigma:.2f}) * SFR', 
                  transform=axs[0,1].transAxes, fontsize=15,
            verticalalignment='top', color='white')
    cb = plt.colorbar(im, ax=axs[0,1])
    cb.set_label('Counts', fontsize=font_size)
    axs[1,0].pcolormesh(sfr_deep_axs, lha_deep_axs, lha_deep_obs, shading='auto', cmap='plasma')
    axs[1,0].set_ylabel(r'${\rm L}_{\rm H\alpha}$ [erg s$^{-1}$]', fontsize=font_size)
    axs[1,0].set_xlabel(r'${\rm SFR}$ [M$_\odot$ yr$^{-1}$]', fontsize=font_size)
    axs[1,0].text(0.05, 0.95, 'Deep', transform=axs[1,0].transAxes, 
            fontsize=font_size, verticalalignment='top', color='white')
    axs[1,0].text(0.05, 0.75, 'observed', transform=axs[1,0].transAxes,
            fontsize=font_size, verticalalignment='top', color='white')
    im = axs[1,1].pcolormesh(sfr_deep_axs, lha_deep_axs, lha_deep_sim, shading='auto', cmap='plasma')
    axs[1,1].text(0.05, 0.85, 'simulated', transform=axs[1,1].transAxes, fontsize=font_size, 
            verticalalignment='top', color='white')
    axs[1,1].set_xlabel(r'${\rm SFR}$ [M$_\odot$ yr$^{-1}$]', fontsize=font_size)
    cb = plt.colorbar(im, ax=axs[1,1])
    cb.set_label('Counts', fontsize=font_size)
    plt.savefig('/mnt/c/Users/sgagn/Documents/phd/lyman_alpha/plots/lha_sfr_muse_fit.pdf', dpi=300)
