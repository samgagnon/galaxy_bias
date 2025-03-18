import numpy as np
import seaborn as sns
import pandas as pd
import kdetools as kde

from astropy.cosmology import Planck18
from astropy import units as u
from astropy.constants import c, G

from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA

def get_beta_bouwens14(muv):
    # https://arxiv.org/pdf/1306.2950
    return -2.05 + -0.2*(muv+19.5)

def get_w_for_flux(muv_range, flux, redshift):
    beta = get_beta_bouwens14(muv_range)
    lum_dens_uv_bkgd = 10**(-0.4*(muv_range - 51.6))
    l_lya_bkgd = 1215.67
    nu_lya_bkgd = (c/(l_lya_bkgd*u.Angstrom)).to('Hz').value
    intensity_bkgd = flux / nu_lya_bkgd
    lum_dens_alpha_bkgd = intensity_bkgd * 4*np.pi*(Planck18.luminosity_distance(redshift).to('cm').value)**2
    w_range = lum_dens_alpha_bkgd * l_lya_bkgd / lum_dens_uv_bkgd / (1215.67/1500)**(beta + 2)
    return w_range

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc) 
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    # plt.style.use('dark_background')

    # dataset from https://arxiv.org/pdf/2202.06642
    idx, z, ze, w, we, muv, muve, llya, llyae, peaksep, peakse, fwhm, fwhme, assym, assyme = np.load('../data/muse.npy').T
    # w[w<200] = 0
    # w[w>=200] = 1

    # muv_range = np.linspace(-23, -14, 1000)
    # # wlim3 = get_w_for_flux(muv_range, 2e-18, 3.0)
    # wlim6 = get_w_for_flux(muv_range, 2e-18, 5.0)

    # # plt.errorbar(muv[z<4], w[z<4], xerr=muve[z<4], yerr=we[z<4], fmt='.', color='blue', alpha=0.7)
    # plt.errorbar(muv[z>5], w[z>5], xerr=muve[z>5], yerr=we[z>5], fmt='.', color='red', alpha=0.7)
    # # plt.plot(muv_range, wlim3, color='blue', label=r'$z=3$')
    # plt.plot(muv_range, wlim6, color='black', label=r'$z=5$')
    # plt.legend()
    # plt.yscale('log')
    # plt.xlabel(r'$M_{\rm UV}$')
    # plt.ylabel(r'$W_{\rm Ly\alpha}$')
    # plt.xlim(-23, -14)
    # plt.show()

    # quit()
    muse_dict = {'z': z/z.mean() - 1, 'muv': muv/muv.mean() - 1, 'w': np.log10(w)/np.log10(w.mean()) - 1, \
                 'peaksep': peaksep/peaksep.mean() - 1, 'fwhm': fwhm/fwhm.mean() - 1, 'assym': assym/assym.mean() - 1}
    
    # clearly, we have two populations: those with ~0 peak separation and those with high peak separation
    # we can use PCA to separate these populations
    pca = PCA(n_components=2)
    subset = np.array([muse_dict['z'], muse_dict['muv'], muse_dict['w'], muse_dict['peaksep'], muse_dict['fwhm'], muse_dict['assym']]).T
    pca.fit(subset)
    print(pca.explained_variance_ratio_)
    # print(pca.components_)
    # print(pca.singular_values_)
    # print(pca.mean_)
    # print(pca.noise_variance_)
    # print(pca.get_covariance())
    # print(pca.get_precision())
    # print(pca.get_params())
    # quit()

    subdict = {'z': muse_dict['z'][fwhm>0.0], 'muv': muse_dict['muv'][fwhm>0.0], 'w': muse_dict['w'][fwhm>0.0], \
               'peaksep': muse_dict['peaksep'][fwhm>0.0], 'fwhm': muse_dict['fwhm'][fwhm>0.0], \
               'assym': muse_dict['assym'][fwhm>0.0]}
    # df = pd.DataFrame(subdict)
    df = pd.DataFrame(muse_dict)
    sns.pairplot(df, kind='hist', diag_kind='kde')
    plt.show()
    quit()
    kernel = kde.gaussian_kde(subset)
    kernel.set_bandwidth(bw_method='cv', bw_type='diagonal')
    samples = kernel.conditional_resample(10000, x_cond=[-17/muv.mean() - 1], dims_cond=[0])
    print(samples)
    sns.pairplot(pd.DataFrame(samples), kind='kde', diag_kind='kde')
    plt.show()