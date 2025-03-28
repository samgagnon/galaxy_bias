import numpy as np

# from scipy.special import erf
# from scipy.integrate import trapezoid
# from scipy.stats import linregress
from scipy import odr

from astropy.cosmology import Planck18
from astropy import units as u
from astropy.constants import c, G, m_p, k_B

# Define the model function (straight line: y = m*x + c)
def linear_func(p, x):
    m = p[0]
    return m * x

def get_beta_bouwens14(muv):
    # https://arxiv.org/pdf/1306.2950
    return -2.05 + -0.2*(muv+19.5)

def get_tang_data():
    # TANG LAE data
    # https://arxiv.org/pdf/2402.06070
    MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, fescA, \
        fescA_err, fescB, fescB_err, ID = np.load('../data/tang24.npy').T
    return MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, \
        fescA, fescA_err, fescB, fescB_err, ID

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc) 
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')

    muv_space = np.linspace(-23, -16, 1000)
    lum_dens_uv = 10**(-0.4*(muv_space - 51.6))
    nu_uv = (c/(1500*u.Angstrom)).to('Hz').value
    sfr = lum_dens_uv*1.15e-28#/3.1557e7
    lum_ha = 1.27e41*sfr
    lum_lya_max_caseA = 8.7*lum_ha  # https://arxiv.org/abs/1505.07483, https://arxiv.org/abs/1505.05149
    lum_lya_max_caseB = 11.4*lum_ha # https://ui.adsabs.harvard.edu/abs/2006agna.book.....O/abstract
    # convert from luminosity to equivalent width
    beta = get_beta_bouwens14(muv_space)
    l_lya = 1215.67 #* (1 + redshift)
    w_max_caseA = (l_lya/nu_uv) * (lum_lya_max_caseA / lum_dens_uv) / (1215.67/1500)**(beta + 2)
    w_max_caseB = (l_lya/nu_uv) * (lum_lya_max_caseB / lum_dens_uv) / (1215.67/1500)**(beta + 2)

    plt.plot(muv_space, w_max_caseA, label='Case A', color='cyan')
    plt.plot(muv_space, w_max_caseB, label='Case B', color='lime')
    plt.xlabel(r'$M_{\rm UV}$')
    plt.ylabel(r'$W_{\rm Ly\alpha}$')
    plt.legend()
    plt.show()

    MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, fescA, \
        fescA_err, fescB, fescB_err, ID = get_tang_data()
    
    # ew_lya_int = ew_lya/fescA
    # ew_lya_int_err = (ew_lya_err/ew_lya + fescA_err/fescA)*ew_lya_int
    
    ew_lya_int = ew_lya/fescB
    ew_lya_int_err = (ew_lya_err/ew_lya + fescB_err/fescB)*ew_lya_int

    plt.errorbar(MUV, ew_lya_int, xerr=MUV_err, yerr=ew_lya_int_err, fmt='o', color='white')
    plt.xlabel(r'$M_{\rm UV}$')
    plt.ylabel(r'$W_{\rm Ly\alpha, int}$')
    plt.show()

    lum_dens_uv = 10**(-0.4*(MUV - 51.6))
    nu_uv = (c/(1500*u.Angstrom)).to('Hz').value
    sfr = lum_dens_uv*1.15e-28#/3.1557e7
    # convert from luminosity to equivalent width
    beta = get_beta_bouwens14(MUV)
    l_lya = 1215.67 #* (1 + redshift)

    lum_lya_max_caseB = ew_lya_int * lum_dens_uv * (1215.67/1500)**(-2.3 + 2) / (l_lya/nu_uv)
    lum_ha = lum_lya_max_caseB / 11.4
    lum_dens_uv_err = np.abs(10**(-0.4*(MUV + MUV_err - 51.6)) - lum_dens_uv)
    lum_ha_err = lum_ha * (ew_lya_int_err/ew_lya_int + lum_dens_uv_err/lum_dens_uv)
    sfr_err = lum_dens_uv_err*1.15e-28

    flux_ha = lum_ha/(4*np.pi*(Planck18.luminosity_distance(z).to('cm').value)**2)
    print(f"Min Halpha Flux: {flux_ha.min()} erg s^-1 cm^-2")
    print(f"Mean Halpha Flux: {flux_ha.mean()} erg s^-1 cm^-2")

    # Create a model and RealData object
    model = odr.Model(linear_func)
    data = odr.RealData(sfr, lum_ha, sx=sfr_err, sy=lum_ha_err)

    # Set up and run the ODR regression
    odr_obj = odr.ODR(data, model, beta0=[1])  # Initial guesses for m and c
    output = odr_obj.run()

    # Extract fitted parameters and their standard errors
    params = output.beta
    param_errors = output.sd_beta

    print(f"Slope: {params[0]} ± {param_errors[0]}")

    plt.errorbar(sfr, lum_ha, xerr=sfr_err, yerr=lum_ha_err, fmt='o', color='white')
    sfr_domain = np.linspace(0, 18, 100)
    plt.plot(sfr_domain, 1.27e41*sfr_domain, linestyle='-', color='cyan')
    plt.plot(sfr_domain, params[0]*sfr_domain, linestyle='-', color='lime')
    # plt.plot(sfr_domain, (params[0]+param_errors[0])*sfr_domain, linestyle='--', color='lime')
    # plt.plot(sfr_domain, (params[0]-param_errors[0])*sfr_domain, linestyle='--', color='lime')
    plt.xlabel(r'SFR [M$_{\odot}$ yr$^{-1}$]')
    plt.ylabel(r'$L_{\rm H\alpha}$ [erg s$^{-1}$]')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    residuals = np.abs(np.log10(lum_ha) - np.log10(params[0]*sfr))
    res_mean = residuals.mean()
    print(params[0], res_mean)

    lognorm_sample = np.random.lognormal(np.log(params[0]), 0.6, 1000)
    muv_sample = np.random.uniform(-22, -16, 1000)
    lum_ha_sample = 10**(-0.4*(muv_sample - 51.6))*1.15e-28*lognorm_sample

    # residuals = (lum_ha - params[0]*sfr)/params[0]*sfr
    # plt.hist(residuals, 100)
    # plt.show()
    # quit()

    muv_domain = np.linspace(-23, -16, 100)
    sfr_domain = 10**(-0.4*(muv_domain - 51.6))*1.15e-28
    fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    plt.plot(muv_sample, lum_ha_sample, '.', color='white', alpha=0.5)
    plt.errorbar(MUV, lum_ha, xerr=MUV_err, yerr=lum_ha_err, fmt='o', color='red')
    plt.plot(muv_domain, 1.27e41*sfr_domain, linestyle='-', color='cyan')
    plt.plot(muv_domain, params[0]*sfr_domain, linestyle='-', color='lime')
    plt.xlabel(r'$M_{\rm UV}$')
    plt.ylabel(r'$L_{\rm H\alpha}$ [erg s$^{-1}$]')
    # plt.xscale('log')
    plt.yscale('log')
    plt.show()
    # plt.savefig('/mnt/c/Users/sgagn/Downloads/lum_ha_vs_muv.pdf', dpi=1000)