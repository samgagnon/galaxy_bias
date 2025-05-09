import numpy as np

from scipy import odr

from astropy.cosmology import Planck18
from astropy import units as u
from astropy.constants import c, G, m_p, k_B

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
    linecolor = 'orange'
    datacolor = 'red'

    # measured lya properties from https://arxiv.org/pdf/2402.06070
    MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, \
        fescA, fescA_err, fescB, fescB_err, ID = get_tang_data()
    
    # get dv ew relation
    model = odr.Model(linear_func)
    ew_logerr = ew_lya_err/ew_lya
    data = odr.RealData(dv_lya, np.log10(ew_lya), sx=dv_lya_err, sy=ew_logerr)
    odr_obj = odr.ODR(data, model, beta0=[1., 0.])  
    out = odr_obj.run()
    m, b = out.beta
    m1, b1 = m, b
    cov = out.cov_beta

    print('logW(dV)', m, b)
    residuals = np.abs(np.log10(ew_lya) - linear_func([m, b], dv_lya))
    print('+/-', np.mean(residuals))

    # get muv dv relation
    model = odr.Model(linear_func)
    data = odr.RealData(MUV, dv_lya, sx=MUV_err, sy=dv_lya_err)
    odr_obj = odr.ODR(data, model, beta0=[1., 0.])
    out = odr_obj.run()
    m, b = out.beta
    m2, b2 = m, b
    covar = out.cov_beta

    print('logdV(muv)', m, b)
    residuals = np.abs(dv_lya - linear_func([m, b], MUV))
    print('+/-', np.mean(residuals))
    # print(covar)

    # compute theoretical maximum LyA emission
    lum_dens_uv = 10**(-0.4*(MUV - 51.6))
    nu_uv = (c/(1500*u.Angstrom)).to('Hz').value
    sfr = lum_dens_uv*1.15e-28#/3.1557e7
    # convert from luminosity to equivalent width
    beta = get_beta_bouwens14(MUV)
    l_lya = 1215.67 #* (1 + redshift)

    ew_lya_int = ew_lya/fescB
    ew_lya_int_err = (ew_lya_err/ew_lya + fescB_err/fescB)*ew_lya_int

    lum_lya_max_caseB = ew_lya_int * lum_dens_uv * (1215.67/1500)**(-2.3 + 2) / (l_lya/nu_uv)
    lum_ha = lum_lya_max_caseB / 11.4
    lum_dens_uv_err = np.abs(10**(-0.4*(MUV + MUV_err - 51.6)) - lum_dens_uv)
    lum_ha_err = lum_ha * (ew_lya_int_err/ew_lya_int + lum_dens_uv_err/lum_dens_uv)
    sfr_err = lum_dens_uv_err*1.15e-28

    # get lumha sfr relation
    model = odr.Model(linear_func_b0)
    data = odr.RealData(sfr, lum_ha, sx=sfr_err, sy=lum_ha_err)
    odr_obj = odr.ODR(data, model, beta0=[1.])  # Initial guesses for m and c
    out = odr_obj.run()
    m = out.beta
    m_err = out.sd_beta
    m3 = m

    print('logL(Ha)', m)
    std = np.std(lum_ha/sfr)
    print('std', std)


    fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # plot the data
    dv_lya_space = np.linspace(0, 1000, 100)
    axs[0].errorbar(dv_lya, np.log10(ew_lya), xerr=dv_lya_err, yerr=ew_logerr, fmt='o', markersize=10, capsize=5, color=datacolor)
    axs[0].plot(dv_lya_space, linear_func([m1, b1], dv_lya_space), color=linecolor)
    axs[0].set_xlabel(r'$\Delta v$ [km s$^{-1}$]', fontsize=font_size)
    axs[0].set_ylabel(r'$\log_{10}{\rm W}_{\rm emerg}$ [$\AA$]', fontsize=font_size)
    axs[0].set_xlim(0, 800)
    axs[0].set_ylim(0.5, 3)

    # plot the data
    muv_space = np.linspace(-22, -16.5, 100)
    axs[1].errorbar(MUV, dv_lya, xerr=MUV_err, yerr=dv_lya_err, fmt='o', markersize=10, capsize=5, color=datacolor)
    axs[1].plot(muv_space, linear_func([m2, b2], muv_space), color=linecolor, label='TANG LAEs')
    axs[1].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
    axs[1].set_ylabel(r'$\Delta v$ [km s$^{-1}$]', fontsize=font_size)
    axs[1].set_xlim(-22, -16.5)

    # plot the data
    sfr_space = np.linspace(0, 18, 100)
    axs[2].errorbar(sfr, lum_ha/1e42, xerr=sfr_err, yerr=lum_ha_err/1e42, fmt='o', markersize=10, capsize=5, color=datacolor, label='Tang et al. (2024)')
    axs[2].plot(sfr_space, linear_func_b0([m3], sfr_space)/1e42, color=linecolor, label='best fit')
    axs[2].plot(sfr_space, sfr_space*1.27e41/1e42, linestyle='--', color='orange', label='Kennicut (1998)')
    axs[2].set_xlabel(r'SFR [M$_{\odot}$ yr$^{-1}$]', fontsize=font_size)
    axs[2].set_ylabel(r'$L_{\rm H\alpha}$ [$10^{42}$ erg s$^{-1}$]', fontsize=font_size)
    axs[2].set_xlim(0, 6)
    axs[2].set_ylim(0, 5)

    axs[2].legend(fontsize=18)
    
    plt.show()