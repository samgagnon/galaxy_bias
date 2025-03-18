import numpy as np

from astropy.io import fits
from astropy import units as u
from astropy.constants import c
from astropy.cosmology import Planck18

from scipy.integrate import trapezoid
from scipy.special import erf

band_wavelengths = {
    'g': 4770,
    'r': 6231,
    'i': 7625,
    'z': 9134,
    'y': 10315,
}

band_frequencies = {
    'g': (c/(band_wavelengths['g']*u.Angstrom)).to('Hz').value,
    'r': (c/(band_wavelengths['r']*u.Angstrom)).to('Hz').value,
    'i': (c/(band_wavelengths['i']*u.Angstrom)).to('Hz').value,
    'z': (c/(band_wavelengths['z']*u.Angstrom)).to('Hz').value,
    'y': (c/(band_wavelengths['y']*u.Angstrom)).to('Hz').value,
}

lbda_lya = 1215.67*u.Angstrom
freq_lya = (c/lbda_lya).to('Hz')

def get_rel_key(label):
    if label == 'NB816':
        return 'i'
    elif label == 'NB921':
        return 'z'
    elif label == 'NB973':
        return 'y'
    else:
        raise ValueError("Label not recognized")

def band_sensitivities():
    glambda, gband = np.loadtxt("./data/hsc_bands/gband.csv", delimiter=',').T
    rlambda, rband = np.loadtxt("./data/hsc_bands/rband.csv", delimiter=',').T
    ilambda, iband = np.loadtxt("./data/hsc_bands/iband.csv", delimiter=',').T
    zlambda, zband = np.loadtxt("./data/hsc_bands/zband.csv", delimiter=',').T
    ylambda, yband = np.loadtxt("./data/hsc_bands/yband.csv", delimiter=',').T
    transmission = {}
    for lbda, band, key in zip([glambda, rlambda, ilambda, zlambda, ylambda], \
                          [gband, rband, iband, zband, yband],\
                            ['g', 'r', 'i', 'z', 'y']):
        range_band = np.linspace(lbda.min(), lbda.max(), 1000)
        band_interp = np.interp(range_band, lbda, band)
        transmission[key] = np.array([range_band, band_interp])
    return transmission

def get_a_star(band_key, transmission, lbda_nb):
    band_lbda, band_sensitivity = transmission[band_key]
    band_sensitivity = band_sensitivity[::-1]
    band_freq = (c/(band_lbda[::-1]*u.Angstrom)).to('Hz').value
    flya_r = (c/(lbda_nb*u.Angstrom)).to('Hz').value

    numerator = np.interp(flya_r, band_freq, band_sensitivity)/flya_r
    denominator = trapezoid(band_sensitivity/band_freq, band_freq)
    return numerator/denominator

def get_c_star(band_key, transmission, lbda_nb):
    band_lbda, band_sensitivity = transmission[band_key]
    band_sensitivity = band_sensitivity[::-1]
    band_freq = ((c/(band_lbda[::-1]*u.Angstrom)).to('Hz')).value
    flya_r = (c/(lbda_nb*u.Angstrom)).to('Hz').value

    numerator = trapezoid(band_sensitivity[band_freq<flya_r]/flya_r, \
                          band_freq[band_freq<flya_r])
    denominator = trapezoid(band_sensitivity/band_freq, band_freq)
    return numerator/denominator

def get_f_star(band_mab):
    return 10**(-0.4*(band_mab + 48.6))

def get_ew_unitless(an, cn, fn, ab, cb, fb):
    return (fn/cn - fb/cb)/(an/cn - ab/cb)

def get_ew(redshift, mab_broad, mab_narrow, lbda_narrow, band_key, transmission):
    # TODO still wrong, something strange about the units in Equation 8 of https://arxiv.org/pdf/2411.15495

    l_bb, t_bb = transmission[band_key]
    t_bb = t_bb[::-1]
    nu_bb = (c/(l_bb[::-1]*u.Angstrom)).to('Hz').value
    nu_nb = (c/(lbda_narrow*u.Angstrom)).to('Hz').value
    t_nb_max = np.interp(nu_nb, nu_bb, t_bb)
    t_nb = t_nb_max*np.exp(-0.5*(l_bb - lbda_narrow)**2/(25**2))
    t_nb = t_nb[::-1]

    # MEASURED INTENSITIES
    # erg s-1 cm-2 Hz-1
    f_nb = 10**(-0.4*(mab_narrow + 48.6))
    f_bb = 10**(-0.4*(mab_broad + 48.6))

    # print(f_nb, f_bb)

    # TRANSMISSION FACTORS
    a_nb = (np.interp(nu_nb, nu_bb, t_bb)/nu_nb) \
        / trapezoid(t_bb/nu_bb, nu_bb)
    a_bb = (np.interp(nu_nb, nu_bb, t_nb)/nu_nb) \
        / trapezoid(t_nb/nu_bb, nu_bb)

    # REDWARD EMISSION FRACTION
    c_nb = trapezoid(t_nb[nu_bb<nu_nb]/nu_nb, nu_bb[nu_bb<nu_nb]) \
        / trapezoid(t_nb/nu_bb, nu_bb)
    c_bb = trapezoid(t_bb[nu_bb<nu_nb]/nu_nb, nu_bb[nu_bb<nu_nb]) \
        / trapezoid(t_bb/nu_bb, nu_bb)

    # BLUEWARD EMISSION FRACTION
    # d_nb = trapezoid(t_nb[nu_bb>nu_nb]/nu_nb, nu_bb[nu_bb>nu_nb]) \
    #     / trapezoid(t_nb/nu_bb, nu_bb)
    d_bb = trapezoid(t_bb[nu_bb>nu_nb]/nu_nb, nu_bb[nu_bb>nu_nb]) \
        / trapezoid(t_bb/nu_bb, nu_bb)
    
    f_nb_red = f_nb*c_nb/a_nb
    f_bb_red = f_bb*c_bb/a_bb

    # print(f_nb, c_nb, d_nb, a_nb)
    # print(f_bb, c_bb, d_bb, a_bb)

    # m_nb_red = -2.5*np.log10(f_nb_red/nu_nb) - 48.6
    # m_bb_red = -2.5*np.log10(f_bb_red/nu_nb) - 48.6

    f_alpha = f_nb_red - f_bb_red

    f_continuum = f_bb_red*(d_bb/c_bb)

    f_1500 = (f_continuum/nu_nb) * (1500/lbda_narrow)**-2
    l_1500 = f_1500 * 4*np.pi*(Planck18.luminosity_distance(redshift).to('cm').value)**2

    muv = 51.6 - 2.5*np.log10(l_1500)
    
    return f_alpha/f_continuum, muv

def get_muv(redshift, g, r, i, z, y):
    # observation-like selection criterion for LyA
    beta = -2
    muv_list = []
    for band, ID in zip([g, r, i, z, y], ['g', 'r', 'i', 'z', 'y']):
        if band == 0:
            muv_list.append(np.nan)
        # band_frequency = band_frequencies[ID] * (1 + redshift)
        # band_wavelength = band_wavelengths[ID] / (1 + redshift)
        # define slope adjustment constant
        # units_constant = (band_frequency / band_wavelength) * (band_wavelength/ 1500) ** (-(beta) - 2)
        # print(band_wavelength)
        intensity_band = 10**(-0.4*(band + 48.6))
        lum_density_band = intensity_band * 4*np.pi*(Planck18.luminosity_distance(redshift).to('cm').value)**2
        lum_density_uv = lum_density_band# * (band_wavelength / 1500)**(beta)
        muv = 51.6 - 2.5*np.log10(lum_density_uv)
        muv_list.append(muv)
    return np.asarray(muv_list)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc) 
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')

    transmission = band_sensitivities()

    # plt.plot(transmission['i'][0], transmission['i'][1])
    # plt.plot(transmission['i'][0], np.exp(-0.5*(transmission['i'][0] - 8160)**2/(25**2)))
    # plt.axvline(1215*(1 + 5))
    # plt.show()
    # quit()

    # Read the table
    with fits.open("data/silverrushxiii.fit") as hdul:
        data = hdul[1].data
        nb_labels = []
        ew_list = []
        muv_list = []
        redshift_list = []
        for row in data:
            redshift, g, r, i, z, y, NB, label = row[0], row[1], row[2], \
                row[3], row[4], row[5], row[6], row[7]
            bands = {'g': g, 'r': r, 'i': i, 'z': z, 'y': y}
            if redshift > 5:
                key = get_rel_key(label)
                ew, muv = get_ew(redshift, bands[key], NB, float(label[2:])*10, key, transmission)
                ew_list.append(ew)
                redshift_list.append(redshift)
                muv_list.append(muv)

        ew_list = np.asarray(ew_list)
        redshift_list = np.asarray(redshift_list)
        muv_list = np.asarray(muv_list)

        plt.scatter(muv_list, ew_list, s=10, marker='o', c=redshift_list, cmap='BuPu')
        plt.xlabel(r'$M_{\rm UV}$')
        plt.ylabel("EW")
        plt.show()
        quit()
                