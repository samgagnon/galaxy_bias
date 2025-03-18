import os

import numpy as np

from astropy.io import fits
from astropy.constants import c

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc) 
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')

    # this contains reduced statistics of high-z lya emitting galaxies
    # with fits.open('../data/2202.06642.fit') as hdul:
            # data = hdul[1].data
            # for row in data:
            #     coordfile.append([row[0], row[1]])

    source_dict = {}
    with fits.open('../data/MW_44fields_main_table_v1.0.fits') as hdul:
        data = hdul[1].data
        for row in data:
            if row[6] == 'Lya':
                source_dict[row[0]] = np.array([row[4], row[5], row[7]])

    counter = 0

    for fn in os.listdir('../data/muse_wide_spectra/'):
        print(counter/470)
        counter += 1

        wavelength_range = []
        flux = []

        # fn = 'emission_spectrum_candels-cdfs-01_101005016.fits'
        ID = int(fn.split('_')[3].split('.')[0])
        with fits.open(f'../data/muse_wide_spectra/{fn}') as hdul:
            data = hdul[1].data
            for row in data:
                wavelength_range.append(row[0])
                flux.append(row[2])

        redshift = source_dict[ID][0]
        lya = 1215.67*(1 + redshift)
        velocity_range = (c.to('km/s')).value*(wavelength_range/lya - 1)
        plt.plot(velocity_range, flux)
    
    plt.xlabel(r'$\Delta v$ [km/s]')
    plt.ylabel(r'$f_{\lambda}(\Delta v)$')
    plt.xlim(-1000, 1000)
    plt.show()