import numpy as np

from astropy.cosmology import Planck18
from astropy import units as u
from astropy.constants import c, G
from astropy.io import fits

if __name__ == "__main__":

    id_list = []
    redshift_list = []
    redshift_err_list = []
    w_list = []
    w_err_list = []
    muv_list = []
    muv_err_list = []
    logllya_list = []
    logllya_err_list = []
    peaksep_list = []
    peaksep_err_list = []
    fwhm_list = []
    fwhm_err_list = []
    assym_list = []
    assym_err_list = []


    with fits.open('../data/2202.06642.fit') as hdul:
        data = hdul[1].data
        for row in data:

            print(row[0])

            id_list.append(row[0])
            redshift_list.append(row[1])
            redshift_err_list.append(row[2])
            w_list.append(row[3])
            w_err_list.append(row[4])
            muv_list.append(row[5])
            muv_err_list.append(row[6])
            logllya_list.append(row[7])
            logllya_err_list.append(row[8])
            peaksep_list.append(row[9])
            peaksep_err_list.append(row[10])
            fwhm_list.append(row[11])
            fwhm_err_list.append(row[12])
            assym_list.append(row[13])
            assym_err_list.append(row[14])

    dat_arr = np.array([id_list, redshift_list, redshift_err_list, w_list, w_err_list, \
                        muv_list, muv_err_list, logllya_list, logllya_err_list, \
                        peaksep_list, peaksep_err_list, fwhm_list, fwhm_err_list, \
                        assym_list, assym_err_list]).T
        
    np.save('../data/muse.npy', dat_arr)