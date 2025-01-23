import numpy as np
from astropy.io import fits

vandels_cont = fits.getdata('./data/vandels_continuum.fit')

vandels_dict = {}
for item in vandels_cont:
    vandels_dict[item[0]] = {}
    vandels_dict[item[0]]['z'] = item[1]
    vandels_dict[item[0]]['type'] = item[2]
    vandels_dict[item[0]]['Flya'] = item[3]
    vandels_dict[item[0]]['Flya_err'] = item[4]

vandels_ew = fits.getdata('./data/vandels_ew.fit')

for item in vandels_ew:
    vandels_dict[item[0]]['EW_lya'] = item[2]
    vandels_dict[item[0]]['EW_lya_err'] = item[3]

vandels_data = np.zeros((len(vandels_dict), 5))

for i, item in enumerate(vandels_dict):
    vandels_data[i] = np.array([vandels_dict[item]['z'], vandels_dict[item]['Flya'], vandels_dict[item]['Flya_err'], vandels_dict[item]['EW_lya'], vandels_dict[item]['EW_lya_err']])
    
print(vandels_data)
np.save('data/vandels.npy', vandels_data)