import numpy as np

import matplotlib.pyplot as plt
rc = {"font.family" : "serif", 
    "mathtext.fontset" : "stix"}
plt.rcParams.update(rc) 
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({'font.size': 14})
import matplotlib as mpl
label_size = 20
font_size = 30
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size
presentation = True
# presentation = False
if presentation:
    plt.style.use('dark_background')
    cmap = 'Blues_r'
else:
    cmap = 'hot_r'

def get_muv(sfr):
    """
    Converts star formation rate to
    absolute UV magnitude.
    """
    kappa = 1.15e-28
    luv = sfr * 3.1557e7 / kappa
    muv = 51.64 - np.log10(luv) / 0.4
    return muv

_, _, _, halo_masses, _, sfr = np.load('../data/halo_fields/halo_field_5.76.npy')
muv = get_muv(sfr)

halo_masses = halo_masses[muv < -16]
muv = muv[muv < -16]

hist, side1, side2 = np.histogram2d(muv, np.log10(halo_masses), bins=[50, 50])
space1 = 0.5*(side1[1:] + side1[:-1])
space2 = 0.5*(side2[1:] + side2[:-1])
mean_arr = []
lower_arr = []
upper_arr = []
for i in range(hist.shape[0]):
    weights = hist[i] / np.sum(hist[i])
    weight_dist = np.cumsum(weights)
    mean = space2[np.argmin(np.abs(weight_dist - 0.5))]
    lower_lim = space2[np.argmin(np.abs(weight_dist - 0.32))]
    upper_lim = space2[np.argmin(np.abs(weight_dist - 0.68))]
    mean_arr.append(mean)
    lower_arr.append(lower_lim)
    upper_arr.append(upper_lim)

mean_arr = np.array(mean_arr)
lower_arr = np.array(lower_arr)
upper_arr = np.array(upper_arr)

su = np.mean(upper_arr-mean_arr)
sl = np.mean(mean_arr-lower_arr)

from scipy.optimize import curve_fit
def linear_fit(x, a, b):
    return a * (x + 18.5) + b

print(mean_arr)
select = space1 > -18.0
popt, _ = curve_fit(linear_fit, space1[select], mean_arr[select])
print(popt, sl, su)

plt.hist2d(muv, np.log10(halo_masses), bins=[50, 50], cmap=cmap)
plt.plot(space1, linear_fit(space1, *popt), color='red', label='Linear Fit')
plt.fill_between(space1, linear_fit(space1, *popt)+su, \
                 linear_fit(space1, *popt)-sl, alpha=0.2, color='red', label='Linear Fit')
# plt.scatter(muv, np.log10(halo_masses), s=1, alpha=0.1, color='cyan')
plt.xlabel('MUV')
plt.ylabel('Log10 Halo Mass')
plt.title('MUV vs Log10 Halo Mass')
plt.show()