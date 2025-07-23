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

from scipy.optimize import curve_fit

def exponential(x, a, b):
    """
    Exponential function.
    """
    return a * np.exp(-b * x)

x = [0.05490184110753669, 0.1019606646369486, 0.14901948816636032, 0.20196066463694856, 0.2490194881663603, 0.29607831169577203, 0.3450979793772978, 0.3450979793772978, 0.3980391558478861, 0.44705882352941184, 0.496078311695772, 0.5450979793772978, 0.8490194881663602, 0.8490194881663602, 1.043137135225184, 0.598039155847886, 0.7450979793772977]
y = [0.7526880079700098, 1.005376262049521, 1.010752524099042, 0.758064516129032, 0.8817202660345261, 0.6182794878559728, 0.6236557499054933, 0.6236557499054933, 0.4892472297914564, 0.1290322580645159, 0.4946234918409774, 0.11290322580645151, 0.10215020948840692, 0.10215020948840692, 0.09677419354838668, 0.4892472297914564, 0.23118246755292327]
x = np.array(x)
y = np.array(y)
y /= np.sum(y)
print(np.sum(x*y))
quit()

lognorm_sample = np.random.lognormal(mean=-1.0, sigma=1.0, size=1000)

hist, bins = np.histogram(lognorm_sample, bins=30, density=True)
popt, _ = curve_fit(exponential, bins[:-1], hist)

print(popt)
plt.plot(bins[:-1], hist, 'o', label='Data')
plt.plot(bins[:-1], exponential(bins[:-1], *popt), '-', label='Fit')
plt.legend()
plt.show()