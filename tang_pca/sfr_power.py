
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

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

# Define the expression whose roots we want to find

# exp = 1/0.88
exp = 0.88*2

def func(x, tau):
    alpha = x
    numerator = 1 + (tau/10)**alpha
    denominator = (1 + (tau/100)**alpha)**exp
    # ensure that the integral equals one
    x = np.linspace(0.01, 10000, 1000)
    integral = np.trapz(1 / (1 + (tau/x)**alpha), x)
    sigma = np.sqrt(1/integral) * tau
    # print(sigma, alpha)
    return (numerator / denominator - sigma**(2 - 2*exp))**2

load = True

if load == False:
    # Use the numerical solver to find the roots
    tau_decor_list = np.linspace(15, 90, 100)
    alpha_list = []
    sigma_list = []
    p10_list = []
    p100_list = []
    for tau in tau_decor_list:
        result = differential_evolution(func, args=(tau,), bounds=[(0, 15)], \
                                        maxiter=500, mutation=(0.1, 1.9),\
                                        popsize=200, disp=True, recombination=0.5)
        alpha = result.x
        alpha_list.append(alpha)
        x = np.linspace(0.01, 10000, 1000)
        integral = np.trapz(1 / (1 + (tau/x)**alpha), x)
        sigma = np.sqrt(1/integral) * tau

        P10 = sigma**2 / (1 + (tau/10)**alpha)
        P100 = sigma**2 / (1 + (tau/100)**alpha)
        p10_list.append(P10)
        p100_list.append(P100)

        sigma_list.append(sigma)
    alpha_list = np.array(alpha_list)
    sigma_list = np.array(sigma_list)
    p100_list = np.array(p100_list)
    p10_list = np.array(p10_list)

else:
    alpha_list = np.load('../data/alpha.npy')
    tau_decor_list = np.load('../data/tau.npy')

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)

axs.plot(tau_decor_list, alpha_list, '-', linewidth=4, color='black')
axs.set_xlabel(r'$\tau_{\rm decor}$ [Myr]', fontsize=font_size)
axs.set_ylabel(r'$\alpha$', fontsize=font_size)
axs.set_xlim(16, 90)
# print("tau_decor_list:", tau_decor_list)
# print("alpha_list:", alpha_list.squeeze())
# print("sigma_list:", sigma_list)
# print("p10_list:", p10_list - p100_list**0.88)
# plt.yscale('log')
plt.savefig("/mnt/c/Users/sgagn/Documents/phd/lyman_alpha/figures/burst_params.pdf", dpi=300)
if load == False:
    np.save('../data/tau.npy', tau_decor_list)
    np.save('../data/alpha.npy', alpha_list)
    np.save('../data/sigma.npy', sigma_list)
    np.save('../data/p10.npy', p10_list)
    np.save('../data/p100.npy', p100_list)
plt.show()