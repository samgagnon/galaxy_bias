import numpy as np
from hmf import MassFunction

import matplotlib.pyplot as plt
rc = {"font.family" : "serif", 
    "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({'font.size': 14})
plt.style.use('dark_background')

modes = ['Jenkins', 'ST', 'PS']

dat_dir = './data/'

for mode in modes:

    hmf = MassFunction(z=6.0, Mmin=8, Mmax=15, dlog10m=0.1, hmf_model=mode)
    np.save(f'{dat_dir}hmf_m_{mode}.npy', hmf.m)
    np.save(f'{dat_dir}hmf_dndm_{mode}.npy', hmf.dndm)

    plt.plot(hmf.m, hmf.dndm)
plt.yscale('log')
plt.xscale('log')
plt.show()