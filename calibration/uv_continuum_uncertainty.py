import numpy as np

from astropy.cosmology import Planck18
from astropy import units as u
from astropy.constants import c, G, m_p, k_B

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

    muv_space = np.linspace(-24, -16, 100)
    muv_sigma = 0.1

    MUV, MUV_err, _, ew_lya, ew_lya_err, _, _, \
        _, _, _, _, ID = get_tang_data()
    
    # muv_err = MUV_err[ID==0].mean()

    w_min_16_01 = 1215.67*((10**(0.4*0.01) - 1)) / (1215.67/1500)**(0.2*(muv_space + 19.5) + 0.05)
    w_min_16_1 = 1215.67*((10**(0.4*0.09) - 1)) / (1215.67/1500)**(0.2*(muv_space + 19.5) + 0.05)
    # print(w_min_16_01, w_min_16_1)
    # quit()

    w_min1 = 1215.67*((10**(0.4*MUV_err[ID==0]) - 1)) / (1215.67/1500)**(0.2*(MUV[ID==0] + 19.5) + 0.05)
    w_min2 = 1215.67*((10**(0.4*MUV_err[ID==1]) - 1)) / (1215.67/1500)**(0.2*(MUV[ID==1] + 19.5) + 0.05)
    w_min3 = 1215.67*((10**(0.4*MUV_err[ID>1]) - 1)) / (1215.67/1500)**(0.2*(MUV[ID>1] + 19.5) + 0.05)

    # plt.plot(MUV_err[ID==0], w_min1, '.', color='cyan')
    # plt.plot(MUV_err[ID==1], w_min2, 'x', color='cyan')
    # plt.plot(MUV_err[ID>1], w_min3, '*', color='cyan')

    plt.plot(MUV[ID==0], ew_lya[ID==0], '.', color='cyan')
    plt.plot(MUV[ID==1], ew_lya[ID==1], 'x', color='lime')
    plt.plot(muv_space, w_min_16_01, color='lime', linestyle='-')
    plt.plot(muv_space, w_min_16_1, color='cyan', linestyle='-')
    # plt.plot(MUV_err[ID>1], ew_lya[ID>1], '*', color='lime')

    plt.yscale('log')
    # plt.xscale('log')
    plt.show()