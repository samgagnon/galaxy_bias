import numpy as np

# BOUWENS 2021
b21_mag = [[-22.52, -22.02, -21.52, -21.02, -20.52, -20.02, -19.52, -18.77, -17.77, -16.77],
          [-22.19, -21.69, -21.19, -20.68, -20.19, -19.69, -19.19, -18.69, -17.94, -16.94],
          [-21.85, -21.35, -20.85, -20.10, -19.35, -18.6, -17.6]]
b21_phi = [[2e-6, 1.4e-5, 5.1e-5, 1.69e-4, 3.17e-4, 7.24e-4, 1.124e-3, 2.82e-3, 8.36e-3, 1.71e-2],
          [1e-6, 4.1e-5, 4.7e-5, 1.98e-4, 2.83e-4, 5.89e-4, 1.172e-3, 1.433e-3, 5.76e-3, 8.32e-3],
          [3e-6, 1.2e-5, 4.1e-5, 1.2e-4, 6.57e-4, 1.1e-3, 3.02e-3]]
b21_phi_err = [[2e-6, 5e-6, 1.1e-5, 2.4e-5, 4.1e-5, 8.7e-5, 1.57e-4, 4.4e-4, 1.66e-3, 5.26e-3],
              [2e-6, 1.1e-5, 1.5e-5, 3.6e-5, 6.6e-5, 1.26e-4, 3.36e-4, 4.19e-4, 1.44e-3, 2.9e-3],
              [2e-6, 4e-6, 1.1e-5, 4e-5, 2.33e-4, 3.4e-4, 1.14e-3]]

b21_6 = np.array(b21_phi[0])
b21_7 = np.array(b21_phi[1])
b21_8 = np.array(b21_phi[2])

b21_6_err = np.array(b21_phi_err[0])
b21_7_err = np.array(b21_phi_err[1])
b21_8_err = np.array(b21_phi_err[2])

logphi_b21_6 = np.log10(b21_6)
logphi_b21_7 = np.log10(b21_7)
logphi_b21_8 = np.log10(b21_8)

logphi_err_b21_6_up = np.log10(b21_6 + b21_6_err) - logphi_b21_6
logphi_err_b21_7_up = np.log10(b21_7 + b21_7_err) - logphi_b21_7
logphi_err_b21_8_up = np.log10(b21_8 + b21_8_err) - logphi_b21_8

logphi_err_b21_6_low = logphi_b21_6 - np.log10(b21_6 - b21_6_err)
logphi_err_b21_7_low = logphi_b21_7 - np.log10(b21_7 - b21_7_err)
logphi_err_b21_8_low = logphi_b21_8 - np.log10(b21_8 - b21_8_err)

logphi_err_b21_6_low[np.isinf(logphi_err_b21_6_low)] = np.abs(logphi_b21_6[np.isinf(logphi_err_b21_6_low)])
logphi_err_b21_7_low[np.isinf(logphi_err_b21_7_low)] = np.abs(logphi_b21_7[np.isinf(logphi_err_b21_7_low)])
logphi_err_b21_8_low[np.isinf(logphi_err_b21_8_low)] = np.abs(logphi_b21_8[np.isinf(logphi_err_b21_8_low)])

logphi_err_b21_6_low[np.isnan(logphi_err_b21_6_low)] = np.abs(logphi_b21_6[np.isnan(logphi_err_b21_6_low)])
logphi_err_b21_7_low[np.isnan(logphi_err_b21_7_low)] = np.abs(logphi_b21_7[np.isnan(logphi_err_b21_7_low)])
logphi_err_b21_8_low[np.isnan(logphi_err_b21_8_low)] = np.abs(logphi_b21_8[np.isnan(logphi_err_b21_8_low)])

def get_silverrush_laelf(z):
    if z==4.9:
        # SILVERRUSH XIV z=4.9 LAELF
        lum_silver = np.array([42.75, 42.85, 42.95, 43.05, 43.15, 43.25, 43.35, 43.45, 43.55, 43.65])
        logphi_silver = -1*np.array([2.91, 3.17, 3.42, 3.78, 3.88, 4.00, 4.75, 4.93, 5.23, 4.93])
        logphi_up_silver = 1e-2*np.array([5, 5, 6, 9, 10, 12, 29, 36, 52, 36])
        logphi_low_silver = 1e-2*np.array([5, 5, 6, 9, 10, 12, 34, 45, 77, 45])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    elif z==5.7:
        # SILVERRUSH XIV z=5.7 LAELF
        lum_silver = np.array([42.85, 42.95, 43.05, 43.15, 43.25, 43.35, 43.45, 43.55, 43.65, 43.75, 43.85, 43.95])
        logphi_silver = -1*np.array([3.05, 3.27, 3.56, 3.85, 4.15, 4.41, 4.72, 5.15, 5.43, 6.03, 6.33, 6.33])
        logphi_up_silver = 1e-2*np.array([4, 2, 2, 3, 4, 5, 7, 12, 17, 36, 52, 52])
        logphi_low_silver = 1e-2*np.array([4, 2, 2, 3, 4, 5, 7, 13, 18, 45, 77, 77])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    elif z==6.6:
        # SILVERRUSH XIV z=6.6 LAELF
        lum_silver = np.array([42.95, 43.05, 43.15, 43.25, 43.35, 43.45, 43.55, 43.65, 43.75, 43.95, 44.05])
        logphi_silver = -1*np.array([3.71, 4.11, 4.37, 4.65, 4.83, 5.28, 5.89, 5.9, 5.9, 6.38, 6.38])
        logphi_up_silver = 1e-2*np.array([9, 5, 6, 7, 8, 14, 29, 29, 29, 52, 52])
        logphi_low_silver = 1e-2*np.array([9, 5, 6, 7, 8, 15, 34, 34, 34, 77, 77])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    elif z==7.0:
        # wip
        # SILVERRUSH XIV z=7.0 LAELF
        lum_silver = np.array([43.25, 43.35])
        logphi_silver = -1*np.array([4.4, 4.95])
        logphi_up_silver = 1e-2*np.array([29, 52])
        logphi_low_silver = 1e-2*np.array([34, 77])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver
    elif z==7.3:
        # wip
        # SILVERRUSH XIV z=7.3 LAELF
        lum_silver = np.array([43.45])
        logphi_silver = -1*np.array([4.81])
        logphi_up_silver = 1e-2*np.array([36])
        logphi_low_silver = 1e-2*np.array([45])
        return lum_silver, logphi_silver, logphi_up_silver, logphi_low_silver

def get_silverrush_ewpdf(z):
    assert z in [4.9, 5.7, 6.6]
    if z==4.9:
        ew0_low = 38.5
        ew0_low_upper = 2.2
        ew0_low_lower = 2.0
        ew0_high = 90.3
        ew0_high_upper = 1.9
        ew0_high_lower = 1.7
    elif z==5.7:
        ew0_low = 32.9
        ew0_low_upper = 0.8
        ew0_low_lower = 0.7
        ew0_high = 76.0
        ew0_high_upper = 0.7
        ew0_high_lower = 0.6
    elif z==6.6:
        ew0_low = 52.3
        ew0_low_upper = 6.5
        ew0_low_lower = 5.3
        ew0_high = 114.5
        ew0_high_upper = 5.6
        ew0_high_lower = 4.9
    return ew0_low, ew0_low_upper, ew0_low_lower, \
        ew0_high, ew0_high_upper, ew0_high_lower

def get_silverrush_clustering(z):
    assert z in [4.9, 5.7, 6.6]
    # r0 [1/h70 1/Mpc^3]
    # gamma is the slope of the correlation function
    # I would rather fit directly to data, but I'll make do 
    # wtih these values for now
    if z==4.9:
        r0 = 6.4
        r0_upper = 0.6
        r0_lower = 0.7
        gamma = 1.97
        gamma_upper = 0.16
        gamma_lower = 0.16
    elif z==5.7:
        r0 = 4.0
        r0_upper = 0.6
        r0_lower = 0.7
        gamma = 1.02
        gamma_upper = 0.17
        gamma_lower = 0.17
    elif z==6.6:
        r0 = 8.0
        r0_upper = 1.9
        r0_lower = 5.8
        gamma = 1.4
        gamma_upper = 0.58
        gamma_lower = 0.88
    return r0, r0_upper, r0_lower, \
        gamma, gamma_upper, gamma_lower

# Konno+2018 referece LAELF
# we need to replace this with the SILVERRUSH XIV LAELFs
# https://arxiv.org/abs/1705.01222
lum_konno = np.array([43.15, 43.25, 43.35, 43.45, \
                    43.55, 43.65, 43.75, 43.85])
logphi_konno = -1*np.array([4.194, 4.407, 4.748, \
                    5.132, 5.433, 5.609, 6.212, 6.226])
logphi_up_konno = 1e-3*np.array([154, 101, 87, 140, \
                    203, 253, 519, 519])
logphi_low_konno = 1e-3*np.array([317, 258, 243, 300, \
                    374, 438, 917, 917])

